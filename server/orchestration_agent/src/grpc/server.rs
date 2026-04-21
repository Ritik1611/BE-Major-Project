// server/orchestration_agent/src/grpc/server.rs
// Zero-trust gRPC server — filesystem/ledger backend
//
// SECURITY FIXES (all preserved):
//   FIX-SERVER-1: require_client_cert() — mTLS enforcement
//   FIX-SERVER-2: UploadUpdate streams bytes → filesystem (was GridFS)
//   FIX-SERVER-3: Per-chunk SHA-256 verification during streaming
//   FIX-SERVER-4: payload_hash in Receipt verified against stored upload
//   FIX-SERVER-5: handle validated as filesystem path within updates dir
//   FIX-SERVER-6: HMAC-chained receipt ledger for tamper-evident audit
//   FIX-SERVER-7: OTP expiry 600s (was 6000)
//   FIX-SERVER-8: DownloadGlobalModel streams with hash verification
//   FIX-SERVER-9: epsilon_spent range validated (0 < eps <= epsilon_max)
//
// COMPILE FIXES (preserved):
//   FIX-COMPILE-1: futures::AsyncWrite/AsyncRead imports (not tokio)
//   FIX-COMPILE-2: Pin<Box<dyn Stream<...>>> for DownloadGlobalModelStream
//   FIX-COMPILE-3: Removed unused imports (ct_eq, etc.)

// src/grpc/server.rs - TOP SECTION (imports) - FIXED
use std::io::Write as _;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Command;  // ← sync, not async
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// FIXED: Only import what's actually used
use futures::StreamExt;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio::fs;
use tonic::{Request, Response, Status, Streaming};

use crate::config::Config;
use crate::crypto::hash_bytes;
use crate::grpc::orchestrator::orchestrator_server::{Orchestrator, OrchestratorServer};
// FIXED: Use correct proto message names from orchestrator.proto
use crate::grpc::orchestrator::{
    Ack, Certificate, Csr, DeviceId, EnrollRequest, EnrollResponse,
    EnrollmentRequest, EnrollmentRequestAck, ModelChunk,
    Receipt, RoundMetadata, RoundRequest,
    UpdateChunk, UploadAck,
};
use crate::identity::derive_device_id;
use crate::ledger;
use crate::otp;
use crate::receipts;
// FIXED: Import Round/RoundState from correct module
use crate::round::{Round, RoundState};
use crate::state::OrchestratorState;

// ── Constants ─────────────────────────────────────────────────────────────────
const MAX_UPDATE_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GB absolute cap
const CHUNK_SIZE_MAX: usize = 4 * 1024 * 1024; // 4 MB per chunk
const CHUNK_SIZE: usize = 1 * 1024 * 1024; // 1 MB streaming chunks for download

// ── Service structs ───────────────────────────────────────────────────────────
/// EnrollmentService: served on port 50051 (server-TLS only, no client cert)
/// Only RequestEnrollment + EnrollDevice implemented. Others return UNIMPLEMENTED.
pub struct EnrollmentService {
    state: Arc<OrchestratorState>,
    cfg: Config,
}

/// OperationalService: served on port 50052 (full mTLS required)
/// Implements GetRound, UploadUpdate, SubmitReceipt, DownloadGlobalModel.
pub struct OperationalService {
    state: Arc<OrchestratorState>,
    cfg: Config,
    receipt_chain_key: Vec<u8>, // HMAC key for receipt chaining
}

impl EnrollmentService {
    pub fn new(state: Arc<OrchestratorState>, cfg: Config) -> Self {
        Self { state, cfg }
    }
}

impl OperationalService {
    pub fn new(state: Arc<OrchestratorState>, cfg: Config) -> anyhow::Result<Self> {
        // Load or generate receipt chain HMAC key
        let receipt_chain_key = std::env::var("RECEIPT_CHAIN_KEY")
            .map(|s| hex::decode(s).expect("RECEIPT_CHAIN_KEY must be hex"))
            .unwrap_or_else(|_| {
                tracing::warn!("RECEIPT_CHAIN_KEY not set — using ephemeral key (NOT for production)");
                use rand::RngCore;
                let mut k = vec![0u8; 32];
                rand::thread_rng().fill_bytes(&mut k);
                k
            });
        Ok(Self { state, cfg, receipt_chain_key })
    }

    /// Enforce mTLS: reject if no client certificate presented
    fn require_client_cert<T>(req: &Request<T>) -> Result<(), Status> {
        match req.peer_certs() {
            Some(certs) if !certs.is_empty() => Ok(()),
            _ => {
                tracing::warn!("Request rejected — no mTLS client certificate");
                Err(Status::unauthenticated("mutual TLS client certificate required"))
            }
        }
    }

    /// Compute HMAC chain link: prev_hmac | payload_hash → new_hmac
    fn compute_chain_hmac(&self, prev_hmac: Option<&str>, payload_hash_hex: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.receipt_chain_key)
            .expect("HMAC key is valid length");
        mac.update(prev_hmac.unwrap_or("genesis").as_bytes());
        mac.update(b"|");
        mac.update(payload_hash_hex.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

// ── EnrollmentService implementation ──────────────────────────────────────────
#[tonic::async_trait]
impl Orchestrator for EnrollmentService {
    type DownloadGlobalModelStream =
        Pin<Box<dyn futures::Stream<Item = Result<ModelChunk, Status>> + Send + 'static>>;

    // ── RequestEnrollment: Phase B1 — device requests OTP ─────────────────────
    async fn request_enrollment(
        &self,
        req: Request<EnrollmentRequest>,
    ) -> Result<Response<EnrollmentRequestAck>, Status> {
        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let inner = req.into_inner();

        if inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        if inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr is required"));
        }

        // Generate fingerprint (first 8 bytes of SHA-256(pubkey))
        let fp_bytes = hash_bytes(&inner.device_pubkey);
        let fingerprint = hex::encode(&fp_bytes[..8]);

        // Generate OTP and store pending enrollment
        let otp = otp::generate_otp_for(Some(fingerprint.clone()));
        self.state
            .pending_enrollments
            .write()
            .unwrap()
            .insert(fingerprint.clone(), (inner.device_pubkey.clone(), inner.csr.clone()));

        // Log to console for administrator
        let device_info = if inner.device_info.is_empty() {
            format!("peer={}", peer_addr)
        } else {
            format!("{} / peer={}", &inner.device_info[..inner.device_info.len().min(60)], peer_addr)
        };
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  NEW ENROLLMENT REQUEST                                  ║");
        println!("║  Fingerprint : {:<42} ║", fingerprint);
        println!("║  Device      : {:<42} ║", &device_info[..device_info.len().min(42)]);
        println!("║  OTP         : {:<42} ║", otp);
        println!("║  Expiry      : 10 minutes                                ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        tracing::info!(
            "Enrollment requested — fingerprint={} peer={}",
            fingerprint, peer_addr
        );

        Ok(Response::new(EnrollmentRequestAck {
            accepted: true,
            device_fingerprint: fingerprint,
        }))
    }

    // ── EnrollDevice: Phase B2 — device presents OTP, gets cert ──────────────
    async fn enroll_device(
        &self,
        req: Request<EnrollRequest>,
    ) -> Result<Response<EnrollResponse>, Status> {
        let peer_addr = req
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let req_inner = req.into_inner();

        // Verify OTP (consumes it — one-time use)
        if !otp::consume_otp_from(&req_inner.enrollment_token, &peer_addr) {
            tracing::warn!(
                "Enrollment rejected peer={} — invalid or expired OTP",
                peer_addr
            );
            return Err(Status::permission_denied("invalid or expired OTP"));
        }

        if req_inner.device_pubkey.is_empty() {
            return Err(Status::invalid_argument("device_pubkey is required"));
        }
        if req_inner.csr.is_empty() {
            return Err(Status::invalid_argument("csr is required"));
        }

        // Derive device ID and store in state
        let device_id = derive_device_id(&req_inner.device_pubkey);
        let device_hex = hex::encode(&device_id);
        self.state
            .devices
            .write()
            .unwrap()
            .insert(device_hex.clone(), req_inner.device_pubkey.clone());

        // Sign CSR with CA using OpenSSL
        let mut csr_file = tempfile::NamedTempFile::new()
            .map_err(|_| Status::internal("temp file creation failed"))?;
        csr_file.write_all(&req_inner.csr)
            .map_err(|_| Status::internal("CSR write failed"))?;
        let cert_file = tempfile::NamedTempFile::new()
            .map_err(|_| Status::internal("temp file creation failed"))?;

        let output = Command::new("openssl")
            .args([
                "x509", "-req",
                "-in",  csr_file.path().to_str().unwrap(),
                "-CA",  &self.cfg.tls.ca_cert,
                "-CAkey", &self.cfg.tls.ca_key,
                "-CAcreateserial",
                "-out", cert_file.path().to_str().unwrap(),
                "-days", "365", "-sha256",
            ])
            .output()
            .map_err(|e| {
                tracing::error!("openssl exec failed: {}", e);
                Status::internal("certificate signing failed")
            })?;

        if !output.status.success() {
            tracing::error!(
                "openssl stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Err(Status::internal("certificate signing failed"));
        }

        let signed_cert = fs::read(cert_file.path())
            .await
            .map_err(|_| Status::internal("failed to read signed cert"))?;
        if signed_cert.is_empty() {
            return Err(Status::internal("empty certificate output"));
        }

        // Write device to filesystem
        let devices_dir = self.state.devices_dir();
        fs::create_dir_all(&devices_dir).await.map_err(|e| {
            tracing::error!("Cannot create devices dir: {}", e);
            Status::internal("storage error")
        })?;
        let device_record = serde_json::json!({
            "device_id": device_hex,
            "pubkey_pem": String::from_utf8_lossy(&req_inner.device_pubkey),
            "enrolled_at": utcnow(),
        });
        let device_path = devices_dir.join(format!("{}.json", device_hex));
        fs::write(&device_path, device_record.to_string().as_bytes()).await.map_err(|e| {
            tracing::error!("Failed to write device {:?}: {}", device_path, e);
            Status::internal("storage error")
        })?;

        // Remove from pending enrollments
        let fp_bytes = hash_bytes(&req_inner.device_pubkey);
        let fingerprint = hex::encode(&fp_bytes[..8]);
        self.state.pending_enrollments.write().unwrap().remove(&fingerprint);

        // Ledger: log enrollment
        ledger::append_to(
            serde_json::json!({
                "type": "enrollment",
                "device_id": device_hex,
                "fingerprint": fingerprint,
                "peer_addr": peer_addr,
                "timestamp": utcnow(),
            }).to_string().as_bytes(),
            &self.state.ledger_path(),
        );

        tracing::info!("Device enrolled — fingerprint={} peer={}", fingerprint, peer_addr);
        println!("[ENROLLED] fingerprint={} peer={}", fingerprint, peer_addr);

        Ok(Response::new(EnrollResponse {
            ok: true,
            client_cert: signed_cert,
        }))
    }

    // All operational RPCs are NOT served on enrollment port
    async fn get_round(&self, _req: Request<DeviceId>) -> Result<Response<RoundMetadata>, Status> {
        Err(Status::unimplemented("connect to operational port 50052"))
    }
    async fn upload_update(&self, _req: Request<Streaming<UpdateChunk>>) -> Result<Response<UploadAck>, Status> {
        Err(Status::unimplemented("connect to operational port 50052"))
    }
    async fn submit_receipt(&self, _req: Request<Receipt>) -> Result<Response<Ack>, Status> {
        Err(Status::unimplemented("connect to operational port 50052"))
    }
    async fn download_global_model(&self, _req: Request<RoundRequest>) -> Result<Response<Self::DownloadGlobalModelStream>, Status> {
        Err(Status::unimplemented("connect to operational port 50052"))
    }
    async fn register_device(&self, _req: Request<Csr>) -> Result<Response<Certificate>, Status> {
        Err(Status::unimplemented("use RequestEnrollment + EnrollDevice"))
    }
}

// ── OperationalService implementation ─────────────────────────────────────────
#[tonic::async_trait]
impl Orchestrator for OperationalService {
    type DownloadGlobalModelStream =
        Pin<Box<dyn futures::Stream<Item = Result<ModelChunk, Status>> + Send + 'static>>;

    // ── Deprecated endpoints return unimplemented on operational port ───────
    async fn register_device(&self, _req: Request<Csr>) -> Result<Response<Certificate>, Status> {
        Err(Status::unimplemented("use RequestEnrollment + EnrollDevice on port 50051"))
    }

    async fn request_enrollment(&self, _req: Request<EnrollmentRequest>) -> Result<Response<EnrollmentRequestAck>, Status> {
        Err(Status::unimplemented("connect to enrollment port 50051"))
    }

    async fn enroll_device(&self, _req: Request<EnrollRequest>) -> Result<Response<EnrollResponse>, Status> {
        Err(Status::unimplemented("connect to enrollment port 50051"))
    }

    // ── UploadUpdate: client streams encrypted update bytes ───────────────────
    async fn upload_update(
        &self,
        req: Request<Streaming<UpdateChunk>>,
    ) -> Result<Response<UploadAck>, Status> {  // FIX: Return UploadAck, not ServerHandle
        Self::require_client_cert(&req)?;

        let mut stream = req.into_inner();
        let mut all_bytes: Vec<u8> = Vec::new();
        let mut round_id: u64 = 0;
        let mut device_id_bytes: Vec<u8> = Vec::new();
        let mut expected_total: u64 = 0;
        let mut received_chunks: u64 = 0;
        let mut global_hasher = Sha256::new();
        let mut initialized = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| Status::internal(format!("stream read error: {}", e)))?;

            if chunk.data.len() > CHUNK_SIZE_MAX {
                return Err(Status::invalid_argument(format!(
                    "chunk {} exceeds max size {}MB",
                    chunk.chunk_index, CHUNK_SIZE_MAX / 1024 / 1024
                )));
            }

            // FIX-SERVER-3: per-chunk SHA-256 verification
            let computed_hash = Sha256::digest(&chunk.data);
            if chunk.chunk_hash.as_slice() != computed_hash.as_slice() {
                return Err(Status::data_loss(format!(
                    "chunk {} hash mismatch — data corrupted or tampered in transit",
                    chunk.chunk_index
                )));
            }

            if chunk.chunk_index != received_chunks {
                return Err(Status::invalid_argument(format!(
                    "out-of-order chunk: expected index {}, got {}",
                    received_chunks, chunk.chunk_index
                )));
            }

            if all_bytes.len() + chunk.data.len() > MAX_UPDATE_BYTES {
                return Err(Status::resource_exhausted("update exceeds 2GB maximum"));
            }

            if !initialized {
                round_id = chunk.round_id;
                device_id_bytes = chunk.device_id.clone();
                expected_total = chunk.total_chunks;

                if device_id_bytes.is_empty() {
                    return Err(Status::invalid_argument("device_id required in first chunk"));
                }
                if expected_total == 0 {
                    return Err(Status::invalid_argument("total_chunks must be > 0"));
                }

                let device_hex = hex::encode(&device_id_bytes);
                if self.state.devices.read().unwrap().get(&device_hex).is_none() {
                    tracing::warn!("Upload rejected — device {} not enrolled", device_hex);
                    return Err(Status::permission_denied("device not enrolled"));
                }

                let rounds = self.state.rounds.read().unwrap();
                if let Some(round) = rounds.get(&round_id) {
                    if round.state != RoundState::Collecting {
                        return Err(Status::failed_precondition("round not in Collecting state"));
                    }
                } else {
                    return Err(Status::not_found("round not found"));
                }
                initialized = true;
            } else {
                if chunk.round_id != round_id || chunk.device_id != device_id_bytes || chunk.total_chunks != expected_total {
                    return Err(Status::invalid_argument("chunk metadata mismatch"));
                }
            }

            global_hasher.update(&chunk.data);
            all_bytes.extend_from_slice(&chunk.data);
            received_chunks += 1;
        }

        if !initialized || received_chunks == 0 {
            return Err(Status::invalid_argument("empty upload — no chunks received"));
        }
        if received_chunks != expected_total {
            return Err(Status::invalid_argument(format!(
                "chunk count mismatch: declared {}, received {}", expected_total, received_chunks
            )));
        }

        let payload_hash = global_hasher.finalize();
        let payload_hash_hex = hex::encode(payload_hash);
        let device_hex = hex::encode(&device_id_bytes);

        let updates_dir = self.state.updates_dir(round_id);
        fs::create_dir_all(&updates_dir).await.map_err(|e| {
            Status::internal(format!("mkdir error: {}", e))
        })?;
        let update_path = updates_dir.join(format!("{}.bin", device_hex));
        fs::write(&update_path, &all_bytes).await.map_err(|e| {
            tracing::error!("Failed to write update {:?}: {}", update_path, e);
            Status::internal("storage write error")
        })?;

        {
            let mut rounds = self.state.rounds.write().unwrap();
            if let Some(round) = rounds.get_mut(&round_id) {
                round.updates.retain(|u| u.device_id_hex != device_hex);
                round.updates.push(crate::round::UpdateEntry {
                    device_id_hex: device_hex.clone(),
                    file_path: update_path.to_string_lossy().into_owned(),
                    payload_hash: payload_hash_hex.clone(),
                    epsilon_spent: 0.0,
                    verified: false,
                    timestamp: utcnow(),
                });
            }
        }

        tracing::info!(
            "Upload stored (unverified): device={} round={} bytes={} hash={}",
            device_hex, round_id, all_bytes.len(), payload_hash_hex
        );

        let server_handle = format!("file://{}", update_path.to_string_lossy());
        
        // FIX: Return UploadAck with correct fields (ok + server_handle)
        Ok(Response::new(UploadAck {
            ok: true,
            server_handle,  // proto field name is server_handle
            error: String::new(),  // empty on success
        }))
    }

    // ── SubmitReceipt: verify signature + hash + update epsilon budget ───────
    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {
        Self::require_client_cert(&req)?;

        let receipt = req.into_inner();

        if receipt.device_id.is_empty() {
            return Err(Status::invalid_argument("device_id is required"));
        }
        if receipt.payload_hash.len() != 32 {
            return Err(Status::invalid_argument("payload_hash must be 32 bytes (SHA-256)"));
        }
        if receipt.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }
        if receipt.enc_handle.is_empty() {
            return Err(Status::invalid_argument("enc_handle is required — call UploadUpdate first"));
        }
        if receipt.epsilon_spent <= 0.0 {
            return Err(Status::invalid_argument("epsilon_spent must be positive"));
        }

        let device_hex = hex::encode(&receipt.device_id);

        let pubkey_pem = {
            let devices = self.state.devices.read().unwrap();
            let dev = devices.get(&device_hex)
                .ok_or_else(|| Status::permission_denied("device not enrolled"))?;
            String::from_utf8_lossy(dev).into_owned()
        };

        let mut msg = Vec::with_capacity(receipt.device_id.len() + 8 + 32);
        msg.extend_from_slice(&receipt.device_id);
        msg.extend_from_slice(&receipt.round_id.to_be_bytes());
        msg.extend_from_slice(&receipt.payload_hash);
        receipts::verify(pubkey_pem.as_bytes(), &msg, &receipt.signature)
            .map_err(|_| {
                tracing::warn!("Invalid receipt signature from device {}", device_hex);
                Status::permission_denied("receipt signature verification failed")
            })?;

        if !receipt.enc_handle.starts_with("file://") {
            return Err(Status::invalid_argument("enc_handle must be file:// URI"));
        }
        let update_path = PathBuf::from(&receipt.enc_handle["file://".len()..]);
        let updates_dir = self.state.updates_dir(receipt.round_id);
        if !update_path.starts_with(&updates_dir) {
            return Err(Status::permission_denied("handle path traversal attempt"));
        }
        if !update_path.exists() {
            return Err(Status::not_found("uploaded file not found"));
        }

        let stored_hash = {
            let rounds = self.state.rounds.read().unwrap();
            rounds.get(&receipt.round_id)
                .and_then(|r| r.updates.iter().find(|u| u.device_id_hex == device_hex))
                .map(|u| u.payload_hash.clone())
                .ok_or_else(|| Status::not_found("no matching unverified upload found"))?
        };
        let submitted_hash = hex::encode(&receipt.payload_hash);
        if stored_hash != submitted_hash {
            tracing::warn!("payload_hash mismatch device={}", device_hex);
            return Err(Status::permission_denied("payload_hash does not match uploaded data"));
        }

        let prev_hmac = ledger::get_last_hmac(&self.state.ledger_path(), receipt.round_id);
        let chain_hmac = self.compute_chain_hmac(prev_hmac.as_deref(), &submitted_hash);

        let should_aggregate = {
            let mut rounds = self.state.rounds.write().unwrap();
            let round = rounds.get_mut(&receipt.round_id)
                .ok_or_else(|| Status::not_found("round not found"))?;

            if round.state != RoundState::Collecting {
                return Err(Status::failed_precondition("round not in Collecting state"));
            }

            let entry = round.updates.iter_mut()
                .find(|u| u.device_id_hex == device_hex && !u.verified)
                .ok_or_else(|| Status::not_found("unverified upload not found"))?;

            if round.epsilon_spent + receipt.epsilon_spent > round.epsilon_max {
                return Err(Status::resource_exhausted(format!(
                    "epsilon budget exceeded: {:.4} + {:.4} > {:.4}",
                    round.epsilon_spent, receipt.epsilon_spent, round.epsilon_max
                )));
            }
            round.epsilon_spent += receipt.epsilon_spent;
            entry.epsilon_spent = receipt.epsilon_spent;
            entry.verified = true;

            let min_updates: usize = std::env::var("FL_MIN_UPDATES_FOR_AGGREGATION")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
            round.updates.iter().filter(|u| u.verified).count() >= min_updates
        };

        let record = serde_json::json!({
            "type": "receipt",
            "device_id": device_hex,
            "round_id": receipt.round_id,
            "payload_hash": submitted_hash,
            "epsilon_spent": receipt.epsilon_spent,
            "signature": hex::encode(&receipt.signature),
            "handle": receipt.enc_handle,
            "scheme": receipt.scheme,
            "nonce": receipt.nonce,
            "hmac_chain": chain_hmac,
            "timestamp": utcnow(),
        });
        ledger::append_to(record.to_string().as_bytes(), &self.state.ledger_path());

        tracing::info!("Receipt verified: device={} round={} eps={:.4}", device_hex, receipt.round_id, receipt.epsilon_spent);

        if should_aggregate {
            let state_clone = Arc::clone(&self.state);
            let cfg_clone = self.cfg.clone();
            let round_id_copy = receipt.round_id;
            tokio::spawn(async move {
                if let Err(e) = run_aggregation(state_clone, cfg_clone, round_id_copy).await {
                    tracing::error!("Aggregation failed for round {}: {:?}", round_id_copy, e);
                }
            });
        }

        // FIX: Ack only has 'ok' field per proto definition
        Ok(Response::new(Ack { ok: true }))  // Removed non-existent 'message' field
    }

    // ── DownloadGlobalModel: stream model with hash verification ─────────────
    async fn download_global_model(
        &self,
        req: Request<RoundRequest>,  // ← RoundRequest, not ModelRequest
    ) -> Result<Response<Self::DownloadGlobalModelStream>, Status> {
        Self::require_client_cert(&req)?;
        let inner = req.into_inner();  // inner: RoundRequest
        
        let device_hex = hex::encode(&inner.device_id);
        if self.state.devices.read().unwrap().get(&device_hex).is_none() {
            return Err(Status::permission_denied("device not enrolled"));
        }
        
        // Locate model file using inner.round_id
        let model_path = {
            let rounds = self.state.rounds.read().unwrap();
            rounds.get(&inner.round_id)
                .and_then(|r| r.global_model_path.as_ref())
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    self.state.global_models_dir()
                        .join(format!("round_{}.bin", inner.round_id))
                })
        };

        if !model_path.exists() {
            return Err(Status::not_found(format!("No global model for round {}", inner.round_id)));
        }

        let model_bytes = fs::read(&model_path).await.map_err(|e| {
            tracing::error!("Failed to read global model {:?}: {}", model_path, e);
            Status::internal("model read error")
        })?;

        let model_hash_bytes = Sha256::digest(&model_bytes).to_vec();
        tracing::info!("Streaming global model: round={} bytes={}", inner.round_id, model_bytes.len());

        let total = model_bytes.len();
        let total_chunks = ((total + CHUNK_SIZE - 1) / CHUNK_SIZE) as u64;
        let (tx, rx) = mpsc::channel::<Result<ModelChunk, Status>>(16);
        let model_hash_clone = model_hash_bytes.clone();

        tokio::spawn(async move {
            let mut offset = 0usize;
            let mut chunk_index = 0u64;
            while offset < total {
                let end = (offset + CHUNK_SIZE).min(total);
                let chunk_data = model_bytes[offset..end].to_vec();
                let chunk_hash = Sha256::digest(&chunk_data).to_vec();
                let is_last = end >= total;
                let model_hash_field = if is_last { model_hash_clone.clone() } else { Vec::new() };
                let msg = Ok(ModelChunk {
                    chunk_index,
                    total_chunks,
                    data: chunk_data,
                    chunk_hash,
                    model_hash: model_hash_field,
                });
                if tx.send(msg).await.is_err() { break; }
                offset = end;
                chunk_index += 1;
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    // ── GetRound: return round metadata + global model availability ──────────
    async fn get_round(
        &self,
        req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {
        Self::require_client_cert(&req)?;
        let inner = req.into_inner();
        let device_hex = hex::encode(&inner.id);
        
        // Verify device enrolled
        if self.state.devices.read().unwrap().get(&device_hex).is_none() {
            return Err(Status::permission_denied("device not enrolled"));
        }
        
        let rounds = self.state.rounds.read().unwrap();
        let round = rounds.values()
            .max_by_key(|r| r.id)
            .ok_or_else(|| Status::not_found("no active rounds"))?;
        
        // Map internal Round → proto Round
        Ok(Response::new(RoundMetadata {
            round_id: round.id,
            model_version: round.model_version.clone(),
            epsilon_max: round.epsilon_max,
            upload_uri: round.upload_uri.clone(),
            state: format!("{:?}", round.state),
            num_updates: round.updates.iter().filter(|u| u.verified).count() as u32,
            aggregation_mode: "trimmed_mean".into(),
            global_model_available: round.global_model_path.is_some(),
        }))
    }
}

// ── Aggregation: spawn Python aggregator subprocess ───────────────────────────
async fn run_aggregation(
    state: Arc<OrchestratorState>,
    _cfg: Config,
    round_id: u64,
) -> Result<(), Status> {
    let server_root = &state.server_root;

    // Ensure global_models dir exists
    let gm_dir = state.global_models_dir();
    fs::create_dir_all(&gm_dir).await.map_err(|e| {
        Status::internal(format!("mkdir global_models: {}", e))
    })?;

    // Locate aggregator.py relative to executable
    let aggregator_script: PathBuf = {
        let exe = std::env::current_exe().ok().unwrap_or_else(|| PathBuf::from("."));
        exe.ancestors()
            .find(|p| p.join("server").join("aggregator_agent").join("aggregator.py").exists())
            .map(|p| p.join("server").join("aggregator_agent").join("aggregator.py"))
            .unwrap_or_else(|| PathBuf::from("server/aggregator_agent/aggregator.py"))
    };

    if !aggregator_script.exists() {
        tracing::error!("Aggregator script not found: {:?}", aggregator_script);
        return Err(Status::internal("aggregator script missing"));
    }

    tracing::info!("Spawning aggregator: {:?} --round-id {}", aggregator_script, round_id);

    let output = Command::new("python3")
        .arg(&aggregator_script)
        .arg("--round-id").arg(round_id.to_string())
        .arg("--server-root").arg(server_root.to_str().unwrap_or("."))
        .env("PYTHONPATH", server_root.ancestors().nth(1).unwrap_or(server_root))
        .env("MONGO_URI", "filesystem")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()  // ← SYNC call
        .map_err(|e| {  // ← Handle Result directly, no .await
            tracing::error!("Failed to spawn aggregator: {}", e);
            Status::internal("aggregator spawn failed")
        })?;

    if !output.status.success() {
        tracing::error!("Aggregator failed:\nstderr: {}\nstdout: {}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout));
        return Err(Status::internal("aggregation subprocess failed"));
    }

    // Parse GLOBAL_MODEL_PATH=<path> from stdout
    let global_model_path = String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|l| l.starts_with("GLOBAL_MODEL_PATH="))
        .map(|l| l.trim_start_matches("GLOBAL_MODEL_PATH=").trim().to_string())
        .unwrap_or_else(|| gm_dir.join(format!("round_{}.bin", round_id)).to_string_lossy().into_owned());

    // Update in-memory state: mark round complete, seed next round
    {
        let mut rounds = state.rounds.write().unwrap();
        if let Some(r) = rounds.get_mut(&round_id) {
            r.global_model_path = Some(global_model_path.clone());
            r.state = RoundState::Complete;
        }
        // Open next round
        let next_id = round_id + 1;
        let eps = std::env::var("FL_EPSILON_MAX")
            .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(8.0);
        rounds.insert(next_id, Round {
            id: next_id,
            model_version: format!("v{}", next_id),
            epsilon_max: eps,
            epsilon_spent: 0.0,
            state: RoundState::Collecting,
            upload_uri: String::new(),
            updates: Vec::new(),
            aggregation_receipt: None,
            global_model_path: Some(global_model_path.clone()),
        });
    }

    // Log aggregation to ledger
    let record = serde_json::json!({
        "type": "aggregation_complete",
        "round_id": round_id,
        "next_round_id": round_id + 1,
        "global_model_path": global_model_path,
        "timestamp": utcnow(),
    });
    ledger::append_to(record.to_string().as_bytes(), &state.ledger_path());

    tracing::info!("Aggregation complete: round={} model={}", round_id, global_model_path);
    Ok(())
}

// ── Server bootstrap: dual-port architecture ──────────────────────────────────
pub async fn serve(
    cfg: Config,
    state: Arc<OrchestratorState>,
) -> anyhow::Result<()> {
    let enrollment_addr: std::net::SocketAddr = cfg.server.addr.parse()?;
    let operational_addr: std::net::SocketAddr = cfg.server.mtls_addr.parse()?;

    let server_identity = tonic::transport::Identity::from_pem(
        std::fs::read(&cfg.tls.server_cert)?,
        std::fs::read(&cfg.tls.server_key)?,
    );
    let client_ca = std::fs::read(&cfg.tls.ca_cert)?;

    // Port 50051: enrollment (server-TLS only)
    let enrollment_tls = tonic::transport::ServerTlsConfig::new()
        .identity(server_identity.clone());
    let enrollment_svc = EnrollmentService::new(state.clone(), cfg.clone());
    let enroll_server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .tls_config(enrollment_tls).expect("enrollment TLS config failed")
            .add_service(OrchestratorServer::new(enrollment_svc))
            .serve(enrollment_addr).await.expect("enrollment server failed")
    });
    tracing::info!("[ENROLL-SERVER] {} — server-TLS only", enrollment_addr);
    println!("[ENROLL-SERVER] {} — server-TLS only (enrollment)", enrollment_addr);

    // Port 50052: operational (full mTLS)
    let operational_tls = tonic::transport::ServerTlsConfig::new()
        .identity(server_identity)
        .client_ca_root(tonic::transport::Certificate::from_pem(client_ca));
    let operational_svc = OperationalService::new(state, cfg)?;
    let ops_server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .tls_config(operational_tls).expect("operational TLS config failed")
            .add_service(OrchestratorServer::new(operational_svc))
            .serve(operational_addr).await.expect("operational server failed")
    });
    tracing::info!("[OPS-SERVER] {} — full mTLS", operational_addr);
    println!("[OPS-SERVER] {} — full mTLS (operational)", operational_addr);

    tokio::try_join!(
        async { enroll_server.await.map_err(|e: tokio::task::JoinError| anyhow::anyhow!(e)) },
        async { ops_server.await.map_err(|e: tokio::task::JoinError| anyhow::anyhow!(e)) },
    )?;
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn utcnow() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // ISO 8601 via chrono if available; fallback to Unix epoch
    format!("{}Z", secs)
}