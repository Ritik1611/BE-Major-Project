use std::sync::Arc;
use tonic::{Request, Response, Status};
use tonic::transport::{
    Server,
    ServerTlsConfig,
    Identity,
    Certificate as TlsCertificate,
};

use crate::crypto::ct_eq;
use crate::config::Config;
use crate::identity::derive_device_id;
use crate::ledger;
use crate::state::OrchestratorState;

// Generated protobuf types
use crate::grpc::orchestrator::{
    Ack,
    Csr,
    Certificate,
    DeviceId,
    Receipt,
    RoundMetadata,
};

use crate::grpc::orchestrator::orchestrator_server::{
    Orchestrator,
    OrchestratorServer,
};

pub struct Service {
    state: Arc<OrchestratorState>,
}

use crate::receipts;

#[tonic::async_trait]
impl Orchestrator for Service {
    async fn register_device(
        &self,
        req: Request<Csr>,
    ) -> Result<Response<Certificate>, Status> {
        let pubkey = req.into_inner().device_pubkey;

        let device_id = derive_device_id(&pubkey);
        self.state.devices.insert(device_id, ());

        Ok(Response::new(Certificate { pem: pubkey }))
    }

    async fn get_round(
        &self,
        _req: Request<DeviceId>,
    ) -> Result<Response<RoundMetadata>, Status> {
        let round = self.state
            .rounds
            .get(&1)
            .ok_or_else(|| Status::not_found("round not found"))?;

        Ok(Response::new(RoundMetadata {
            round_id: round.id,
            model_version: round.model_version.clone(),
            epsilon_max: round.epsilon_max,
            upload_uri: round.upload_uri.clone(),
        }))
    }

    async fn submit_receipt(
        &self,
        req: Request<Receipt>,
    ) -> Result<Response<Ack>, Status> {

        let receipt = req.into_inner();

        // 1. Device must be known
        let known = self.state.devices.iter().any(|entry| {
            ct_eq(entry.key(), &receipt.device_id)
        });

        if !known {
            return Err(Status::permission_denied("unknown device"));
        }

        // 2. Cryptographic receipt verification
        receipts::verify(
            &receipt.device_id,
            &receipt.payload_hash,
            &receipt.signature,
        )
        .map_err(|_| Status::permission_denied("invalid receipt signature"))?;

        // 3. Append to immutable ledger
        ledger::append(&receipt.payload_hash);

        Ok(Response::new(Ack { ok: true }))
    }

}

pub async fn serve(
    cfg: Config,
    state: Arc<OrchestratorState>,
) -> anyhow::Result<()> {

    let svc = Service { state };

    let server_identity = Identity::from_pem(
        std::fs::read(&cfg.tls.server_cert)?,
        std::fs::read(&cfg.tls.server_key)?,
    );

    let client_ca = TlsCertificate::from_pem(
        std::fs::read(&cfg.tls.ca_cert)?,
    );

    let tls = ServerTlsConfig::new()
        .identity(server_identity)
        .client_ca_root(client_ca);

    Server::builder()
        .tls_config(tls)?
        .add_service(OrchestratorServer::new(svc))
        .serve(cfg.server.addr.parse()?)
        .await?;

    Ok(())
}

