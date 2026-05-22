# Multi-Agent Privacy Orchestrated Framework for Secure Multimodal AI

> **BE Major Project (2025–26) — Semester VII**
> Vivekanand Education Society's Institute of Technology (V.E.S.I.T)
> Department of Computer Engineering | Group No. 42

---

## ⚠️ CRITICAL SECURITY ISSUES — ACTION REQUIRED IMMEDIATELY

> These issues were identified during a line-by-line security audit of the entire codebase.
> **Do not ignore these. They represent live credential and key exposure.**

| Severity | File | Issue | Action |
|---|---|---|---|
| 🔴 CRITICAL | `installer/runtime/agents/lda/pipelines/.env` | HuggingFace API token `` committed in plaintext | **Revoke token immediately at huggingface.co/settings/tokens. Rotate. Never commit secrets.** |
| 🔴 CRITICAL | `server/orchestration_agent/certs/ca.key` | CA private key committed to repository | **Rotate all certificates. Run `gen_certs.sh` fresh. Treat all previously issued certs as compromised.** |
| 🔴 CRITICAL | `server/orchestration_agent/certs/server.key` | Server TLS private key committed to repository | **Rotate immediately. Any MITM attacker can now decrypt all historical TLS traffic.** |
| 🟡 HIGH | `installer/runtime/core/centralised_receipts.py` | HMAC receipt key stored at `~/.federated/state/receipt_hmac.key` in plaintext | Should be derived from TPM-sealed master secret via `derive_receipt_hmac_key()` in `military_security.py` |
| 🟡 HIGH | `installer/security/tpm_seal.py` → `create_master_secret_windows()` | Windows fallback stores master secret as plaintext binary at `~/.federated/secrets/master.bin` | Wrap with DPAPI via `protect_master_key_windows()` from `military_security.py` |
| 🟡 HIGH | `installer/installer_core.py` → `_generate_csr()` | RSA private key written with `serialization.NoEncryption()` | Encrypt with TPM-derived passphrase or generate inside TPM as non-exportable |
| 🟠 MEDIUM | `installer/security/military_security.py` | `CA_PUBKEY_PIN_SHA256` currently set to a placeholder — certificate pinning is disabled | Compute and commit the real CA public key SHA-256 after cert rotation |
| 🟠 MEDIUM | `installer/fs/install_ffmpeg.py` | On first install, FFmpeg ZIP hash is not verified (only stored for future runs) | Pre-compute and hardcode the expected hash before distributing the installer |
| 🟠 MEDIUM | `server/orchestration_agent/src/otp.rs` | OTP is 6 digits (10⁶ space). Rate limiting exists (5 attempts → 5 min lockout), but OTP channel is terminal display | Deliver OTP out-of-band (email/SMS). Increase to 8+ digits. |
| 🟠 MEDIUM | `installer/runtime/configs/requirements.txt` | Hash pinning noted in comments as placeholder — `--require-hashes` not enforced | Run `pip-compile --generate-hashes requirements.in` and enforce with `pip install --require-hashes` |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team](#2-team)
3. [Architecture Overview](#3-architecture-overview)
4. [Repository Structure](#4-repository-structure)
5. [Agent Descriptions](#5-agent-descriptions)
6. [Federated Learning Algorithms](#6-federated-learning-algorithms)
7. [Privacy Mechanisms](#7-privacy-mechanisms)
8. [Security System](#8-security-system)
9. [Multimodal Data Pipeline](#9-multimodal-data-pipeline)
10. [gRPC Protocol & Network Design](#10-grpc-protocol--network-design)
11. [Encrypted Storage (SecureStore)](#11-encrypted-storage-securestore)
12. [Audit & Compliance System](#12-audit--compliance-system)
13. [TPM & Hardware Security](#13-tpm--hardware-security)
14. [Installer & Enrollment](#14-installer--enrollment)
15. [Server (Rust Orchestrator)](#15-server-rust-orchestrator)
16. [Configuration Reference](#16-configuration-reference)
17. [Installation Guide](#17-installation-guide)
18. [Running the System](#18-running-the-system)
19. [Dataset — DAIC-WOZ](#19-dataset--daic-woz)
20. [Model — MentalBERT Multimodal Fusion](#20-model--mentalbert-multimodal-fusion)
21. [Evaluation & Research Suite](#21-evaluation--research-suite)
22. [FL Algorithm Comparison Tool](#22-fl-algorithm-comparison-tool)
23. [Hardware Requirements](#23-hardware-requirements)
24. [Dependencies & Tech Stack](#24-dependencies--tech-stack)
25. [Known Bugs Fixed](#25-known-bugs-fixed)
26. [References](#26-references)

---

## 1. Project Overview

This project implements a **multi-agent privacy orchestration framework** for secure multimodal federated learning, applied to depression detection using the DAIC-WOZ dataset.

The system enables multiple hospital clients to collaboratively train a multimodal AI model (combining audio prosody, facial action units, and MentalBERT text embeddings) **without any raw patient data ever leaving the client device**. Privacy is enforced through a layered stack of:

- **Differential Privacy** (DP-SGD with RDP accounting via Opacus)
- **AES-GCM Encrypted Storage** at rest (HKDF per-agent key isolation)
- **Homomorphic Encryption** option (CKKS/BFV via Pyfhel)
- **Secure Aggregation** (Byzantine-robust trimmed mean)
- **TPM hardware attestation** (Linux tpm2-tools / Windows CNG NCrypt)
- **mTLS mutual authentication** on all gRPC channels
- **HMAC-chained tamper-evident audit ledger**

### Research Objectives

| ID | Objective |
|---|---|
| RO1 | Federated Learning with FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi |
| RO2 | DP-SGD integration with RDP accountant (ε, δ reporting) |
| RO3 | Homomorphic Encryption (CKKS/BFV) and SMPC (Shamir sharing) |
| RO4 | Secure Aggregation (Bonawitz protocol with dropout resilience) |
| RO5 | Multi-agent orchestration with mTLS/gRPC and policy enforcement |
| RO6 | Threat modelling, attack testbeds (MIA, gradient inversion, poisoning), compliance evidence |

---

## 2. Team

| Role | Name | Roll No. | Email |
|---|---|---|---|
| Student | **Ritik Shetty** | D17B / 50 | 2022.ritik.shetty@ves.ac.in |
| Student | **Nickhil Shivakumar** | D17B / 35 | 2022.shivakumar.nickhil@ves.ac.in |
| Student | **Shivam Pandey** | D17B / 39 | 2022.shivam.j.pandey@ves.ac.in |
| Student | **Samarth Nilkanth** | D17B / 38 | 2022.samarth.nilkanth@ves.ac.in |
| Mentor | Mrs. Sujata Khandaskar | — | CMPN Dept, V.E.S.I.T |
| Co-Guide | Mrs. Priti Joshi | — | CMPN Dept, V.E.S.I.T |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT DEVICE                            │
│                                                                 │
│  ┌──────────┐   ┌─────────┐   ┌────────┐   ┌────────────────┐ │
│  │   LDA    │──▶│ Trainer │──▶│   DP   │──▶│  Encryption    │ │
│  │  Agent   │   │  Agent  │   │ Agent  │   │    Agent       │ │
│  │          │   │(MentalBERT│  │DP-SGD  │   │ AES-GCM/CKKS  │ │
│  │Audio/Vid/│   │Multimodal│  │RDP acct│   │                │ │
│  │  Text    │   │ Fusion) │   │        │   │                │ │
│  └──────────┘   └─────────┘   └────────┘   └───────┬────────┘ │
│       │               │            │                │          │
│  SecureStore    SecureStore   SecureStore     gRPC stream      │
│  (AES-GCM)     (AES-GCM)    (AES-GCM)              │          │
│                                                     │          │
│  TPM ──── Runtime Guard ──── Integrity Watcher      │          │
│  (attestation) (anti-debug)   (inotify/SHA3)        │          │
└─────────────────────────────────────────────────────┼──────────┘
                                                       │ mTLS:50052
                                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RUST ORCHESTRATION SERVER                   │
│                                                                 │
│  Port 50051 (server-TLS only):  Enrollment Service              │
│  Port 50052 (full mTLS):        Operational Service             │
│                                                                 │
│  ┌────────────┐   ┌────────────┐   ┌───────────────────────┐   │
│  │ OTP / Enroll│  │  GetRound  │   │   UploadUpdate (stream)│  │
│  │  Manager   │   │  Metadata  │   │  Per-chunk SHA-256     │  │
│  └────────────┘   └────────────┘   └───────────┬───────────┘   │
│                                                 │               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              AGGREGATION (Python subprocess)               │ │
│  │  trimmed_mean / median / krum / FedAdam / FedYogi          │ │
│  │  Byzantine-robust · Per-param norm clamp · Global L2 scale │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│                           │                                     │
│  ┌────────────────────────▼───────────────────────────────────┐ │
│  │            HMAC-CHAINED AUDIT LEDGER (append-only)         │ │
│  │  Receipt chain · Device enrollments · Aggregation receipts │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Modularity**: Each privacy primitive is implemented by a dedicated agent with a clearly defined API.
- **Composability**: Correct composition of DP + AES encryption + secure aggregation; no double-counting.
- **Zero-trust**: Every component validates every input. Paths are canonicalized, tokens are single-use, all messages are signed.
- **Auditability**: Every privacy-relevant event produces a signed HMAC receipt stored in an append-only ledger.
- **Operational Resilience**: Handles dropouts, retries, stale locks, Windows read-only file errors on reinstall, frozen executables.

---

## 4. Repository Structure

```
BE-Major-Project/
├── installer/
│   ├── installer_core.py          # Two-phase installer orchestrator
│   ├── installer_gui.py           # Tkinter GUI (UAC elevation on Windows)
│   ├── fl_algorithm_comparison.py # Standalone FL benchmark script
│   ├── fs/
│   │   ├── secure_layout.py       # Creates ~/.federated/ directory tree
│   │   ├── install_runtime.py     # Copies agents, configs, grpc stubs
│   │   ├── install_python_deps.py # pip install into venv
│   │   ├── install_openface.py    # OpenFace binary deployment
│   │   ├── install_opensmile.py   # openSMILE binary deployment
│   │   ├── install_ffmpeg.py      # FFmpeg install + SHA-256 verification
│   │   └── install_spacy_model.py # spaCy en_core_web_sm download
│   ├── security/
│   │   ├── anti_debug.py          # ptrace / TracerPid / timing checks
│   │   ├── integrity.py           # SHA3-256 tree hash, TPM-signed baseline, inotify watcher
│   │   ├── military_security.py   # Comprehensive threat model & defenses
│   │   ├── runtime_guard.py       # Security gate (installer context)
│   │   ├── self_destruct.py       # Secure wipe of ~/.federated/
│   │   ├── tpm_attestation.py     # TPM provisioning (Linux tpm2-tools / Windows CNG)
│   │   ├── tpm_seal.py            # TPM PCR sealing of master secret
│   │   ├── research_evaluation_suite.py  # Research-grade evaluation (MIA, DLG, poisoning)
│   │   ├── deps_windows.py        # Windows VC++ runtime & binary checks
│   │   └── windows_runtime.py     # VC++ 2015–2022 x64 registry check
│   ├── windows_signer/
│   │   ├── Cargo.toml             # Rust crate for Windows TPM signing
│   │   ├── Cargo.lock
│   │   └── src/main.rs            # NCrypt ECDSA P-256 signer (--init, --pubkey, --sign)
│   └── runtime/
│       ├── federated_client.py    # Main client entrypoint (mode: daemon | run-once)
│       ├── pipeline.py            # LDA → Trainer → DP → Enc → Upload → Receipt
│       ├── grpc_client.py         # Dual-channel mTLS gRPC client
│       ├── runtime_guard.py       # Runtime security gate (hardened v2)
│       ├── daemon.py              # Background daemon (session / continuous modes)
│       ├── capture.py             # Audio/video capture (ffmpeg + cv2 fallbacks)
│       ├── idle.py                # System idle detection (xprintidle / Windows)
│       ├── logging_config.py      # Structured JSON logging + MetricsCollector + HealthReporter
│       ├── config_validator.py    # YAML schema validation before pipeline
│       ├── offline_queue.py       # Encrypted offline queue for failed uploads
│       ├── validate_deps.py       # Runtime dependency verification
│       ├── tpm_guard.py           # TPM signing / master secret unseal at runtime
│       ├── self_destruct.py       # Runtime self-destruct (separate from installer version)
│       ├── keys/
│       │   └── ca.pem             # CA certificate (distributed with installer)
│       ├── grpc/
│       │   ├── orchestrator.proto → compiled stubs
│       │   ├── orchestrator_pb2.py
│       │   └── orchestrator_pb2_grpc.py
│       ├── configs/
│       │   ├── local_config.yaml  # Client configuration file
│       │   └── requirements.txt   # Python dependency list
│       ├── core/
│       │   ├── centralized_secure_store.py   # AES-GCM encrypted store (HKDF isolation)
│       │   └── centralised_receipts.py       # HMAC receipt manager
│       └── agents/
│           ├── lda/
│           │   ├── main.py                   # LDA entrypoint (PreprocessRequest dispatcher)
│           │   └── pipelines/
│           │       ├── audio.py              # OpenSMILE + wav2vec2 + prosody
│           │       ├── video.py              # OpenFace + face blur
│           │       ├── text.py               # spaCy NER scrub + ASR
│           │       ├── session_processor.py  # VAD → diarization → ASR → QA pairs
│           │       └── .env                  # ⚠️ CONTAINS EXPOSED HF_TOKEN — revoke now
│           ├── trainer/
│           │   └── trainer_mentalbert_privacy.py  # MentalBERT multimodal trainer (autonomous/supervised/RL)
│           ├── dp/
│           │   └── dp_agent.py               # DP-SGD, gradient clipping, RDP accounting
│           └── enc/
│               └── enc_agent.py              # AES-GCM / HE-CKKS / KMS-envelope encryption
├── server/
│   ├── aggregator_agent/
│   │   ├── aggregator.py                     # Trimmed mean aggregation (Python)
│   │   ├── test_aggregator_local.py          # Local test harness
│   │   └── core/
│   │       ├── centralized_secure_store.py   # (mirror of client SecureStore)
│   │       └── centralised_receipts.py       # (mirror of client ReceiptManager)
│   └── orchestration_agent/                  # Rust orchestrator
│       ├── Cargo.toml
│       ├── Cargo.lock
│       ├── build.rs
│       ├── config/orchestrator.toml          # Server addresses + TLS config
│       ├── proto/orchestrator.proto          # gRPC service definition
│       ├── certs/
│       │   ├── gen_certs.sh                  # Certificate generation script
│       │   ├── ca.pem                        # CA certificate
│       │   ├── ca.key                        # ⚠️ CA PRIVATE KEY — ROTATE IMMEDIATELY
│       │   ├── server.pem                    # Server certificate
│       │   ├── server.key                    # ⚠️ SERVER PRIVATE KEY — ROTATE IMMEDIATELY
│       │   ├── server.csr
│       │   ├── server.ext
│       │   └── ca.srl
│       └── src/
│           ├── main.rs                       # Tokio entrypoint, directory setup
│           ├── config.rs                     # TOML config deserialization
│           ├── crypto.rs                     # SHA-256 + constant-time comparison
│           ├── errors.rs                     # Error enum
│           ├── identity.rs                   # Device ID derivation (SHA-256 of pubkey)
│           ├── ledger.rs                     # HMAC-chained append-only ledger
│           ├── otp.rs                        # OTP generation + rate limiting
│           ├── pubsub.rs                     # Round status broadcaster
│           ├── receipts.rs                   # ECDSA signature verification (PEM→DER→SEC1)
│           ├── round.rs                      # Round / RoundState / UpdateEntry types
│           ├── state.rs                      # In-memory state + filesystem helpers
│           └── grpc/
│               ├── mod.rs
│               └── server.rs                 # EnrollmentService (50051) + OperationalService (50052)
└── BE_Project_Synopsis_Template_25-26.docx
```

---

## 5. Agent Descriptions

### 5.1 Local Data Agent (LDA)

**File**: `installer/runtime/agents/lda/main.py`

Responsible for ingesting raw multimodal patient data, performing PII scrubbing, and producing encrypted feature artifacts.

**Processing Modes**:

| Mode | Description | FL Update? |
|---|---|---|
| `session` | Recorded therapist–patient session; outputs QA-paired segments | ✅ Yes |
| `interactive` | Live clinician-guided session; inference only | ❌ No |
| `continuous` | Background monitoring in 5-min windows | ❌ No (inference only) |
| `batch` | Offline batch processing of directories | ✅ Yes |
| `text` | Text-only input (no audio/video) | ✅ Yes |

**Sub-pipelines**:

- **Audio** (`audio.py`): Runs OpenSMILE eGeMAPS, wav2vec2 embeddings, basic prosody (energy, ZCR, pitch). Falls back gracefully if binaries are missing.
- **Video** (`video.py`): OpenFace feature extraction (Action Units, gaze, pose, blink). Blurs faces before any video storage.
- **Text** (`text.py`): spaCy NER scrubbing (`PERSON`, `GPE`, `ORG`), phone/email regex removal. Supports ASR output and raw text.
- **Session Processor** (`session_processor.py`): Full pipeline — VAD (WebRTC VAD → energy fallback), pyannote speaker diarization, openai-whisper / HF transformers ASR, face tracking, QA pair assembly.

**Output**: Encrypted parquet files + encrypted JSONL manifest in `~/.federated/data/secure_store/`, with per-segment receipts.

**API** (internal Python call):
```python
from agents.lda.main import preprocess, PreprocessRequest
result = preprocess(PreprocessRequest(
    mode="session",
    inputs={"video_dir": "/path/to/sessions"},
    config_uri="file://~/.federated/configs/local_config.yaml"
))
# Returns: {"session_id", "artifact_manifest", "receipts", "count"}
```

---

### 5.2 Trainer Agent

**File**: `installer/runtime/agents/trainer/trainer_mentalbert_privacy.py`

MentalBERT-based multimodal model with three training modes.

**Training Modes**:

| Mode | Description |
|---|---|
| `autonomous` | Inference-only; generates predictions + explainability (modality ablation) |
| `supervised` | Fine-tunes on PHQ-8 labelled data; accepts clinician correction via CLI |
| `rl` | REINFORCE-style online update using clinician-corrected PHQ as reward signal |

**Model Architecture** (`MultiModalModel`):

```
Input:
  text    → MentalBERT (768-d CLS embedding)
  audio   → SmallMLP(audio_dim → 128)
  visual  → SmallMLP(visual_dim → 128)
                        ↓
  Concatenate: [text(768) | audio(128) | visual(128)] = 1024-d
                        ↓
  FusionHead:
    Linear(1024 → 256) → ReLU → Dropout
    Linear(256 → 2)          → class logits
    Linear(256 → 1)          → PHQ-8 score μ
    Linear(256 → 1)          → PHQ-8 score log_σ  (for RL)
```

**Safety Policy** applied to every delta before upload:
- Per-parameter absolute clamp: `max_param_change = 1e-3`
- Global L2 norm scaling: `max_global_delta_norm = 1.0`
- Gradient clipping during optimization: `clip_norm = 1.0`

**Warm-start**: If a global model path is provided (from `DownloadGlobalModel` RPC), the trainer loads it with `load_state_dict(strict=False)` before local fine-tuning, implementing the FL averaging warm-start.

**Output**: Encrypted `.pt.enc` delta state dict in SecureStore + HMAC receipt.

---

### 5.3 Differential Privacy Agent

**File**: `installer/runtime/agents/dp/dp_agent.py`

Implements DP-SGD with multiple noise mechanisms and real RDP epsilon accounting.

**Supported Mechanisms**: `gaussian`, `laplace`, `uniform`, `exponential`, `student_t`, `none`

**DP-SGD Process**:
1. Flatten all model parameters into a single vector
2. Compute L2 norm
3. Clip: `flat = flat * (clip_norm / (l2_norm + 1e-12))` if `l2 > clip_norm`
4. Add noise: `noisy = flat + Normal(0, noise_mult * clip_norm)` (Gaussian)
5. Unflatten back to state dict

**RDP Epsilon Accounting** (Mironov 2017):
```
RDP(α) = α / (2 · noise_multiplier²)
ε(δ)   = min_α [ RDP(α) + log(1/δ) / (α - 1) ]   for α ∈ {2, ..., 256}
```

**Output**:
```python
{
  "receipt":        dict,       # HMAC-signed receipt
  "receipt_uri":    "file://...",
  "update_uri":     "file://...",  # encrypted DP-noised update
  "l2_norm_before": float,
  "l2_norm_after":  float,
  "epsilon_spent":  float,      # real RDP-derived epsilon
}
```

---

### 5.4 Encryption Agent

**File**: `installer/runtime/agents/enc/enc_agent.py`

Final encryption stage before upload. Modes:

| Mode | Description |
|---|---|
| `aes` (default) | AES-GCM via SecureStore (update already encrypted at rest) |
| `he_ckks` | Homomorphic Encryption via Pyfhel CKKS scheme |
| `kms_envelope` | AWS KMS envelope encryption (requires `kms_key_id`) |

In `aes` mode, since the DP update is already AES-GCM encrypted by SecureStore, the agent simply wraps the existing URI in a new receipt for chain-of-custody purposes. This avoids double-encryption overhead while preserving the audit trail.

---

### 5.5 Aggregator Agent (Server-side Python)

**File**: `server/aggregator_agent/aggregator.py`

Invoked as a subprocess by the Rust orchestrator after enough verified receipts arrive.

**Aggregation Strategies**:

| Strategy | Description | Byzantine-robustness |
|---|---|---|
| `trimmed_mean` (default) | Sort per-coordinate, trim top/bottom `trim_ratio` (default 10%), average remainder | Medium |
| `median` | Coordinate-wise median | High |
| `krum` | Select update closest to others (Blanchard et al.) | High (f=1 by default) |
| `mean` | Weighted average by sample count | None |

**Safety after aggregation**:
- Per-parameter delta clamp: `MAX_PARAM_DELTA = 1e-3`
- Global L2 norm scaling to `MAX_GLOBAL_NORM = 1.0`

**Security**:
- All update paths canonicalized and validated against `server_root`
- Updates decrypted via SecureStore (`AES-GCM-SecureStore` scheme)
- `weights_only=True` on `torch.load()` — prevents pickle exploit
- Output encrypted via SecureStore before writing to disk
- Aggregation receipt written to HMAC-chained ledger

**CLI** (called by Rust):
```bash
python3 aggregator.py --server-root ~/.federated/server --round-id 3
# Outputs: GLOBAL_MODEL_PATH=/abs/path/to/round_3.bin
```

---

### 5.6 Orchestration Agent (Rust Server)

**File**: `server/orchestration_agent/src/`

A Rust/Tokio async gRPC server implementing a dual-port architecture.

**Port 50051** — `EnrollmentService` (server-TLS only, no client certificate):
- `RequestEnrollment` — generates OTP for administrator, stores pending enrollment
- `EnrollDevice` — validates OTP, signs CSR with CA, stores device record

**Port 50052** — `OperationalService` (full mTLS mandatory):
- `GetRound` — returns current round metadata + global model availability flag
- `UploadUpdate` — client-streaming; verifies per-chunk SHA-256; saves to filesystem
- `SubmitReceipt` — verifies ECDSA signature + payload hash; triggers aggregation when threshold reached
- `DownloadGlobalModel` — server-streaming; sends global model with per-chunk + full-model SHA-256

**Aggregation trigger**: When verified receipt count ≥ `FL_MIN_UPDATES_FOR_AGGREGATION` (default: 3), spawns Python aggregator subprocess asynchronously (non-blocking tokio task).

---

## 6. Federated Learning Algorithms

Implemented in `installer/fl_algorithm_comparison.py`.

### 6.1 FedAvg (McMahan et al. 2017)

Standard federated averaging. Each client trains locally for `E` epochs, sends delta `Δw`. Server aggregates weighted by sample count.

```
w_{t+1} = Σ_k (n_k / N) · (w_t + Δw_k)
```

### 6.2 FedProx (Li et al. 2020)

Adds a proximal regularization term to local objective to constrain client drift:

```
h_k(w; w_t) = F_k(w) + (μ/2) · ||w - w_t||²
```

Default `μ = 0.01`.

### 6.3 SCAFFOLD (Karimireddy et al. 2020)

Uses control variates (`c_i`, `c`) to correct for client drift. Each client maintains a local control variate updated after every round:

```
c_i^+ = c_i - c + (w_i^0 - w_i^T) / (K · lr)
```

### 6.4 FedAdam / FedYogi (Reddi et al. 2021)

Server-side adaptive optimizers applied to the aggregated delta.

- **FedAdam**: Standard Adam moment updates on server side
- **FedYogi**: Yogi second-moment update (avoids aggressive LR increase):
  ```
  v_t = v_{t-1} + (1-β₂) · sign(g²-v) · g²
  ```

### 6.5 Aggregation Strategies

| Strategy | Robustness | Notes |
|---|---|---|
| `mean` | None | Baseline |
| `trimmed_mean` | Byzantine (10% trim ratio) | Default |
| `median` | Byzantine | Slower |
| `krum` | Byzantine | Requires ≥5 clients |

### 6.6 Hospital Client Grouping (FIX-14)

Patients are grouped into hospital-style clients using stratified round-robin assignment, ensuring each hospital client receives a mix of both depressed and not-depressed patients. This prevents the single-class client degeneracy that caused FedAvg to collapse to majority-class prediction (acc=0.5, F1=0.0).

---

## 7. Privacy Mechanisms

### 7.1 Differential Privacy (ε, δ)-DP

**Implementation**: `dp_agent.py`, `fl_algorithm_comparison.py`

- **Mechanism**: Gaussian with sensitivity = `clip_norm`
- **Noise**: `σ = noise_multiplier × clip_norm`, added per-parameter
- **Accounting**: Rényi DP (RDP) accountant, optimized over α ∈ {2,...,512}
- **Amplification**: Privacy amplification by Poisson subsampling (Wang et al. 2019)
- **FIX-6**: Noise correctly divided by batch_size (was 8× too large before fix)

**DP budget table** (typical setup, δ=1e-5, 30 rounds):

| noise_mult (σ) | Final ε | Meets ε≤8 |
|---|---|---|
| 0.5 | ~120 | ❌ |
| 1.0 | ~8.5 | ❌ |
| 1.1 | ~6.8 | ✅ |
| 1.5 | ~3.1 | ✅ |
| 2.0 | ~1.8 | ✅ |

### 7.2 Homomorphic Encryption

**Implementation**: `enc_agent.py`

- **Scheme**: CKKS (approximate arithmetic, suitable for real-valued gradients)
- **Library**: Pyfhel (Python wrapper for Microsoft SEAL)
- **Parameters**: `n=2^14`, `scale=2^30`
- Alternative: BFV for integer arithmetic
- Note: HE is computationally expensive; used selectively for aggregation or inference use cases

### 7.3 Secure Aggregation

**Implementation**: Rust `OperationalService` + Python aggregator

The system implements a simplified secure aggregation: updates are encrypted at rest (AES-GCM) and transmitted over mTLS. A full Bonawitz et al. protocol with masking and Shamir secret sharing is documented in the synopsis as future work.

### 7.4 Key Derivation

All keys are derived from a single master secret using HKDF-SHA256:

```python
derived_key = HKDF(SHA256, length=32, salt=None,
                   info=f"{agent}:{context}".encode()
                  ).derive(master_key)
```

This provides cryptographic isolation between agents and contexts. The master secret is TPM-sealed (Linux PCRs 0,2,4,7) or stored via DPAPI on Windows (plaintext fallback documented as risk).

---

## 8. Security System

### 8.1 File Integrity

**File**: `installer/security/integrity.py`

A ten-bypass-proof integrity system:

| Bypass Fixed | Mechanism |
|---|---|
| BYPASS-1 | Write-once install token — `write_baseline()` requires single-use token |
| BYPASS-2 | Baseline file made read-only/immutable after installation |
| BYPASS-3 | `integrity/` excluded from hash scope but baseline.sha256 itself is TPM-signed |
| BYPASS-4 | `max_violations=1` in `IntegrityWatcher` (was 2) |
| BYPASS-5 | Real-time inotify watcher (Linux) responds in <100ms (not 300s poll) |
| BYPASS-6 | Baseline TPM-signed with ECDSA P-256 device key |
| BYPASS-7 | Agent files frozen with `chattr +i` (Linux) / `chmod 0o444` after install |
| BYPASS-8 | Integrity checker's own source included in hash scope |
| BYPASS-9 | Seccomp filter available via `apply_seccomp_filter()` (requires python-seccomp) |
| BYPASS-10 | No sleep before integrity check; random 0–99ms jitter only |

**Hash Algorithm**: SHA3-256 (SHA-256 replaced for collision resistance)

**Tree Hash Structure**:
```
For each file in sorted(scope):
  hash.update(struct.pack(">I", len(path_bytes)))
  hash.update(path_bytes)
  hash.update(struct.pack(">Q", file_size))
  hash.update(file_content)
```

**Integrity Scope**: `bin/`, `runtime/`, `agents/`, `core/`, `installer/security/`

**Excluded**: `logs/`, `data/`, `venv/`, `deps/`, `tpm/`, `secrets/`, `state/`, `__pycache__/`, `keys/`, `integrity/`, `configs/`

### 8.2 Anti-Debug

**File**: `installer/security/anti_debug.py`

| Platform | Checks |
|---|---|
| Linux | `ptrace(PTRACE_TRACEME)`, `/proc/self/status` TracerPid, `LD_PRELOAD`/`LD_DEBUG` env vars, timing anomaly (1M iterations benchmark) |
| Windows | `IsDebuggerPresent()`, `PYTHONINSPECT`/`PYTHONDEBUG`/`PYDEVD_LOAD_VALUES_ASYNC` env vars, timing anomaly (relaxed threshold 0.5s) |

### 8.3 Self-Destruct

**Files**: `installer/security/self_destruct.py`, `installer/runtime/self_destruct.py`

Triggered on integrity failure, TPM unseal failure, debugger detection, or concurrent runtime detection:
1. Overwrites each file with `os.urandom(file_size)` before unlinking
2. Recursively removes `~/.federated/`
3. Calls `os._exit(1)` (hard exit, no cleanup hooks)

### 8.4 Runtime Guard (v2)

**File**: `installer/runtime/runtime_guard.py`

Ordered checks at startup:
1. Random jitter 0–99ms (prevents timing-based attach)
2. Anti-debug (ptrace + TracerPid + LD_PRELOAD)
3. Core dump disabled (`setrlimit(RLIMIT_CORE, 0)` + `PR_SET_DUMPABLE=0`)
4. `/proc/self/maps` scan for unexpected shared libraries
5. Canary file integrity check
6. SHA3-256 integrity tree verification (vs TPM-signed baseline)
7. TPM master secret unseal (≥16 bytes)
8. No-root enforcement (Linux)
9. PID-aware single-instance lock (`~/.federated/state/runtime.lock`)

### 8.5 Canary Files

**File**: `installer/security/military_security.py` → `CanaryMonitor`

Planted in `~/.federated/data/secure_store/.canaries/`:
- N files (default 5) with random content, SHA-256 recorded
- Any modification or deletion detected on each check
- Detects filesystem-level probing before the full integrity hash runs

### 8.6 Real-time File Watcher

**File**: `installer/security/integrity.py` → `_InotifyWatcher`

Linux-only, uses `inotify_init` + `inotify_add_watch` directly via ctypes. Watches all scope directories and subdirectories. Responds in <100ms on `IN_MODIFY`, `IN_CREATE`, `IN_DELETE`, `IN_ATTRIB`, `IN_MOVE`.

### 8.7 Eval/Exec Audit Hook

**File**: `installer/security/military_security.py` → `_audit_no_eval_exec()`

Installs `sys.addaudithook()` (Python 3.8+) that intercepts any `eval`, `exec`, or `compile` call and immediately triggers self-destruct.

### 8.8 Certificate Pinning

**File**: `installer/security/military_security.py` → `verify_cert_pin()`

Pins CA public key SHA-256. Configured via `CA_PUBKEY_PIN_SHA256` constant. ⚠️ Currently still requires update after cert rotation.

### 8.9 Receipt Nonce / Replay Prevention

**File**: `installer/runtime/runtime_guard.py` → `generate_receipt_nonce()`, `validate_and_consume_nonce()`

- 256-bit random nonce per receipt
- Persisted to `~/.federated/state/nonce_store.json`
- Capped at 10,000 entries (oldest purged)

---

## 9. Multimodal Data Pipeline

### 9.1 Audio Features

| Feature Set | Tool | Dimensions |
|---|---|---|
| eGeMAPS (v02) | OpenSMILE `SMILExtract` | 88 features (23 LLDs + functionals) |
| wav2vec2 | `facebook/wav2vec2-base-960h` via Transformers | configurable (default 512, pooled) |
| Prosody | torchaudio | 3 (energy, pitch mean, ZCR) |
| BoAW | Bag of Audio Words CSV | variable |

### 9.2 Video Features

| Feature Set | Tool | Notes |
|---|---|---|
| Action Units (AU) | OpenFace `FeatureExtraction.exe` | 17 AUs (r + c variants) |
| Gaze | OpenFace | x,y,z angles per eye |
| Head Pose | OpenFace | Rx,Ry,Rz rotation |
| Face landmark | OpenFace | 68 2D + 68 3D points |
| Face blur | OpenCV Gaussian blur (kernel 99×99) | Privacy protection before storage |

### 9.3 Text Features

| Feature Set | Tool | Notes |
|---|---|---|
| Contextual embeddings | MentalBERT (`mental/mental-bert-base-uncased`) | 768-d CLS pooled |
| ASR Transcription | openai-whisper (primary) / HF pipeline (fallback) | Session-mean vector for FL training |
| PII scrubbing | spaCy NER (`PERSON`, `GPE`, `ORG`) + regex | Phone, email patterns |

### 9.4 Voice Activity Detection (VAD)

1. WebRTC VAD (`webrtcvad`) — primary, frame-level 30ms
2. Energy-based VAD — fallback using librosa/wave, configurable threshold
3. Full-audio fallback — if both fail, single segment covering full duration

### 9.5 Speaker Diarization

1. pyannote.audio (requires `HF_TOKEN`) — primary
2. VAD-based 2-speaker heuristic — fallback (even/odd segment assignment)

### 9.6 ASR (Automatic Speech Recognition)

- **Primary**: openai-whisper (model: `small` by default, configurable)
- **Fallback**: HF Transformers `AutoModelForCTC` pipeline (`openai/whisper-small`)
- **FIX-2**: When whisper is configured but unavailable, explicit WARNING is now logged (previously silent fallback caused confusing quality degradation)

---

## 10. gRPC Protocol & Network Design

### 10.1 Proto Definition

**File**: `server/orchestration_agent/proto/orchestrator.proto`

```protobuf
service Orchestrator {
  rpc RegisterDevice       (CSR)                      returns (Certificate);
  rpc RequestEnrollment    (EnrollmentRequest)         returns (EnrollmentRequestAck);
  rpc EnrollDevice         (EnrollRequest)             returns (EnrollResponse);
  rpc GetRound             (DeviceId)                  returns (RoundMetadata);
  rpc UploadUpdate         (stream UpdateChunk)        returns (UploadAck);
  rpc SubmitReceipt        (Receipt)                   returns (Ack);
  rpc DownloadGlobalModel  (RoundRequest)              returns (stream ModelChunk);
}
```

### 10.2 UpdateChunk (client → server streaming)

```protobuf
message UpdateChunk {
  string  session_id   = 1;
  uint64  round_id     = 2;
  bytes   device_id    = 3;   // SHA-256(device_pubkey)
  uint64  chunk_index  = 4;   // 0-based, must be sequential
  uint64  total_chunks = 5;
  bytes   data         = 6;   // encrypted update bytes (1MB per chunk)
  bytes   chunk_hash   = 7;   // SHA-256(data) — server verifies each chunk
}
```

### 10.3 Receipt Submission

```protobuf
message Receipt {
  bytes   device_id     = 1;
  uint64  round_id      = 2;
  bytes   payload_hash  = 3;   // SHA-256 of ALL uploaded bytes
  double  epsilon_spent = 4;   // from real RDP accountant
  bytes   signature     = 5;   // ECDSA-DER over (device_id || round_id_BE8 || payload_hash)
  string  enc_handle    = 6;   // file:// URI of stored update
  string  scheme        = 7;   // "AES-GCM-DP-ECDSA"
  string  nonce         = 8;   // 256-bit hex random nonce
}
```

### 10.4 Channel Architecture

| Port | TLS Mode | Purpose | Service |
|---|---|---|---|
| 50051 | Server-TLS only (no client cert) | Device enrollment | `EnrollmentService` |
| 50052 | Full mTLS (client cert mandatory) | Operational FL | `OperationalService` |

**Security Fixes Applied**:
- FIX-GRPC-1: Removed `ssl_target_name_override` (was disabling hostname verification)
- FIX-GRPC-2: Added `UploadUpdate` client-streaming method
- FIX-GRPC-3: Added `DownloadGlobalModel` server-streaming method

### 10.5 gRPC Channel Options

```python
_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms",              10_000),
    ("grpc.keepalive_timeout_ms",            5_000),
    ("grpc.keepalive_permit_without_calls",      1),
    ("grpc.http2.max_pings_without_data",        0),
    ("grpc.max_send_message_length",    8 * 1024 * 1024),
    ("grpc.max_receive_message_length", 8 * 1024 * 1024),
]
```

---

## 11. Encrypted Storage (SecureStore)

**File**: `installer/runtime/core/centralized_secure_store.py`

### Binary Format (v0x01)

```
[1 byte  version = 0x01        ]
[1 byte  agent_len             ]
[N bytes agent name            ]
[1 byte  context_len           ]
[M bytes context               ]
[12 bytes nonce (AES-GCM)      ]
[remaining: ciphertext + 16-byte GCM authentication tag]
```

### Key Derivation

```python
HKDF(SHA256, length=32, salt=None,
     info=f"{agent}:{context}".encode()
    ).derive(master_key)
```

**Context** is derived from the parent directory name of the file URI. This isolates encryption keys by agent and storage location.

### Path Safety Fixes

- **FIX-SS-1**: Uses `Path.is_relative_to()` (Python 3.9+) or explicit `os.sep` suffix check to prevent adjacent-directory path traversal attacks
- **FIX-SS-2**: Guards against agent/context names > 200 bytes (would cause `bytes([n])` overflow)
- **FIX-SS-3**: Validates all header offsets before indexing into the buffer

### Single Master Key

The master key lives at `~/.federated/data/secure_store/master.key` (base64-encoded). All agents share this one key, with per-agent/context HKDF isolation. This is intentional — agents must be able to share encrypted artifacts.

---

## 12. Audit & Compliance System

### 12.1 Receipt Manager

**File**: `installer/runtime/core/centralised_receipts.py`

Every operation produces an HMAC-SHA256 signed receipt:

```python
{
  "agent":      "dp-agent",
  "session_id": "sess-abc123",
  "operation":  "dp_process_update",
  "params":     { "clip_norm": 1.0, "epsilon_spent": 4.23, ... },
  "outputs":    ["file:///..."],
  "timestamp":  "2025-08-10T12:34:56.789Z",
  "signature":  "<base64 HMAC-SHA256>"
}
```

### 12.2 Rust Ledger

**File**: `server/orchestration_agent/src/ledger.rs`

An append-only filesystem ledger with:
- HMAC-SHA256 chain: each entry links to previous via `HMAC(key, prev_hmac | payload_hash)`
- In-process `Mutex` + `O_APPEND` for atomic writes
- Configurable key via `RECEIPT_CHAIN_KEY` environment variable (32-byte hex)
- `get_last_hmac(path, round_id)` for chain traversal

### 12.3 Compliance Evidence Generation

**File**: `installer/security/research_evaluation_suite.py` → `ComplianceEngine`

Generates machine-readable DP privacy certificates:

```json
{
  "certificate_version": "1.0",
  "algorithm": "Gaussian Mechanism + RDP Accounting",
  "references": ["Mironov 2017 (RDP)", "Wang et al. 2019 (amplification)", "Abadi et al. 2016 (DP-SGD)"],
  "parameters": { "noise_multiplier": 1.1, "clip_norm": 1.0, "sample_rate": 0.4, "delta": 1e-5 },
  "guarantee": {
    "epsilon": 6.8,
    "privacy_guarantee": "(6.8000, 1e-05)-DP",
    "meets_epsilon_8": true,
    "meets_epsilon_4": false
  },
  "receipts_summary": { "n_receipts": 80, "chain_intact": true }
}
```

---

## 13. TPM & Hardware Security

### 13.1 Linux (tpm2-tools)

**File**: `installer/security/tpm_attestation.py`

```bash
# Primary key creation (ECC P-256)
tpm2_createprimary -C o -G ecc -g sha256 -c ~/.federated/tpm/primary.ctx

# Device signing key (non-exportable, fixedtpm)
tpm2_create -C primary.ctx -G ecc -g sha256 \
  -a "sign|fixedtpm|fixedparent|sensitivedataorigin|userwithauth" \
  -u device.pub -r device.priv

# Load and export public key
tpm2_load -C primary.ctx -u device.pub -r device.priv -c device.ctx
tpm2_readpublic -c device.ctx -f pem -o device_pubkey.pem

# PCR-seal master secret (PCRs 0,2,4,7)
tpm2_create -C o -i secret.bin -L sha256:0,2,4,7 -u sealed.pub -r sealed.priv
tpm2_load -C o -u sealed.pub -r sealed.priv -c sealed_secret.ctx
tpm2_unseal -c sealed_secret.ctx  # → master secret bytes
```

### 13.2 Windows (CNG NCrypt)

**File**: `installer/windows_signer/src/main.rs`

Rust binary using Windows CNG `NCrypt` API:
- Key storage: Microsoft Platform Crypto Provider (TPM-backed on TPM machines, software on others)
- Algorithm: ECDSA P-256 (`ECDSA_P256`)
- Key name: `FederatedDeviceKey` (persistent in TPM key storage)
- Commands: `--init`, `--pubkey <file>`, `--sign` (reads stdin, writes DER signature to stdout)

### 13.3 Signing and Verification

Receipt signature covers: `device_id || round_id_BE8 || payload_hash`

Server verification (`receipts.rs`):
1. Parse PEM → base64-decode → DER
2. Skip 26-byte SPKI header for P-256
3. Verify `ec_point[0] == 0x04` (uncompressed marker)
4. `VerifyingKey::from_sec1_bytes(ec_point[8..73])`
5. `verifying_key.verify(msg, &Signature::from_der(sig))`

---

## 14. Installer & Enrollment

### 14.1 Two-Phase Installation

**Phase A — Software Setup** (no server contact):

```
[1]  Anti-debug check
[2]  Create ~/.federated/ secure directory tree (chmod 0700)
[3]  Copy agents, configs, gRPC stubs, CA cert
[3b] Create Python venv (finds real system Python even in frozen .exe)
[4]  TPM identity provisioning
[5]  (Windows) VC++ runtime verification
[6]  pip install all dependencies into venv
[7]  OpenFace install
[8]  openSMILE install
     FFmpeg install + SHA-256 verify
[9]  Windows binary verification
[10] MentalBERT model install (HuggingFace Hub or local payload)
     Generate one-time install write token (captured in memory only)
```

**Phase B — Enrollment** (requires server):

```
[B1] Request enrollment OTP
     → server generates OTP, prints to admin console
     → device receives fingerprint
[B2] User enters OTP
     → complete_enrollment()
     → server validates OTP, signs CSR, returns client certificate
[11] TPM secret sealing
[12] Write install state JSON
[13] write_baseline() with one-time token
      → computes SHA3-256 tree hash
      → TPM-signs baseline
      → freezes all agent files (chattr +i / chmod 0444)
      → creates INSTALL_LOCK (prevents future write_baseline() calls)
[14] Register daemon (systemd user service / Windows Task Scheduler / crontab)
```

### 14.2 Virtual Environment

- Located at `~/.federated/venv/`
- Python path: `~/.federated/venv/bin/python` (Linux) or `~/.federated/venv/Scripts/python.exe` (Windows)
- FIX-6: When running as PyInstaller frozen `.exe`, locates real system Python via `py` launcher, then `python`/`python3` on PATH, then common install paths

### 14.3 OTP Security

**File**: `server/orchestration_agent/src/otp.rs`

- 6-digit cryptographically random OTP (`rand::Rng.gen_range(100_000..=999_999)`)
- Expiry: 600 seconds (10 minutes) — FIX-OTP-1 (was incorrectly 6000s)
- Rate limiting: 5 failures → 5-minute lockout per device
- One-time use: marked `used=true` on consumption, pruned on expiry

### 14.4 GUI Installer

**File**: `installer/installer_gui.py`

- Tkinter-based two-panel GUI
- UAC elevation on Windows (ShellExecuteW with "runas" verb)
- Phase 1 panel: server address input, install log
- Phase 2 panel: device fingerprint display + copy button, OTP entry, enrollment button
- All installer I/O redirected to log widget via StringIO

---

## 15. Server (Rust Orchestrator)

### 15.1 Building

```bash
cd server/orchestration_agent
cargo build --release
# Binary: target/release/orchestrator
```

### 15.2 Configuration

**File**: `config/orchestrator.toml`

```toml
[server]
addr      = "0.0.0.0:50051"   # enrollment port — server-TLS only
mtls_addr = "0.0.0.0:50052"   # operational port — full mTLS

[tls]
ca_cert     = "certs/ca.pem"
ca_key      = "certs/ca.key"
server_cert = "certs/server.pem"
server_key  = "certs/server.key"
```

### 15.3 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FL_SERVER_ROOT` | `~/.federated/server` | Canonical server data directory |
| `FL_EPSILON_MAX` | `8.0` | Maximum ε budget per round |
| `FL_MIN_UPDATES_FOR_AGGREGATION` | `3` | Number of verified receipts before aggregation triggers |
| `FL_TRIM_RATIO` | `0.1` | Fraction trimmed from each end in trimmed mean |
| `FL_MAX_PARAM_DELTA` | `1e-3` | Per-parameter clamp on aggregated delta |
| `FL_MAX_GLOBAL_NORM` | `1.0` | Maximum L2 norm of aggregated delta |
| `RECEIPT_CHAIN_KEY` | ephemeral | 32-byte hex HMAC key for ledger chaining |
| `CONFIG_PATH` | `config/orchestrator.toml` | Path to config file |
| `AGGREGATOR_LOG_LEVEL` | `INFO` | Python aggregator log level |

### 15.4 Certificate Generation

```bash
cd server/orchestration_agent
bash certs/gen_certs.sh <SERVER_IP>
# Example: bash certs/gen_certs.sh 192.168.1.7
```

The script:
1. Generates 4096-bit RSA CA key + self-signed cert (10-year validity)
2. Generates 2048-bit RSA server key + CSR + signed cert (1-year)
3. Server cert SAN includes: `IP.1=<SERVER_IP>`, `IP.2=127.0.0.1`, `DNS.1=localhost`
4. Copies `ca.pem` to `installer/runtime/keys/ca.pem`

**After running**: rebuild server (`cargo build --release`) and reinstall client (or replace `~/.federated/keys/ca.pem`).

### 15.5 Server Filesystem Layout

```
~/.federated/server/
├── devices/
│   └── <device_id_hex>.json    # enrollment record per device
├── rounds/
│   └── <round_id>/
│       └── updates/
│           └── <device_id_hex>.bin   # encrypted update file
├── global_models/
│   └── round_<id>.bin               # aggregated global model
└── ../logs/
    └── audit_ledger.log              # HMAC-chained append-only ledger
```

---

## 16. Configuration Reference

### 16.1 `local_config.yaml` (Client)

**File**: `installer/runtime/configs/local_config.yaml`

```yaml
mode: interactive   # interactive | batch | continuous | session | text

ingest:
  video:
    enabled: true
    params:
      openface:
        binary_path: "~/.federated/deps/windows/OpenFace/FeatureExtraction.exe"
        haar_path:   "~/.federated/deps/windows/OpenFace/classifiers/..."
  audio:
    enabled: true
    sr: 16000
    params:
      features:
        egemaps:
          enabled: true
          opensmile_binary: "~/.federated/deps/.../SMILExtract.exe"
          opensmile_config: "~/.federated/deps/.../eGeMAPSv02.conf"
        wav2vec2:
          enabled: true
          model: "facebook/wav2vec2-base-960h"
  text:
    enabled: true
    asr_model: "small"              # openai-whisper model name
    asr_hf_model: "openai/whisper-small"  # HF fallback

text_pipe:
  asr_backend: "whisper"  # "whisper" or "hf"
  asr_enabled: true

storage:
  root: "~/.federated/data/secure_store"
  encrypt: true

limits:
  max_concurrent_sessions: 4
  max_upload_mb: 200
```

### 16.2 Config Validation

**File**: `installer/runtime/config_validator.py`

Validated at load time:
- `storage.root` must be non-empty
- `mode` must be one of the five valid modes
- `ingest.audio.sr` must be a standard sample rate
- `text_pipe.asr_backend` must be `"whisper"` or `"hf"`
- `text_pipe.asr_model` must be set
- OpenFace / openSMILE paths warned if not found (not error — Linux may not have bundled binaries)

---

## 17. Installation Guide

### 17.1 Prerequisites

**Server**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv

# Install protobuf compiler (for Rust gRPC)
sudo apt install protobuf-compiler

# Install tpm2-tools (Linux)
sudo apt install tpm2-tools tpm2-abrmd

# Install OpenSSL (for CSR signing)
sudo apt install openssl
```

**Client (Linux)**:
```bash
sudo apt install python3.11 python3.11-venv ffmpeg
sudo apt install tpm2-tools        # if TPM available
sudo apt install libx11-dev        # for OpenFace dependencies
pip install --break-system-packages openai-whisper  # requires tiktoken
```

**Client (Windows)**:
- Python 3.11+ (tick "Add to PATH" during install)
- Visual C++ 2015–2022 x64 Redistributable
- ffmpeg (auto-installed by installer, or manually from ffmpeg.org)
- TPM 2.0 chip (recommended, software fallback available)

### 17.2 Server Setup

```bash
cd server/orchestration_agent

# 1. Generate certificates
bash certs/gen_certs.sh <YOUR_SERVER_IP>

# 2. Build
cargo build --release

# 3. Set environment
export FL_SERVER_ROOT="$HOME/.federated/server"
export RECEIPT_CHAIN_KEY=$(openssl rand -hex 32)
export FL_EPSILON_MAX=8.0
export FL_MIN_UPDATES_FOR_AGGREGATION=3

# 4. Run
./target/release/orchestrator
# Enrollment port: 0.0.0.0:50051
# Operational port: 0.0.0.0:50052
```

### 17.3 Client Installation

**GUI (Windows)**:
```
Run installer.exe as Administrator
→ Enter server address (e.g., 192.168.1.7:50051)
→ Click "Install Software"
→ Wait for Phase 1 to complete
→ Give Device Fingerprint to admin
→ Enter OTP from admin
→ Click "Complete Enrollment"
```

**CLI (Linux)**:
```bash
cd installer
python installer_core.py
# Enter server address when prompted
# Enter OTP when prompted
```

### 17.4 Install HF_TOKEN Securely (IMPORTANT)

**Never** commit HF_TOKEN to git. After revoking the exposed token:

```bash
# Option 1: huggingface-cli
pip install huggingface_hub
huggingface-cli login

# Option 2: environment variable (set before running installer)
export HF_TOKEN="hf_NEW_TOKEN_HERE"

# Option 3: .env file (in pipelines/ directory, gitignored)
echo "HF_TOKEN=hf_NEW_TOKEN_HERE" > installer/runtime/agents/lda/pipelines/.env
# Verify it's in .gitignore
cat installer/runtime/agents/lda/pipelines/.gitignore  # should list .env
```

---

## 18. Running the System

### 18.1 Client Modes

```bash
# Run as scheduled FL daemon (captures session, trains, uploads, sleeps 1 hour)
FED_SERVER="192.168.1.7:50052" python ~/.federated/bin/federated-client daemon

# Run once (reuses recent session or captures fresh one)
FED_SERVER="192.168.1.7:50052" python ~/.federated/bin/federated-client run-once

# Continuous background monitoring mode (5-min windows, no FL update)
FED_DAEMON_MODE=continuous FED_SERVER="192.168.1.7:50052" \
  python ~/.federated/bin/federated-client daemon
```

### 18.2 FL Algorithm Comparison

```bash
cd installer

# Run all FL algorithms on DAIC-WOZ data
python fl_algorithm_comparison.py \
  --data_dir ./data \
  --rounds 30 \
  --use_mentalbert \
  --n_hospitals 8 \
  --noise_mult 1.1 \
  --eval_threshold 0.4

# Without DP
python fl_algorithm_comparison.py --data_dir ./data --rounds 30 --no_dp

# With pre-trained global model warm-start
python fl_algorithm_comparison.py \
  --data_dir ./data \
  --global_model_path ~/.federated/data/global_models/global_round1.pt

# Output: fl_daic_results/results.json, fl_daic_results/fl_daic_comparison.png
```

### 18.3 Research Evaluation Suite

```bash
cd installer
python security/research_evaluation_suite.py \
  --n_rounds 20 \
  --n_clients 8 \
  --noise_mult 1.1 \
  --seed 42 \
  --plots

# Outputs:
#   eval_results/results.json
#   eval_results/privacy_certificate.json
#   eval_results/plots/research_evaluation_dashboard.png
```

### 18.4 Aggregator Test (Local)

```bash
cd server/aggregator_agent
python test_aggregator_local.py
# Creates 3 synthetic encrypted updates, runs trimmed mean aggregation
```

### 18.5 Pipeline Modes

```python
# Session mode (structured therapy session → FL update)
from runtime.pipeline import run_pipeline
run_pipeline(stub, device_id, master_secret,
             session_dir=Path("/path/to/session"),
             pipeline_mode="session")

# Continuous monitoring (inference only)
run_pipeline(stub, device_id, master_secret,
             session_dir=Path("/path/to/window"),
             pipeline_mode="continuous")
```

---

## 19. Dataset — DAIC-WOZ

**Western Psychiatric Institute and Clinic — Distress Analysis Interview Corpus**

### Structure Expected

```
data/
  labels.csv                    # Participant_ID, PHQ8_Binary, PHQ8_Score
  <ID>_P/
    features/
      <ID>_OpenSMILE*.csv       # eGeMAPS audio features
      <ID>_OpenFace2.csv        # Action Units, gaze, pose
      <ID>_BoAW_openSMILE.csv   # Bag of Audio Words
      <ID>_BoVW_openpose.csv    # Bag of Visual Words
      <ID>_CNN_*.csv            # CNN visual features
      <ID>_Transcript.csv       # Turn-by-turn conversation
    <ID>_AUDIO.wav
```

### Labels

- **PHQ8_Binary**: 0 (PHQ-8 score < 10) or 1 (PHQ-8 score ≥ 10 = depressed)
- **PHQ8_Score**: Continuous score 0–24

### Hospital Client Grouping

Patients are assigned to `n_hospitals` (default: 8) hospital clients via stratified round-robin:
- Depressed and not-depressed patients interleaved across clients
- Every hospital has both classes → meaningful gradients
- Prevents FedAvg majority-class collapse (FIX-14)

---

## 20. Model — MentalBERT Multimodal Fusion

### 20.1 MentalBERT

- Model: `mental/mental-bert-base-uncased` (HuggingFace Hub)
- Fallback: `bert-base-uncased` (identical architecture)
- Embedding: 768-d CLS token, mean-pooled across utterances per session
- **FIX-7**: Session-mean text vector used for both training and test (eliminates train/test distribution mismatch)

### 20.2 DepressionNet (FL version)

Mirrors `MultimodalFusionModel` from centralized training for checkpoint compatibility:

```
audio_encoder  = SmallMLP(audio_dim → max(64, audio_dim//2) → 128)
visual_encoder = SmallMLP(visual_dim → max(64, visual_dim//2) → 128)
fusion         = Linear(1024 → 256) → LayerNorm(256) → ReLU → Dropout
               → Linear(256 → 128) → ReLU → Dropout
               → Linear(128 → 2)
```

**LayerNorm vs BatchNorm** (FIX-3): LayerNorm weight/bias shapes match BatchNorm (both `[hidden_dim]`), so centralized checkpoints load correctly. LayerNorm works for batch_size=1 during FL local training, while BatchNorm would crash.

### 20.3 Checkpoint Partial Loading

**FIX-13**: `_load_partial_checkpoint()` + `_selective_reinit_encoders()`:
- Skips keys with shape mismatches (e.g., `audio_encoder.net.0.weight` when audio_dim differs between centralized and FL setups)
- Returns `shape_skipped_keys` set
- Reinitializes **only** the first linear of encoders that had shape mismatches (via Kaiming init)
- Preserves all other correctly-loaded weights (was previously destroying 10/11 encoder keys)

### 20.4 Evaluation

**FIX-16**: Soft probability thresholding (default: 0.4) instead of argmax:

```python
probs = F.softmax(logits, dim=1)[:, 1]  # P(depressed)
preds = (probs >= threshold).astype(int)
```

Threshold 0.4 biases slightly toward detecting the minority class (depressed). `find_best_threshold()` searches threshold space on held-out set to maximize F1.

---

## 21. Evaluation & Research Suite

**File**: `installer/security/research_evaluation_suite.py`

### Section 1 — Privacy Analysis

- RDP epsilon trajectory across rounds for σ ∈ {0.5, 0.8, 1.0, 1.1, 1.5, 2.0}
- zCDP Gaussian bound
- Privacy amplification gain (Poisson subsampling)
- Checks ε ≤ 8 (HIPAA threshold) and ε ≤ 4 (strict)
- **FIX**: Overflow protection for `math.exp()` via threshold check + `log1p`/`expm1`

### Section 2 — Security Verification (8 tests)

| Test | What it verifies |
|---|---|
| AES-GCM confidentiality | Correct decrypt; tamper detection; nonce-reuse detectability |
| Path traversal prevention | `relative_to()` vs `startswith()` bug (FIX-SS-1) |
| HMAC receipt chain | Chain integrity, tamper detection at any link |
| Write-once install token | One-time use, replay rejection, wrong-token rejection |
| Timing-safe comparison | `hmac.compare_digest` within 2× timing ratio |
| Canary tamper detection | Planted canaries, modify one, verify detection |
| HKDF key isolation | 4 distinct keys for 4 (agent, context) pairs |
| SHA3-256 tree hash | Baseline vs tampered file detection |

### Section 3 — Attack Testbeds

**Membership Inference Attack (Shokri et al. 2017)**:
- Shadow model approach with 4 shadow models
- Meta-classifier: logistic regression, 5-fold CV
- Metrics: AUC-ROC with 95% CI (DeLong), TPR at 10% FPR, Cohen's d
- Tests: no-DP (σ=0), weak-DP (σ=0.5), strong-DP (σ=1.5)

**Gradient Inversion (Zhu et al. NeurIPS 2019)**:
- Gradient matching objective: `||∂L/∂W(x̂) − g_observed||²`
- Finite-difference gradient approximation
- Metrics: reconstruction MSE, SNR (dB), PSNR
- Tests: no-DP vs DP-defended (σ=1.1, C=1.0)

**Byzantine Poisoning**:
- Label-flip attack: train on inverted labels
- Scaling attack: amplify gradient 10×
- Fraction malicious: 30%
- Defense: trimmed mean (10% trim ratio)
- Metrics: F1 clean / F1 poisoned / F1 defended

**Free-rider Detection**:
- Zero-gradient detection via L2 norm threshold
- Cosine similarity to mean honest update

### Section 4 — Compliance Evidence

- 80 HMAC-chained receipts across 10 rounds × 8 clients
- Signature verification on all receipts
- Chain integrity verification
- Tamper detection (epsilon manipulation)
- DP privacy certificate generation
- Audit ledger statistics (min/max/mean/P95 ε)

### Statistical Hypotheses

| Hypothesis | Test |
|---|---|
| H1: DP reduces MIA AUC toward 0.5 | AUC reduction + Cohen's d effect size |
| H2: DP increases gradient inversion MSE | MSE increase factor |
| H3: Trimmed mean defends against poisoning | F1 recovery (pp) |
| H4: Default σ=1.1 achieves ε≤8 | Final epsilon vs threshold |

---

## 22. FL Algorithm Comparison Tool

**File**: `installer/fl_algorithm_comparison.py`

Comprehensive standalone benchmark of all 5 FL algorithms on DAIC-WOZ.

### Key Fixes Applied

| Fix | Description |
|---|---|
| FIX-1 | DepressionNet architecture mirrors centralized checkpoint (same layer names, same dims) |
| FIX-2 | `_load_partial_checkpoint()` filters by name AND shape (not just strict=False) |
| FIX-3 | LayerNorm replaces BatchNorm1d (works for batch_size=1 in FL local training) |
| FIX-4 | All FL warm-start blocks use `_load_partial_checkpoint()` |
| FIX-5 | `model_factory()` uses `fusion_hidden=256` (not old signature `hidden=64`) |
| FIX-6 | DP noise divided by batch_size (was 8× too large) |
| FIX-7 | Session-mean text vector for both train and test (was per-utterance in train) |
| FIX-8 | Orphaned checkpoint-reinit block removed |
| FIX-9 | FedYogi server optimizer properly instantiated |
| FIX-10 | Dead variable `first_w` removed |
| FIX-11 | Dead variable `new_client_c` removed from SCAFFOLD |
| FIX-12 | Double class-weight application removed (aggregation by sample count only) |
| FIX-13 | Selective encoder reinit (only mismatched first linear, not entire encoder) |
| FIX-14 | Hospital client grouping (stratified mix replaces single-patient clients) |
| FIX-15 | Adam optimizer for local training (SGD+momentum caused monotone collapse) |
| FIX-16 | Soft-probability threshold (default 0.4) replaces argmax |
| FIX-17 | Experiment deduplication before running (--no_dp caused silent overwrites) |

### Output Files

- `fl_daic_results/results.json` — per-round metrics for all experiments
- `fl_daic_results/fl_daic_comparison.png` — 4-panel plot (accuracy, F1, ε trajectory, privacy-utility tradeoff)
- `fl_daic_results/latex_table.tex` — LaTeX tables for paper

---

## 23. Hardware Requirements

### Client / Edge Device

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Any x86-64 with AES-NI | Intel Core i7+ / AMD Ryzen 7+ |
| RAM | 4 GB | 16 GB |
| GPU | None (CPU inference) | CUDA-capable (NVIDIA GTX 1060+) |
| Storage | 10 GB free | 50 GB SSD |
| TPM | Software fallback | TPM 2.0 chip (Intel PTT / AMD fTPM) |
| OS | Windows 10+, Ubuntu 20.04+, macOS 12+ | — |

### Server

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 32+ cores |
| RAM | 16 GB | 64 GB |
| GPU | None | Multiple NVIDIA A100/V100 for aggregation |
| Storage | 100 GB | 1 TB NVMe SSD |
| Network | 1 Gbps | 10 Gbps |
| HSM/KMS | Environment variable | Hardware HSM for production `RECEIPT_CHAIN_KEY` |

---

## 24. Dependencies & Tech Stack

### Python (Client)

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.6.0 | Model training & inference |
| `torchaudio` | 2.6.0 | Audio loading & resampling |
| `transformers` | 4.44.0 | MentalBERT tokenizer & model |
| `opacus` | (via Opacus) | DP-SGD reference implementation |
| `Pyfhel` | optional | Homomorphic encryption (CKKS/BFV) |
| `grpcio` | 1.62.2 | gRPC client |
| `protobuf` | 4.25.3 | Protocol buffers |
| `cryptography` | 42.0.5 | HKDF, AESGCM, RSA CSR generation |
| `pydantic` | 1.10.13 | Request validation |
| `PyYAML` | 6.0.2 | Config file parsing |
| `fastapi` | 0.116.1 | (Future) HTTP API |
| `openai-whisper` | 20240930 | ASR transcription |
| `spacy` + `en_core_web_sm` | 3.8.7 | NER-based PII scrubbing |
| `librosa` | 0.11.0 | Audio feature extraction |
| `webrtcvad` | 2.0.10 | Voice activity detection |
| `opencv-python` | 4.11.0.86 | Video processing |
| `pandas` | 2.3.2 | Tabular data handling |
| `pyarrow` | 21.0.0 | Parquet serialization |
| `scikit-learn` | 1.7.1 | Metrics, cross-validation |
| `numpy` | 1.26.4 | Numerical operations |
| `pymongo` | 4.7.2 | (Optional) MongoDB receipts |
| `motor` | 3.4.0 | (Optional) Async MongoDB |
| `boto3` | 1.40.32 | (Optional) AWS KMS |
| `pyannote.audio` | 3.1.1 | Speaker diarization |
| `mediapipe` | 0.10.11 | (Optional) Face landmarks |
| `huggingface_hub` | ≥0.24.0 | MentalBERT model download |
| `python-dotenv` | 1.0.1 | .env file loading |
| `scipy` | — | Statistical tests in eval suite |

### Rust (Server)

| Crate | Version | Purpose |
|---|---|---|
| `tokio` | 1.37 | Async runtime |
| `tonic` | 0.11 | gRPC server |
| `prost` | 0.12 | Protocol buffers |
| `p256` | 0.13 | ECDSA P-256 verification |
| `ecdsa` | 0.16 | ECDSA signature parsing |
| `sha2` | 0.10 | SHA-256 |
| `hmac` | 0.12 | HMAC-SHA256 |
| `rand` | 0.8 | OTP generation |
| `rcgen` | 0.13 | X.509 certificate generation |
| `dashmap` | 5.5 | Concurrent hash map |
| `serde` / `serde_json` | 1.0 | Serialization |
| `toml` | 0.8 | Config parsing |
| `hex` | 0.4 | Hex encoding |
| `base64` | 0.21 | Base64 encoding |
| `uuid` | 1.8 | UUID v4 |
| `once_cell` | 1 | Lazy statics |
| `dirs` | 5.0 | Home directory resolution |
| `anyhow` | 1.0 | Error handling |
| `tracing` | 0.1 | Structured logging |

### Rust (Windows Signer)

| Crate | Purpose |
|---|---|
| `windows = {version="0.56", features=["Win32_Security_Cryptography","Win32_Foundation"]}` | NCrypt ECDSA |
| `sha2` | SHA-256 digest |
| `base64` | PEM encoding |

### Native Binaries

| Binary | Platform | Purpose |
|---|---|---|
| `FeatureExtraction.exe` | Windows | OpenFace face analysis |
| `SMILExtract.exe` | Windows | OpenSMILE eGeMAPS extraction |
| `ffmpeg` | All | Audio/video processing |
| `tpm2_*` | Linux | TPM 2.0 operations |
| `openssl` | Server | CSR signing |

---

## 25. Known Bugs Fixed

A complete list of all documented fixes across the codebase:

| Fix ID | File | Description |
|---|---|---|
| BUG-1 / FIX-RMTREE-1 | `install_runtime.py` | `shutil.rmtree()` PermissionError on Windows read-only files after `freeze_all_agent_files()` |
| BUG-2 | `tpm_seal.py` | `CREATE_NO_WINDOW` (Windows-only flag) passed unconditionally on Linux |
| BUG-4 | `install_openface.py` | `SRC` pointed to openSMILE directory instead of OpenFace |
| FIX-6 (venv) | `installer_core.py` | Frozen `.exe` can't use `sys.executable` as Python interpreter for venv creation |
| FIX-1 (grpc) | `grpc_client.py` | `ssl_target_name_override` disabled hostname verification |
| FIX-FFMPEG-404 | `install_ffmpeg.py` | Hardcoded FFmpeg URL returned 404; switched to stable redirect URL |
| FIX-FFMPEG-HASH | `install_ffmpeg.py` | Hash verification was completely skipped; now computed and stored on first install |
| FIX-A/B (federated_client) | `federated_client.py` | Adding `runtime/` to `sys.path` shadowed the real `grpcio` package |
| FIX-SS-1/2/3 | `centralized_secure_store.py` | Path traversal, name overflow, header bounds check |
| FIX-DP-1 | `dp_agent.py` | `epsilon_spent` was never computed (hardcoded 1.0 in pipeline) |
| FIX-QUEUE-1 | `offline_queue.py` | Offline queue entries stored as plaintext JSON |
| FIX-PIPELINE-1..6 | `pipeline.py` | Local file path sent to server; no hash; no streaming; global model download |
| FIX-RG1..7 | `runtime_guard.py` | Race window (sleep), no canary check, no no-root check, predictable lock |
| FIX-CAP-WIN1/2/3 | `capture.py` | DirectShow enumeration parser bug; cv2 fallback; WASAPI audio fallback |
| FIX-CAP-SIL | `capture.py` | Silence placeholder capped at 2s (too short for VAD) → raised to 10s |
| FIX-CAP-DUR | `capture.py` | No `has_real_media` flag in session metadata |
| FIX-LEDGER-1..6 | `ledger.rs` | Duplicate imports, unused imports, `nix` optional feature compile error, `windows_sys` not in Cargo.toml, TPM feature block calling non-existent function, incorrect test assertion |
| FIX-OTP-1 | `otp.rs` | OTP expiry was 6000s (100 min) instead of 600s (10 min) |
| FIX-AGG-1 | `server.rs` | `std::process::Command` blocked Tokio executor; switched to `tokio::process::Command` |
| FIX-AGG-2 | `server.rs` | Race condition: multiple tasks could trigger aggregation; fixed with `RoundState::Aggregating` check under lock |
| FIX-1..17 (FL comparison) | `fl_algorithm_comparison.py` | Architecture mismatch, shape filtering, BatchNorm crash, DP noise scale, text distribution mismatch, dead variables, FedYogi not instantiated, class weight double-application, hospital grouping, Adam optimizer, soft threshold, experiment deduplication |
| BYPASS-1..10 | `integrity.py` | Ten integrity bypass vectors, all patched |

---

## 26. References

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data**. AISTATS.

2. Bonawitz, K., Ivanov, V., Kreuter, B., et al. (2017). **Practical Secure Aggregation for Privacy-Preserving Machine Learning**. ACM CCS.

3. Abadi, M., Chu, A., Goodfellow, I., et al. (2016). **Deep Learning with Differential Privacy**. ACM CCS.

4. Mironov, I. (2017). **Rényi Differential Privacy**. IEEE CSF.

5. Li, T., et al. (2020). **Federated Optimization in Heterogeneous Networks (FedProx)**. MLSys.

6. Karimireddy, S. P., et al. (2020). **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**. ICML.

7. Reddi, S., et al. (2021). **Adaptive Federated Optimization (FedAdam/FedYogi)**. ICLR.

8. Wang, Y. X., et al. (2019). **Subsampled Rényi Differential Privacy and Analytical Moments Accountant**. AISTATS.

9. Zhu, L., et al. (2019). **Deep Leakage from Gradients**. NeurIPS.

10. Shokri, R., & Shmatikov, V. (2015). **Privacy-Preserving Deep Learning**. ACM CCS.

11. Nasr, M., Shokri, R., & Houmansadr, A. (2019). **Comprehensive Privacy Analysis of Deep Learning**. IEEE S&P.

12. Blanchard, P., et al. (2017). **Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum)**. NeurIPS.

13. Chillotti, I., et al. (2020). **TFHE: Fast Fully Homomorphic Encryption over the Torus**. Journal of Cryptology.

14. Gratch, J., et al. (2014). **The Distress Analysis Interview Corpus of Human and Computer Interviews (DAIC-WOZ)**. LREC.

15. Ji, S., et al. (2021). **MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare**. arXiv.

16. Microsoft SEAL (CKKS/BFV) — https://github.com/microsoft/SEAL

17. Opacus (PyTorch DP-SGD) — https://opacus.ai

18. Flower (Federated Learning) — https://flower.dev

19. tpm2-tools — https://github.com/tpm2-software/tpm2-tools

---

## Appendix A — Directory Permissions Reference

```
~/.federated/            chmod 0700  (owner read/write/exec only)
  bin/                   chmod 0700
    federated-client     chmod 0700  (execute)
    windows_signer.exe   chmod 0700
  agents/                chmod 0700  → frozen to 0444 (chattr +i on Linux) after install
  runtime/               chmod 0700  → frozen
  core/                  chmod 0700  → frozen
  installer/security/    chmod 0700  → frozen
  configs/               chmod 0700
  keys/
    ca.pem               chmod 0644  (world-readable for TLS verification)
    client.key           chmod 0600
    client.pem           chmod 0600
  data/secure_store/
    master.key           chmod 0600
  state/
    install_state.json   chmod 0600
    health.json          chmod 0600
    runtime.lock         chmod 0600
    receipt_hmac.key     chmod 0600
  logs/                  chmod 0700
  integrity/
    baseline.sha256      chmod 0444  → immutable
    baseline.sig         chmod 0444  → immutable
    install.complete     chmod 0444  → immutable
  tpm/
    device_pubkey.pem    chmod 0600
    sealed_secret.ctx    chmod 0600
  secrets/
    master.bin           chmod 0600  (Windows fallback only — use DPAPI in production)
```

---

## Appendix B — Health File

Written by `HealthReporter` to `~/.federated/state/health.json` (chmod 0600):

```json
{
  "status":   "healthy",
  "ts":       "2025-08-10T12:34:56Z",
  "started":  "2025-08-10T12:00:00Z",
  "pid":      12345,
  "platform": "Linux",
  "python":   "3.11.5",
  "metrics": {
    "rounds_attempted":  10,
    "rounds_succeeded":   9,
    "rounds_failed":      1,
    "avg_latency_s":     42.3,
    "success_rate":       0.9
  }
}
```

---

*Last updated: August 2025 | V.E.S.I.T Department of Computer Engineering | Group 42*