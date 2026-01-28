#!/usr/bin/env python3
"""
client_agent.py

Autonomous UNTRUSTED client that:
  - connects to orchestrator (insecure gRPC)
  - runs full local pipeline:
        LDA -> Trainer -> DP -> Encryption
  - submits encrypted update metadata + receipt pointer
"""

import uuid
import hashlib
from pathlib import Path

import grpc

# -----------------------------
# gRPC imports (generated)
# -----------------------------
from client_agent.grpc import orchestrator_pb2 as pb
from client_agent.grpc.orchestrator_pb2_grpc import OrchestratorStub

# -----------------------------
# Local agents
# -----------------------------
from LDA.app.main import preprocess, PreprocessRequest
from trainer_agent.trainer_mentalbert_privacy import orchestrate as trainer_orchestrate
from dp_agent.dp_agent import DPAgent
from enc_agent.enc_agent import EncryptionAgent
from centralized_secure_store import SecureStore

# -----------------------------
# Config
# -----------------------------
ORCHESTRATOR_ADDR = "127.0.0.1:50051"
CLIENT_ID_PATH = Path("./client_id.bin")
SECURE_STORE_ROOT = Path("./secure_store")

INPUT_PATH = "./client_data"
LDA_MODE = "session"
CONFIG_URI = "file://configs/local_config.yaml"
VIDEO_DIR = "./videos"

# -----------------------------
# Helpers
# -----------------------------
def load_or_create_device_id() -> bytes:
    if CLIENT_ID_PATH.exists():
        return CLIENT_ID_PATH.read_bytes()
    did = uuid.uuid4().bytes
    CLIENT_ID_PATH.write_bytes(did)
    return did


# -----------------------------
# Main client flow
# -----------------------------
def run_client_once():
    # ---- identity ----
    device_id = load_or_create_device_id()

    # ---- INSECURE gRPC channel (MATCHES RUST SERVER) ----
    channel = grpc.insecure_channel(ORCHESTRATOR_ADDR)
    stub = OrchestratorStub(channel)

    # ---- query round ----
    round_meta = stub.GetRound(pb.DeviceId(id=device_id))
    print(f"[client] round={round_meta.round_id} state={round_meta.state}")

    if round_meta.state != "Collecting":
        print("[client] round not collecting, exiting")
        return

    session_id = f"client-{uuid.uuid4().hex}"

    # =====================================================
    # 1) LDA
    # =====================================================
    print("[client] running LDA")
    lda_req = PreprocessRequest(
        mode=LDA_MODE,
        inputs={"video_dir": VIDEO_DIR},
        config_uri=CONFIG_URI,
        session_id=session_id,
    )
    lda_result = preprocess(lda_req)
    print(lda_result)

    # =====================================================
    # 2) Trainer
    # =====================================================
    print("[client] running trainer")
    trainer_out = trainer_orchestrate(
        input_path=lda_result["artifact_manifest"],
        session_id=session_id,
        mode="supervised",
        epochs=1,
        batch_size=8,
    )

    local_update_uri = trainer_out["local_update_uri"]

    # =====================================================
    # 3) Differential Privacy
    # =====================================================
    print("[client] running DP")
    store = SecureStore(agent="client", root=SECURE_STORE_ROOT)

    dp_agent = DPAgent(
        clip_norm=1.0,
        noise_multiplier=1.0,
        mechanism="gaussian",
        store=store,
    )

    dp_result = dp_agent.process_local_update(
        local_update_uri,
        metadata={"session_id": session_id},
    )

    dp_update_uri = dp_result["update_uri"]

    # =====================================================
    # 4) Encryption
    # =====================================================
    print("[client] running encryption")
    enc_agent = EncryptionAgent(mode="aes")

    enc_result = enc_agent.process_dp_update(dp_result["receipt_uri"])
    final_update_uri = enc_result["receipt"]["outputs"][0]

    # =====================================================
    # 5) Submit receipt metadata to server
    # =====================================================
    print("[client] submitting update")

    payload_hash = hashlib.sha256(
        final_update_uri.encode("utf-8")
    ).digest()

    receipt = pb.Receipt(
        device_id=device_id,
        round_id=round_meta.round_id,
        payload_hash=payload_hash,
        epsilon_spent=1.0,
        signature=b"",          # server validates via receipt chain
        enc_uri=final_update_uri,
        scheme="AES-GCM",
        nonce="",
    )

    ack = stub.SubmitReceipt(receipt)
    print("[client] server ack:", ack.ok)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_client_once()
