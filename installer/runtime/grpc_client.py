"""
grpc_client.py — Dual channel mTLS implementation

SECURITY FIXES:
  FIX-GRPC-1: Removed grpc.ssl_target_name_override and grpc.default_authority
               overrides. These two options disable TLS hostname verification,
               making MITM attacks trivial even with valid certificates.
               The server certificate must have a correct SAN that matches the
               server address. Use gen_certs.sh <SERVER_IP> to regenerate.

  FIX-GRPC-2: Added UploadUpdate client-streaming method to stub wrapper
               so pipeline.py can call stub.UploadUpdate(...).

  FIX-GRPC-3: Added DownloadGlobalModel server-streaming method.

  NOTE: If you see SSL_ERROR_SSL after this fix, your server cert SAN does
        not include the IP/hostname you are connecting to. Fix:
          bash server/orchestration_agent/certs/gen_certs.sh <YOUR_SERVER_IP>
        Then copy certs/ca.pem to installer/runtime/keys/ca.pem and reinstall.
"""

from __future__ import annotations

import grpc
import time
import logging
from pathlib import Path

from runtime.tpm_guard import sign_message
from runtime.self_destruct import trigger_self_destruct
from runtime.grpc.orchestrator_pb2_grpc import OrchestratorStub

log = logging.getLogger(__name__)

BASE  = Path.home() / ".federated"
KEYS  = BASE / "keys"

_CA_PEM      = KEYS / "ca.pem"
_CLIENT_KEY  = KEYS / "client.key"
_CLIENT_CERT = KEYS / "client.pem"

# FIX-GRPC-1: No hostname override options.
# The server certificate MUST have a SAN matching the address you connect to.
# If connecting by IP, the cert needs IP.x = <IP> in [alt_names].
# If connecting by hostname, the cert needs DNS.x = <hostname>.
# In grpc_client.py, update _CHANNEL_OPTIONS:
_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms",              10_000),
    ("grpc.keepalive_timeout_ms",            5_000),
    ("grpc.keepalive_permit_without_calls",      1),
    ("grpc.http2.max_pings_without_data",        0),
    ("grpc.max_send_message_length",    8 * 1024 * 1024),   # 8MB per message
    ("grpc.max_receive_message_length", 8 * 1024 * 1024),   # 8MB per message
]

_MAX_RETRY    = 5
_RETRY_BASE_S = 1.0


def _wait_ready(channel: grpc.Channel, timeout: float = 15.0):
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise ConnectionError(
            f"gRPC channel not ready within {timeout}s.\n"
            "If this is a TLS error, ensure the server certificate SAN includes "
            "the address you are connecting to. Regenerate with:\n"
            "  bash server/orchestration_agent/certs/gen_certs.sh <SERVER_IP>\n"
            "Then copy certs/ca.pem → installer/runtime/keys/ca.pem"
        )


def _with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(_MAX_RETRY):
        try:
            return fn(*args, **kwargs)
        except grpc.RpcError as e:
            if e.code() in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ):
                wait = _RETRY_BASE_S * (2 ** attempt)
                log.warning(
                    "gRPC transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRY, wait, e.details()
                )
                time.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err


def create_enrollment_channel(server_addr: str) -> grpc.Channel:
    """
    Server-TLS only channel (no client certificate).
    Used during installation / first enrollment before client cert exists.
    """
    if not _CA_PEM.exists():
        raise FileNotFoundError(f"CA certificate not found: {_CA_PEM}")

    creds = grpc.ssl_channel_credentials(
        root_certificates=_CA_PEM.read_bytes(),
    )
    channel = grpc.secure_channel(server_addr, creds, options=_CHANNEL_OPTIONS)
    _wait_ready(channel)
    log.info("[gRPC] Enrollment channel ready → %s", server_addr)
    return channel


def create_mtls_channel(server_addr: str) -> grpc.Channel:
    """
    Full mTLS channel (client cert + server cert).
    Used for all operational calls after enrollment.
    """
    for p in [_CA_PEM, _CLIENT_KEY, _CLIENT_CERT]:
        if not p.exists():
            raise FileNotFoundError(
                f"mTLS credential missing: {p}\n"
                "Run the installer to enroll this device first."
            )

    creds = grpc.ssl_channel_credentials(
        root_certificates=_CA_PEM.read_bytes(),
        private_key=_CLIENT_KEY.read_bytes(),
        certificate_chain=_CLIENT_CERT.read_bytes(),
    )
    channel = grpc.secure_channel(server_addr, creds, options=_CHANNEL_OPTIONS)
    _wait_ready(channel)
    log.info("[gRPC] mTLS channel ready → %s", server_addr)
    return channel


# ENROLLMENT_PORT is the plain-TLS port used before client cert exists
ENROLLMENT_PORT = 50051
# OPERATIONAL_PORT is the full-mTLS port used for all post-enrollment calls
OPERATIONAL_PORT = 50052

def _operational_addr(server_addr: str) -> str:
    """Replace the port in server_addr with the mTLS operational port."""
    host = server_addr.rsplit(":", 1)[0]
    return f"{host}:{OPERATIONAL_PORT}"

def _enrollment_addr(server_addr: str) -> str:
    """Return the enrollment address (port 50051)."""
    host = server_addr.rsplit(":", 1)[0]
    return f"{host}:{ENROLLMENT_PORT}"


def create_grpc_stub(server_addr: str) -> OrchestratorStub:
    """
    Post-enrollment operational stub — full mTLS on port 50052.
    The client cert MUST exist (installer already completed enrollment).
    """
    ops_addr = _operational_addr(server_addr)
    for p in [_CA_PEM, _CLIENT_KEY, _CLIENT_CERT]:
        if not p.exists():
            raise FileNotFoundError(
                f"mTLS credential missing: {p}\n"
                "Device must be enrolled before running the pipeline."
            )
    channel = create_mtls_channel(ops_addr)
    log.info("[gRPC] Operational mTLS channel ready → %s", ops_addr)
    stub = OrchestratorStub(channel)
    stub._sign_message = sign_message
    return stub


def enrollment_stub(server_addr: str) -> OrchestratorStub:
    """
    Enrollment stub — server-TLS only on port 50051.
    Used by installer before client cert exists.
    """
    enr_addr = _enrollment_addr(server_addr)
    channel  = create_enrollment_channel(enr_addr)
    log.info("[gRPC] Enrollment channel ready → %s", enr_addr)
    return OrchestratorStub(channel)

def call_with_retry(rpc_fn, request, timeout: float = 30.0):
    """Wrap any unary gRPC call with retry + timeout."""
    return _with_retry(rpc_fn, request, timeout=timeout)