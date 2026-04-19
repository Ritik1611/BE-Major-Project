"""
daemon.py — Continuous background federated learning daemon

Flow:
  1. Wait for system idle
  2. Capture audio/video for CAPTURE_WINDOW_S seconds
  3. Run full pipeline (LDA → Trainer → DP → Enc → Upload)
  4. Sleep UPLOAD_INTERVAL_S
  5. Repeat

The daemon writes a heartbeat file so the integrity watcher knows it's alive.
"""

import os
import time
import signal
import logging
import platform
import threading
from pathlib import Path
from typing import Optional

from runtime.idle import wait_until_idle
from runtime.pipeline import run_pipeline
from runtime.capture import capture_session

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system().lower() == "windows"

BASE = Path.home() / ".federated"
HEARTBEAT_FILE = BASE / "state" / "daemon.heartbeat"
LOCK_FILE      = BASE / "state" / "runtime.lock"

CONTINUOUS_WINDOW_S   = 300   # 5-minute analysis windows
CONTINUOUS_PAUSE_S    = 5     # brief pause between windows (overlap buffer flush)
FL_UPLOAD_INTERVAL_S  = 3600  # FL batch submissions still happen every hour
HEARTBEAT_INTERVAL = 30          # write heartbeat every 30s


# ── Heartbeat writer ──────────────────────────────────────────────────────────
class _HeartbeatThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="heartbeat")
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
                HEARTBEAT_FILE.write_text(str(time.time()))
            except Exception:
                pass
            self._stop.wait(HEARTBEAT_INTERVAL)

    def stop(self):
        self._stop.set()


# ── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown_event = threading.Event()

def _signal_handler(sig, frame):
    log.info("Daemon received signal %s — shutting down gracefully", sig)
    _shutdown_event.set()


def _register_signals():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)


# ── Lock helpers ──────────────────────────────────────────────────────────────
def _acquire_lock() -> bool:
    try:
        if LOCK_FILE.exists():
            # Check if the PID in the lock file is still running
            try:
                pid = int(LOCK_FILE.read_text().strip())
                if IS_WINDOWS:
                    import ctypes
                    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                    if handle:
                        ctypes.windll.kernel32.CloseHandle(handle)
                        log.warning("Another daemon instance (PID %d) is running", pid)
                        return False
                else:
                    os.kill(pid, 0)  # raises OSError if process is dead
                    log.warning("Another daemon instance (PID %d) is running", pid)
                    return False
            except (ValueError, OSError, ProcessLookupError):
                log.info("Stale lock file found, removing")
                LOCK_FILE.unlink(missing_ok=True)

        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCK_FILE.write_text(str(os.getpid()))
        return True
    except Exception as e:
        log.error("Failed to acquire daemon lock: %s", e)
        return False


def _release_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Main daemon loop ──────────────────────────────────────────────────────────
def daemon_loop(stub, device_id: bytes, master_secret: bytes, mode: str = "session"):
    """
    mode="session"    → scheduled FL upload daemon (5 min capture, 1-hour sleep)
    mode="continuous" → continuous monitoring (5 min windows, 5-second pause, no long sleep)
    """
    _register_signals()

    if not _acquire_lock():
        log.error("Daemon already running — exiting")
        return

    hb = _HeartbeatThread()
    hb.start()

    log.info("Federated daemon started (PID=%d mode=%s)", os.getpid(), mode)

    try:
        if mode == "continuous":
            _continuous_monitoring_loop(stub, device_id, master_secret)
        else:
            _scheduled_upload_loop(stub, device_id, master_secret)
    finally:
        hb.stop()
        _release_lock()
        log.info("Daemon stopped cleanly")


def _continuous_monitoring_loop(stub, device_id: bytes, master_secret: bytes):
    """
    Records in back-to-back windows. Each window is processed for inference.
    No long sleep between windows — this is the TRUE continuous monitoring path.
    A separate FL batch job would handle submitting training updates (batch/session modes).
    """
    log.info(
        "[continuous] Starting continuous monitoring "
        "(window=%ds, pause=%ds between windows)",
        CONTINUOUS_WINDOW_S, CONTINUOUS_PAUSE_S,
    )
    while not _shutdown_event.is_set():
        session_dir: Optional[Path] = None
        try:
            log.info("[continuous] Starting %ds capture window", CONTINUOUS_WINDOW_S)
            session_dir = capture_session(duration_s=CONTINUOUS_WINDOW_S)
            log.info("[continuous] Window captured → %s", session_dir)
        except Exception as e:
            log.error("[continuous] Capture failed: %s — retrying after %ds",
                      e, CONTINUOUS_PAUSE_S)
            _shutdown_event.wait(timeout=CONTINUOUS_PAUSE_S)
            continue

        try:
            # Inference only — NO FL update submitted
            run_pipeline(
                stub, device_id, master_secret,
                session_dir=session_dir,
                pipeline_mode="continuous",   # ← inference-only path
            )
        except Exception as e:
            log.error("[continuous] Inference pipeline failed: %s", e)

        # Brief pause to allow disk flush and prevent CPU spinning
        _shutdown_event.wait(timeout=CONTINUOUS_PAUSE_S)


def _scheduled_upload_loop(stub, device_id: bytes, master_secret: bytes):
    """
    Original FL update daemon: captures when idle, submits FL update, sleeps 1 hour.
    Used for batch/session modes where labeled data is processed.
    """
    log.info(
        "[scheduled] FL upload daemon "
        "(capture=%ds, sleep=%ds)",
        CAPTURE_WINDOW_S, FL_UPLOAD_INTERVAL_S,
    )
    while not _shutdown_event.is_set():
        log.info("[scheduled] Waiting for system idle...")
        wait_until_idle(max_wait_seconds=FL_UPLOAD_INTERVAL_S)

        if _shutdown_event.is_set():
            break

        session_dir: Optional[Path] = None
        try:
            log.info("[scheduled] Starting %ds capture", CAPTURE_WINDOW_S)
            session_dir = capture_session(duration_s=CAPTURE_WINDOW_S)
        except Exception as e:
            log.error("[scheduled] Capture failed: %s — skipping cycle", e)
            _shutdown_event.wait(timeout=60)
            continue

        try:
            run_pipeline(
                stub, device_id, master_secret,
                session_dir=session_dir,
                pipeline_mode="session",   # ← FL update path
            )
            log.info("[scheduled] Pipeline complete ✓")
        except Exception as e:
            log.error("[scheduled] Pipeline failed: %s", e)

        log.info("[scheduled] Sleeping %ds until next cycle", FL_UPLOAD_INTERVAL_S)
        _shutdown_event.wait(timeout=FL_UPLOAD_INTERVAL_S)