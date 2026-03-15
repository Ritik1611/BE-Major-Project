import subprocess
import time
import platform

IDLE_THRESHOLD_SECONDS = 300  # 5 minutes


def is_system_idle() -> bool:
    system = platform.system().lower()

    # ✅ Windows: no idle blocking
    if system == "windows":
        return True

    # ✅ Linux: xprintidle
    try:
        output = subprocess.check_output(
            ["xprintidle"], stderr=subprocess.DEVNULL
        )
        idle_ms = int(output.strip())
        return idle_ms > IDLE_THRESHOLD_SECONDS * 1000
    except Exception:
        # If idle detection fails, assume idle
        # prevents infinite wait on headless machines
        print("[WARN] Idle detection unavailable, assuming idle")
        return True


def wait_until_idle(max_wait_seconds=600):
    """
    Wait until system becomes idle.
    Fallback: continue after max_wait_seconds to prevent deadlock.
    """

    start = time.time()

    while not is_system_idle():

        # Safety timeout
        if time.time() - start > max_wait_seconds:
            print("[WARN] Idle wait timeout reached, continuing pipeline")
            return

        time.sleep(30)
