"""
capture_v3.py — Fixed video/audio capture

FIXES OVER v2:
  FIX-CAP-WIN1  enumerate_dshow_devices() filtered ALL lines that had a
                quoted string when current_type was set, but the code used
                an `else:` that prevented parsing lines containing
                "video devices" / "audio devices" even if they also held a
                device name.  More critically, "Alternative name" lines were
                being added as real device names.  Fixed with explicit
                skip of "alternative name" lines.

  FIX-CAP-WIN2  When ffmpeg DirectShow enumeration returns nothing, fall
                back to cv2.VideoCapture(idx, cv2.CAP_DSHOW) to probe
                camera indices 0-3.  cv2 bypasses the ffmpeg enumeration
                step entirely and works on most Windows laptops.

  FIX-CAP-WIN3  Audio capture on Windows now tries wasapi (Windows Audio
                Session API) when DirectShow enumeration finds nothing.
                ffmpeg -f wasapi -i default works on Windows 7+ with any
                audio driver.

  FIX-CAP-SIL   Silence placeholder was capped at 2 s regardless of the
                requested duration.  The pipeline needs enough audio for
                VAD and segment extraction.  Raised cap to 10 s so that
                session_processor can at least produce one segment row even
                when the microphone is unavailable.

  FIX-CAP-DUR   capture_session() now returns a flag in session_meta.json
                indicating whether any real media was captured so that the
                caller can decide to skip the FL round.
"""

import logging
import os
import platform
import re
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system().lower() == "windows"
IS_LINUX   = platform.system().lower() == "linux"
IS_MACOS   = platform.system().lower() == "darwin"

BASE     = Path.home() / ".federated"
DATA_DIR = BASE / "data" / "input"


# ═══════════════════════════════════════════════════════════════════
# DEVICE ENUMERATION
# ═══════════════════════════════════════════════════════════════════

def _popen_kw() -> dict:
    kw: dict = {}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def enumerate_dshow_devices() -> Tuple[List[str], List[str]]:
    """
    FIX-CAP-WIN1: Enumerate DirectShow devices on Windows.

    Key changes vs previous version:
    - Skips lines containing "alternative name" (those are device GUIDs,
      not human-readable names, and must NOT be used as -i arguments).
    - Uses `text=False` + manual decode so locale-specific characters in
      device names don't raise UnicodeDecodeError.
    - Sets current_type before the else-branch so the first quoted name
      on the same line as "video devices" / "audio devices" is captured
      correctly (rare but possible in some ffmpeg builds).
    """
    if not IS_WINDOWS or not _ffmpeg_available():
        return [], []

    try:
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,   # do NOT use text=True (encoding issues)
            timeout=10,
            **_popen_kw(),
        )
        # ffmpeg writes the device list to stderr
        output = result.stderr.decode("utf-8", errors="replace")

        video_devs: List[str] = []
        audio_devs: List[str] = []
        current_type: Optional[str] = None

        for line in output.splitlines():
            line_lower = line.lower()

            # Skip alternative-name / GUID lines — they are NOT device names
            if "alternative name" in line_lower:
                continue

            # Detect section headers FIRST, then check for device names
            if "video devices" in line_lower:
                current_type = "video"
            elif "audio devices" in line_lower:
                current_type = "audio"

            # Attempt to parse a quoted device name on this line
            if current_type:
                m = re.search(r'"([^"]+)"', line)
                if m:
                    name = m.group(1).strip()
                    # Reject section-header strings accidentally captured
                    if name and "devices" not in name.lower():
                        if current_type == "video" and name not in video_devs:
                            video_devs.append(name)
                        elif current_type == "audio" and name not in audio_devs:
                            audio_devs.append(name)

        log.info(
            "[capture] DirectShow devices — video: %s, audio: %s",
            video_devs, audio_devs,
        )
        return video_devs, audio_devs

    except Exception as e:
        log.warning("[capture] DirectShow enumeration failed: %s", e)
        return [], []


def _probe_cv2_cameras() -> List[int]:
    """
    FIX-CAP-WIN2: Probe camera indices 0-3 using cv2 with the DirectShow
    backend.  Returns a list of working integer indices.
    cv2 does its own DirectShow negotiation independently of ffmpeg and
    succeeds on most Windows laptops even when ffmpeg enumeration fails.
    """
    try:
        import cv2
        working = []
        for idx in range(4):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    working.append(idx)
            cap.release()
        return working
    except Exception as e:
        log.debug("[capture] cv2 camera probe failed: %s", e)
        return []


def enumerate_v4l2_devices() -> List[str]:
    """Enumerate V4L2 video devices on Linux (unchanged from v2)."""
    if not IS_LINUX:
        return []

    devices = []
    for path in sorted(Path("/dev").glob("video*")):
        if os.access(str(path), os.R_OK | os.W_OK):
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "--device", str(path), "--list-formats"],
                    capture_output=True, text=True, timeout=3,
                )
                caps = result.stdout.lower()
                if "yuyv" in caps or "mjpg" in caps or "h264" in caps or "rgb" in caps:
                    devices.append(str(path))
                elif not devices:
                    devices.append(str(path))
            except FileNotFoundError:
                devices.append(str(path))
            except Exception:
                pass
        else:
            log.warning(
                "[capture] No permission for %s. Add user to 'video' group:\n"
                "  sudo usermod -aG video $USER && newgrp video",
                path,
            )

    log.info("[capture] V4L2 devices available: %s", devices)
    return devices


def check_audio_devices_linux() -> List[str]:
    """Find working audio capture device on Linux (unchanged from v2)."""
    candidates = []

    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "input" in line.lower() or "capture" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        candidates.append(f"pulse:{parts[1]}")
    except Exception:
        pass

    candidates.extend(["default", "pulse", "hw:0,0", "plughw:0,0"])

    working = []
    for dev in candidates:
        try:
            alsa_fmt = dev.replace("pulse:", "") if dev.startswith("pulse:") else dev
            f = "pulse" if dev.startswith("pulse:") else "alsa"
            result = subprocess.run(
                ["ffmpeg", "-f", f, "-i", alsa_fmt, "-t", "0.5",
                 "-vn", "-hide_banner", "-loglevel", "error", "-f", "null", "-"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                working.append(dev)
                break
        except Exception:
            continue

    if not working:
        log.warning("[capture] No working audio capture device found")
    else:
        log.info("[capture] Audio device: %s", working[0])

    return working


# ═══════════════════════════════════════════════════════════════════
# CAPTURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _capture_video_windows_cv2(out_path: Path, duration_s: int) -> bool:
    """
    FIX-CAP-WIN2: Capture video on Windows using cv2 + DirectShow.
    Used as fallback when ffmpeg DirectShow enumeration returns nothing.
    Audio is NOT captured this way (handled separately by _capture_audio_windows).
    """
    try:
        import cv2

        working_indices = _probe_cv2_cameras()
        if not working_indices:
            log.warning("[capture] cv2: no working camera index found (0-3 all failed)")
            return False

        idx = working_indices[0]
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log.warning("[capture] cv2: could not open camera index %d", idx)
            return False

        fps  = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = 15.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        log.info(
            "[capture] cv2: recording camera idx=%d @ %.0f fps %dx%d → %s",
            idx, fps, w, h, out_path,
        )

        start = time.time()
        frames_written = 0
        while time.time() - start < duration_s:
            ret, frame = cap.read()
            if not ret:
                log.warning("[capture] cv2: cap.read() returned False — camera dropped")
                break
            writer.write(frame)
            frames_written += 1

        cap.release()
        writer.release()

        if out_path.exists() and out_path.stat().st_size > 1024 and frames_written > 0:
            log.info(
                "[capture] cv2: %d frames captured (%.1f s)",
                frames_written, frames_written / fps,
            )
            return True

        log.warning("[capture] cv2: output file too small or no frames written")
        return False

    except ImportError:
        log.warning("[capture] cv2 not installed — cannot use as video fallback")
        return False
    except Exception as e:
        log.warning("[capture] cv2 capture error: %s", e)
        return False


def _capture_video_windows(out_path: Path, duration_s: int) -> bool:
    """
    FIX-CAP-WIN1+WIN2: Windows video capture.
    Tries ffmpeg DirectShow first (named devices), then cv2 fallback.
    """
    video_devs, audio_devs = enumerate_dshow_devices()

    # ── ffmpeg DirectShow (named device) ─────────────────────────────────────
    if video_devs:
        video_dev = video_devs[0]
        log.info("[capture] Using DirectShow video device: %s", video_dev)

        if audio_devs:
            audio_dev = audio_devs[0]
            cmd = [
                "ffmpeg", "-y",
                "-f", "dshow",
                "-i", f"video={video_dev}:audio={audio_dev}",
                "-t", str(duration_s),
                "-vcodec", "libx264", "-preset", "ultrafast",
                "-acodec", "aac",
                "-hide_banner", "-loglevel", "warning",
                str(out_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "dshow",
                "-i", f"video={video_dev}",
                "-t", str(duration_s),
                "-vcodec", "libx264", "-preset", "ultrafast",
                "-an",
                "-hide_banner", "-loglevel", "warning",
                str(out_path),
            ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=duration_s + 30, **_popen_kw()
            )
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
                return True
            # Log stderr at WARNING so it is visible in INFO log level
            log.warning(
                "[capture] ffmpeg DirectShow failed:\n%s",
                proc.stderr.decode(errors="replace")[-600:],
            )
        except subprocess.TimeoutExpired:
            log.warning("[capture] ffmpeg video timed out after %ds", duration_s + 30)
        except Exception as e:
            log.warning("[capture] ffmpeg video error: %s", e)

    # ── cv2 fallback ──────────────────────────────────────────────────────────
    log.info("[capture] Falling back to cv2 for video capture")
    return _capture_video_windows_cv2(out_path, duration_s)


def _capture_audio_windows(out_path: Path, duration_s: int) -> bool:
    """
    FIX-CAP-WIN3: Windows audio capture.
    Tries DirectShow named devices first, then wasapi (Windows Audio
    Session API) as a driver-agnostic fallback, then dshow with no
    explicit device name (lets ffmpeg pick the default).
    """
    # ── 1. Named DirectShow device ────────────────────────────────────────────
    _, audio_devs = enumerate_dshow_devices()
    if audio_devs:
        audio_dev = audio_devs[0]
        log.info("[capture] Using DirectShow audio device: %s", audio_dev)
        cmd = [
            "ffmpeg", "-y",
            "-f", "dshow", "-i", f"audio={audio_dev}",
            "-t", str(duration_s),
            "-ac", "1", "-ar", "16000",
            "-hide_banner", "-loglevel", "warning",
            str(out_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=duration_s + 15, **_popen_kw()
            )
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 512:
                return True
            log.warning(
                "[capture] DirectShow audio failed:\n%s",
                proc.stderr.decode(errors="replace")[-400:],
            )
        except Exception as e:
            log.warning("[capture] DirectShow audio error: %s", e)

    # ── 2. WASAPI (Windows Audio Session API) ─────────────────────────────────
    log.info("[capture] Trying wasapi audio capture")
    cmd = [
        "ffmpeg", "-y",
        "-f", "wasapi", "-i", "default",
        "-t", str(duration_s),
        "-ac", "1", "-ar", "16000",
        "-hide_banner", "-loglevel", "warning",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=duration_s + 15, **_popen_kw()
        )
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 512:
            log.info("[capture] wasapi audio capture succeeded")
            return True
        log.warning(
            "[capture] wasapi audio failed:\n%s",
            proc.stderr.decode(errors="replace")[-400:],
        )
    except Exception as e:
        log.warning("[capture] wasapi error: %s", e)

    # ── 3. dshow with no device name (let ffmpeg pick) ────────────────────────
    log.info("[capture] Trying dshow audio with no explicit device name")
    # Some ffmpeg builds accept an empty device name and pick the default
    for device_str in ["audio=", "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}"]:
        cmd = [
            "ffmpeg", "-y",
            "-f", "dshow", "-i", device_str,
            "-t", str(duration_s),
            "-ac", "1", "-ar", "16000",
            "-hide_banner", "-loglevel", "warning",
            str(out_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=duration_s + 15, **_popen_kw()
            )
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 512:
                log.info("[capture] dshow fallback audio succeeded with: %s", device_str)
                return True
        except Exception:
            continue

    log.warning("[capture] All Windows audio capture methods failed")
    return False


def _capture_video_linux(out_path: Path, duration_s: int) -> bool:
    """Linux video capture with V4L2 device auto-detection (unchanged from v2)."""
    devices = enumerate_v4l2_devices()
    if not devices:
        log.warning("[capture] No V4L2 video device found")
        return False

    for device in devices:
        log.info("[capture] Trying V4L2 device: %s", device)
        cmd = [
            "ffmpeg", "-y",
            "-f", "v4l2", "-i", device,
            "-t", str(duration_s),
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-an",
            "-hide_banner", "-loglevel", "warning",
            str(out_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=duration_s + 30)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
                log.info(
                    "[capture] Video captured from %s: %d bytes",
                    device, out_path.stat().st_size,
                )
                return True
            log.warning(
                "[capture] Device %s failed:\n%s",
                device, proc.stderr.decode(errors="replace")[-300:],
            )
        except subprocess.TimeoutExpired:
            log.warning("[capture] Timeout on device %s", device)
        except Exception as e:
            log.warning("[capture] Error on device %s: %s", device, e)

    return False


def _capture_audio_linux(out_path: Path, duration_s: int) -> bool:
    """Linux audio capture with device auto-detection (unchanged from v2)."""
    working_devs = check_audio_devices_linux()
    if not working_devs:
        return False

    dev = working_devs[0]
    if dev.startswith("pulse:"):
        fmt  = "pulse"
        idev = dev[len("pulse:"):]
    else:
        fmt  = "alsa"
        idev = dev

    cmd = [
        "ffmpeg", "-y",
        "-f", fmt, "-i", idev,
        "-t", str(duration_s),
        "-ac", "1", "-ar", "16000",
        "-hide_banner", "-loglevel", "warning",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=duration_s + 15)
        if proc.returncode != 0:
            log.warning(
                "[capture] Audio capture failed:\n%s",
                proc.stderr.decode(errors="replace")[-300:],
            )
            return False
        return out_path.exists() and out_path.stat().st_size > 512
    except Exception as e:
        log.warning("[capture] Audio error: %s", e)
        return False


def _extract_audio_from_video(video_path: Path, out_path: Path, sr: int = 16000) -> bool:
    """Extract audio track from a captured video file (unchanged)."""
    if not video_path.exists():
        log.warning("[capture] Cannot extract audio: video not found at %s", video_path)
        return False

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sr),
        "-vn", "-hide_banner", "-loglevel", "warning",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, **_popen_kw())
        if proc.returncode != 0:
            log.warning(
                "[capture] Audio extraction failed:\n%s",
                proc.stderr.decode(errors="replace")[-300:],
            )
            return False
        return out_path.exists() and out_path.stat().st_size > 512
    except Exception as e:
        log.warning("[capture] Audio extraction error: %s", e)
        return False


def _write_silence_wav(out_path: Path, duration_s: int, sr: int = 16000):
    """
    FIX-CAP-SIL: Write a silence WAV of up to 10 s (was 2 s).
    10 s of silence gives the VAD and session_processor enough signal to
    produce at least one segment row, preventing the empty-manifest crash
    downstream.  The pipeline will recognise that no real speech was
    detected and will produce an empty transcript row, which is handled
    gracefully.
    """
    actual_s    = min(duration_s, 10)   # cap at 10 s (was 2 s)
    num_samples = sr * actual_s
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    log.info("[capture] Wrote silence placeholder: %s (%ds)", out_path, actual_s)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def capture_session(
    duration_s: int = 300,
    force_audio_only: bool = False,
) -> Path:
    """
    Capture audio (and optionally video) for duration_s seconds.

    Returns path to session directory containing:
      session_TIMESTAMP/
        audio.wav          (always present — silence if capture fails)
        video.mp4          (present if camera available)
        session_meta.json  (includes has_real_media flag)

    FIX-CAP-DUR: session_meta.json now includes has_real_media=True/False.
    The pipeline uses this flag to decide whether to skip the FL round
    rather than crashing when no usable data was captured.
    """
    if not _ffmpeg_available():
        log.error(
            "[capture] ffmpeg not on PATH — video/audio capture disabled. "
            "Install from https://ffmpeg.org/download.html"
        )

    ts          = int(time.time())
    session_dir = DATA_DIR / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    audio_path = session_dir / "audio.wav"
    video_path = session_dir / "video.mp4"

    log.info("[capture] Starting %ds capture → %s", duration_s, session_dir)

    has_video     = False
    has_real_audio = False   # True = real mic, False = silence placeholder

    # ── Video ─────────────────────────────────────────────────────────────────
    if not force_audio_only and _ffmpeg_available():
        if IS_WINDOWS:
            has_video = _capture_video_windows(video_path, duration_s)
        elif IS_LINUX:
            has_video = _capture_video_linux(video_path, duration_s)
        elif IS_MACOS:
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation", "-framerate", "30",
                "-i", "0:0",
                "-t", str(duration_s),
                "-vcodec", "libx264", "-preset", "ultrafast",
                "-hide_banner", "-loglevel", "warning",
                str(video_path),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=duration_s + 30)
                has_video = proc.returncode == 0 and video_path.exists()
            except Exception as e:
                log.warning("[capture] macOS capture error: %s", e)

    # ── Extract audio from video ───────────────────────────────────────────────
    if has_video:
        log.info(
            "[capture] Video captured: %s (%.1f MB)",
            video_path.name, video_path.stat().st_size / 1e6,
        )
        extracted = _extract_audio_from_video(video_path, audio_path)
        if extracted:
            has_real_audio = True
        else:
            log.warning("[capture] Audio extraction failed — trying direct mic capture")
            has_video = False   # still attempt separate audio capture

    # ── Audio-only capture (or fallback after video+audio failed) ─────────────
    if not audio_path.exists():
        log.info("[capture] Attempting audio-only capture")
        if IS_WINDOWS:
            has_real_audio = _capture_audio_windows(audio_path, duration_s)
        elif IS_LINUX:
            has_real_audio = _capture_audio_linux(audio_path, duration_s)
        # macOS audio is captured together with video above; if that failed, fall through

        if not has_real_audio:
            log.warning(
                "[capture] All audio capture methods failed — writing silence placeholder"
            )
            _write_silence_wav(audio_path, duration_s)

    log.info(
        "[capture] Session ready: %s | video=%s audio=%s real_audio=%s",
        session_dir.name, has_video, audio_path.exists(), has_real_audio,
    )

    # ── Write session metadata ────────────────────────────────────────────────
    import json
    has_real_media = has_video or has_real_audio
    meta = {
        "timestamp":      ts,
        "duration_s":     duration_s,
        "has_video":      has_video,
        "has_audio":      audio_path.exists() and audio_path.stat().st_size > 512,
        "has_real_audio": has_real_audio,
        "has_real_media": has_real_media,   # FIX-CAP-DUR: used by pipeline
        "video_bytes":    video_path.stat().st_size if has_video else 0,
        "audio_bytes":    audio_path.stat().st_size if audio_path.exists() else 0,
    }
    (session_dir / "session_meta.json").write_text(json.dumps(meta, indent=2))

    return session_dir


def scan_existing_sessions() -> List[Path]:
    """Return captured session directories, newest first."""
    sessions = sorted(
        [d for d in DATA_DIR.glob("session_*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    log.info("[capture] Found %d existing session(s)", len(sessions))
    return sessions


def get_or_capture_session(duration_s: int = 60) -> Path:
    """
    In run-once mode: reuse the most recent session if <24 h old,
    otherwise capture a fresh one.
    """
    sessions = scan_existing_sessions()
    if sessions:
        newest  = sessions[0]
        age_h   = (time.time() - newest.stat().st_mtime) / 3600
        if age_h < 24:
            log.info(
                "[capture] Reusing session from %.1fh ago: %s",
                age_h, newest.name,
            )
            return newest

    log.info("[capture] No recent session found — capturing %ds", duration_s)
    return capture_session(duration_s)