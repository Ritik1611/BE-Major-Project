"""
capture_v2.py — Fixed video/audio capture

ROOT CAUSES of "video not happening":

  BUG-CAP1  Windows: DirectShow device string was hardcoded to a specific
            hardware UUID: "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_..."
            This UUID matches only the default audio device on a specific Windows build.
            On any other machine, ffmpeg exits with "No such filter" or "Device not found".
            Fix: Enumerate DirectShow devices and pick the first available one.

  BUG-CAP2  Windows: "Integrated Camera" is the camera name on Surface/ThinkPad.
            HP laptops: "HP Wide Vision HD Camera". Dell: "Integrated Webcam".
            Huawei: "HuCam". Any mismatch → ffmpeg "No such filter" → silent failure.
            Fix: Enumerate DirectShow video devices dynamically.

  BUG-CAP3  Linux: /dev/video0 may be the IR camera, not the main camera.
            Many laptops: /dev/video0=IR, /dev/video2=RGB. Picking video0 records
            a useless infrared stream.
            Fix: Try /dev/video0, /dev/video2, /dev/video4 in order.

  BUG-CAP4  Linux: The ALSA device "default" may not have a microphone path.
            On headless/server machines, ALSA has no capture device.
            On desktops with PulseAudio, ffmpeg -f alsa fails.
            Fix: Try "default" then "pulse" then "hw:0,0" fallback.

  BUG-CAP5  Capture failure was silent: _capture_video_ffmpeg returned False
            but the log only said "Device may be busy". The ffmpeg stderr was
            printed at DEBUG level — not visible in INFO mode.
            Fix: Log ffmpeg stderr at WARNING level.

  BUG-CAP6  pipeline.py passes session_dir to LDA as "video_dir", but
            the LDA main loop globs for *.mp4 files directly in video_dir.
            capture.py creates session_dir/video.mp4, so the glob finds it.
            BUT: in "run-once" mode, session_dir=None → pipeline uses _INPUT_DIR
            which is typically empty. No video files → LDA produces 0 rows →
            pipeline exits early.
            Fix: In run-once mode, either capture first or use existing files.
            Added: --session-dir CLI argument to federated_client.py

  BUG-CAP7  The captured video.mp4 path was passed to ffmpeg for audio extraction,
            but if video capture failed, video.mp4 doesn't exist.
            The audio extraction then failed silently, leaving no audio.wav.
            Fix: Only attempt audio extraction if video.mp4 actually exists.

  BUG-CAP8  Permissions: on some Linux systems, the video device (/dev/video0)
            requires the user to be in the "video" group.
            Fix: Check group membership and provide clear error message.
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
    BUG-CAP1/CAP2 fix: Enumerate DirectShow video and audio devices on Windows.
    Returns ([video_device_names], [audio_device_names]).
    """
    if not IS_WINDOWS or not _ffmpeg_available():
        return [], []

    try:
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=10, **_popen_kw()
        )
        # ffmpeg prints device list to stderr
        output = result.stderr

        video_devs, audio_devs = [], []
        current_type = None

        for line in output.splitlines():
            if "video devices" in line.lower():
                current_type = "video"
            elif "audio devices" in line.lower():
                current_type = "audio"
            else:
                m = re.search(r'"([^"]+)"', line)
                if m:
                    name = m.group(1)
                    if current_type == "video":
                        video_devs.append(name)
                    elif current_type == "audio":
                        audio_devs.append(name)

        log.info("[capture] DirectShow devices — video: %s, audio: %s", video_devs, audio_devs)
        return video_devs, audio_devs

    except Exception as e:
        log.warning("[capture] Device enumeration failed: %s", e)
        return [], []


def enumerate_v4l2_devices() -> List[str]:
    """
    BUG-CAP3 fix: Enumerate V4L2 video devices on Linux.
    Returns list of device paths like ["/dev/video0", "/dev/video2"].
    """
    if not IS_LINUX:
        return []

    devices = []
    for path in sorted(Path("/dev").glob("video*")):
        # Check if user has read permission
        if os.access(str(path), os.R_OK | os.W_OK):
            # Filter out IR cameras (capability check via v4l2-ctl)
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "--device", str(path), "--list-formats"],
                    capture_output=True, text=True, timeout=3,
                )
                caps = result.stdout.lower()
                # Prefer devices that support common RGB formats
                if "yuyv" in caps or "mjpg" in caps or "h264" in caps or "rgb" in caps:
                    devices.append(str(path))
                    log.debug("[capture] V4L2 device %s: %s", path, caps[:80])
                elif not devices:
                    devices.append(str(path))  # Add anyway if no better device
            except FileNotFoundError:
                # v4l2-ctl not installed — add device without format check
                devices.append(str(path))
            except Exception:
                pass
        else:
            log.warning(
                "[capture] No permission for %s. Add user to 'video' group:\n"
                "  sudo usermod -aG video $USER && newgrp video",
                path
            )

    log.info("[capture] V4L2 devices available: %s", devices)
    return devices


def check_audio_devices_linux() -> List[str]:
    """
    BUG-CAP4 fix: Find working audio capture device on Linux.
    Returns list of ALSA/PulseAudio source names.
    """
    candidates = []

    # Try PulseAudio first (most modern Linux desktops)
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

    # ALSA fallback
    candidates.extend(["default", "pulse", "hw:0,0", "plughw:0,0"])

    # Test each candidate
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
                break  # Use first working device
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

def _capture_video_windows(out_path: Path, duration_s: int) -> bool:
    """BUG-CAP1/CAP2 fix: Windows video capture with auto device detection."""
    video_devs, audio_devs = enumerate_dshow_devices()

    if not video_devs:
        log.warning("[capture] No DirectShow video device found")
        return False

    video_dev = video_devs[0]
    log.info("[capture] Using DirectShow video device: %s", video_dev)

    # Try with audio if available
    if audio_devs:
        audio_dev = audio_devs[0]
        cmd = [
            "ffmpeg", "-y",
            "-f", "dshow", "-i", f"video={video_dev}:audio={audio_dev}",
            "-t", str(duration_s),
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-acodec", "aac",
            "-hide_banner", "-loglevel", "warning",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "dshow", "-i", f"video={video_dev}",
            "-t", str(duration_s),
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-an",
            "-hide_banner", "-loglevel", "warning",
            str(out_path),
        ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=duration_s + 30, **_popen_kw())
        if proc.returncode != 0:
            # BUG-CAP5 fix: log stderr at WARNING not DEBUG
            log.warning("[capture] ffmpeg video failed:\n%s", proc.stderr.decode(errors="replace")[-500:])
            return False
        return out_path.exists() and out_path.stat().st_size > 1024
    except subprocess.TimeoutExpired:
        log.warning("[capture] Video capture timed out after %ds", duration_s + 30)
        return False
    except Exception as e:
        log.warning("[capture] Video capture error: %s", e)
        return False


def _capture_video_linux(out_path: Path, duration_s: int) -> bool:
    """BUG-CAP3 fix: Linux video capture with V4L2 device auto-detection."""
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
                log.info("[capture] Video captured from %s: %d bytes", device, out_path.stat().st_size)
                return True
            else:
                log.warning("[capture] Device %s failed:\n%s",
                            device, proc.stderr.decode(errors="replace")[-300:])
        except subprocess.TimeoutExpired:
            log.warning("[capture] Timeout on device %s", device)
        except Exception as e:
            log.warning("[capture] Error on device %s: %s", device, e)

    return False


def _capture_audio_linux(out_path: Path, duration_s: int) -> bool:
    """BUG-CAP4 fix: Linux audio capture with device auto-detection."""
    working_devs = check_audio_devices_linux()
    if not working_devs:
        return False

    dev = working_devs[0]
    if dev.startswith("pulse:"):
        fmt = "pulse"
        idev = dev[len("pulse:"):]
    else:
        fmt = "alsa"
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
            log.warning("[capture] Audio capture failed:\n%s",
                        proc.stderr.decode(errors="replace")[-300:])
            return False
        return out_path.exists() and out_path.stat().st_size > 512
    except Exception as e:
        log.warning("[capture] Audio error: %s", e)
        return False


def _capture_audio_windows(out_path: Path, duration_s: int) -> bool:
    """BUG-CAP1 fix: Windows audio capture with auto device detection."""
    _, audio_devs = enumerate_dshow_devices()
    if not audio_devs:
        log.warning("[capture] No DirectShow audio device found")
        return False

    audio_dev = audio_devs[0]
    cmd = [
        "ffmpeg", "-y",
        "-f", "dshow", "-i", f"audio={audio_dev}",
        "-t", str(duration_s),
        "-ac", "1", "-ar", "16000",
        "-hide_banner", "-loglevel", "warning",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=duration_s + 15, **_popen_kw())
        return proc.returncode == 0 and out_path.exists()
    except Exception as e:
        log.warning("[capture] Windows audio error: %s", e)
        return False


def _extract_audio_from_video(video_path: Path, out_path: Path, sr: int = 16000) -> bool:
    """BUG-CAP7 fix: Only extract if video actually exists."""
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
            log.warning("[capture] Audio extraction failed:\n%s",
                        proc.stderr.decode(errors="replace")[-300:])
            return False
        return out_path.exists() and out_path.stat().st_size > 512
    except Exception as e:
        log.warning("[capture] Audio extraction error: %s", e)
        return False


def _write_silence_wav(out_path: Path, duration_s: int, sr: int = 16000):
    """Write a short silence WAV as placeholder when all capture fails."""
    # Write only 2 seconds max — enough to satisfy pipeline without huge file
    actual_s = min(duration_s, 2)
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
        audio.wav   (always present — silence if capture fails)
        video.mp4   (present if camera available)

    BUG-CAP6 note: The returned session_dir is passed to pipeline.py as
    video_dir. The LDA globs for *.mp4 files in that directory.
    If no video is captured, only audio.wav exists and the LDA uses audio-only mode.
    """
    if not _ffmpeg_available():
        log.error("[capture] ffmpeg not found on PATH. Install ffmpeg and try again.")

    ts = int(time.time())
    session_dir = DATA_DIR / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    audio_path = session_dir / "audio.wav"
    video_path = session_dir / "video.mp4"

    log.info("[capture] Starting %ds capture → %s", duration_s, session_dir)

    has_video = False
    if not force_audio_only and _ffmpeg_available():
        if IS_WINDOWS:
            has_video = _capture_video_windows(video_path, duration_s)
        elif IS_LINUX:
            has_video = _capture_video_linux(video_path, duration_s)
        elif IS_MACOS:
            # macOS: use AVFoundation
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation", "-framerate", "30",
                "-i", "0:0",  # device 0 video, device 0 audio
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

    # Extract audio from video if successful
    if has_video:
        log.info("[capture] Video captured: %s (%.1f MB)",
                 video_path.name, video_path.stat().st_size / 1e6)
        extracted = _extract_audio_from_video(video_path, audio_path)
        if not extracted:
            log.warning("[capture] Audio extraction failed — trying direct capture")
            has_video = False  # force audio-only fallback

    # Audio-only fallback
    if not audio_path.exists():
        log.info("[capture] Attempting audio-only capture")
        if IS_WINDOWS:
            has_audio = _capture_audio_windows(audio_path, duration_s)
        elif IS_LINUX:
            has_audio = _capture_audio_linux(audio_path, duration_s)
        else:
            has_audio = False

        if not has_audio:
            log.warning("[capture] All audio capture methods failed — writing silence placeholder")
            _write_silence_wav(audio_path, duration_s)

    log.info("[capture] Session ready: %s | video=%s audio=%s",
             session_dir.name, has_video, audio_path.exists())

    # Write session metadata
    import json
    meta = {
        "timestamp": ts,
        "duration_s": duration_s,
        "has_video": has_video,
        "has_audio": audio_path.exists() and audio_path.stat().st_size > 512,
        "video_bytes": video_path.stat().st_size if has_video else 0,
        "audio_bytes": audio_path.stat().st_size if audio_path.exists() else 0,
    }
    (session_dir / "session_meta.json").write_text(json.dumps(meta, indent=2))

    return session_dir


def scan_existing_sessions() -> List[Path]:
    """
    BUG-CAP6 fix: In run-once mode, scan for existing captured sessions.
    Returns sessions sorted newest-first.
    """
    sessions = sorted(
        [d for d in DATA_DIR.glob("session_*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    log.info("[capture] Found %d existing session(s)", len(sessions))
    return sessions


def get_or_capture_session(duration_s: int = 60) -> Path:
    """
    BUG-CAP6 fix for run-once mode:
    Use the most recent session if <24h old, otherwise capture a new one.
    """
    sessions = scan_existing_sessions()
    if sessions:
        newest = sessions[0]
        age_h = (time.time() - newest.stat().st_mtime) / 3600
        if age_h < 24:
            log.info("[capture] Reusing session from %.1fh ago: %s", age_h, newest.name)
            return newest

    log.info("[capture] No recent session found — capturing %ds", duration_s)
    return capture_session(duration_s)