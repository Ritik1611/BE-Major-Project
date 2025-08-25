# app/pipelines/video.py
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.utils.receipts import ReceiptManager
import subprocess
import csv
import statistics
import shutil
import tempfile
import os
import logging

log = logging.getLogger(__name__)

class VideoPreprocessor:
    """
    Video pipeline using OpenCV Haar cascade for face detection (no Mediapipe).
    Detects faces per frame, crops frames to faces, runs OpenFace FeatureExtraction
    on those cropped frames, aggregates numeric features from the CSV, and also
    writes an anonymized (faces-blurred) video.
    """

    def __init__(self, output_dir: str, receipt_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.receipts = ReceiptManager(receipt_dir)

        # lazy-loaded libs
        self.cv2 = None
        self.np = None
        self._deps_loaded = False

        # openface config + session id
        self.openface_cfg: Optional[Dict[str, Any]] = None
        self.session_id: Optional[str] = None

    def _ensure_deps(self):
        """Lazy-load cv2 and numpy, raise a friendly error if missing."""
        if self._deps_loaded:
            return
        try:
            import cv2 as _cv2
            import numpy as _np
        except Exception as e:
            raise ImportError("Install opencv-python and numpy in the interpreter that runs the server.") from e

        self.cv2 = _cv2
        self.np = _np
        self._deps_loaded = True

    def _haar_cascade_path(self) -> Optional[str]:
        """Return Haar cascade path from openface_cfg or default to common OpenCV/ OpenFace classifier."""
        # Check explicit config
        cfg = self.openface_cfg or {}
        haar = cfg.get("haar_path")
        if haar and Path(haar).exists():
            return str(Path(haar))
        # fallback: if openface binary path given, try to find classifiers relative to it
        binary = cfg.get("binary_path")
        if binary:
            # assume OpenFace tree: .../build/bin/FeatureExtraction -> ../classifiers/haarcascade_frontalface_alt.xml
            p = Path(binary).resolve()
            # climb up to OpenFace build/bin/classifiers or build/bin/../classifiers
            candidate = p.parent / "classifiers" / "haarcascade_frontalface_alt.xml"
            if candidate.exists():
                return str(candidate)
            # some builds place classifiers sibling to bin
            candidate2 = p.parent.parent / "classifiers" / "haarcascade_frontalface_alt.xml"
            if candidate2.exists():
                return str(candidate2)
        # final fallback: use OpenCV builtin path
        try:
            import cv2
            cand = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cand).exists():
                return cand
        except Exception:
            pass
        return None

    def _detect_and_save_cropped_frames(self, video_path: str, frames_dir: Path, pad: float = 0.15, max_frames: Optional[int] = None) -> int:
        """
        Detect largest face per frame using Haar cascade, crop with padding, and save to frames_dir.
        Returns number of saved frames.
        """
        self._ensure_deps()
        cv2 = self.cv2

        frames_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")

        frame_idx = 0
        saved = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        haar_path = self._haar_cascade_path()
        if not haar_path:
            log.warning("No Haar cascade found; cropped-frame extraction will not run.")
            cap.release()
            return 0

        face_cascade = cv2.CascadeClassifier(haar_path)
        if face_cascade.empty():
            log.warning("Haar cascade failed to load; path: %s", haar_path)
            cap.release()
            return 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            except Exception as e:
                log.debug("Haar detect error on frame %d: %s", frame_idx, e)
                detections = []

            if len(detections) > 0:
                # pick largest face bbox
                x, y, w, h = max(detections, key=lambda b: b[2] * b[3])
                pad_w = int(w * pad)
                pad_h = int(h * pad)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    fname = frames_dir / f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(fname), crop)
                    saved += 1

            if max_frames and saved >= max_frames:
                break

        cap.release()
        return saved

    def _run_openface_on_frames(self, frames_dir: Path, out_dir: Path, openface_bin: str) -> Dict[str, Any]:
        """
        Run OpenFace FeatureExtraction on a directory of images (-fdir).
        Collect the first CSV and compute aggregated numeric features.
        """
        result: Dict[str, Any] = {
            "openface_csv": None,
            "openface_aggregates": {},
            "openface_error": None,
            "openface_receipt": None
        }

        if not frames_dir.exists():
            result["openface_error"] = "frames_dir does not exist"
            return result

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(openface_bin),
            "-fdir", str(frames_dir),
            "-out_dir", str(out_dir),
            "-pose", "-gaze", "-aus"
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                result["openface_error"] = f"OpenFace failed (rc={proc.returncode}): {proc.stderr[:200]}"
                return result
        except FileNotFoundError:
            result["openface_error"] = "OpenFace binary not found"
            return result
        except Exception as e:
            result["openface_error"] = f"OpenFace run exception: {e}"
            return result

        csv_files = list(out_dir.glob("*.csv"))
        if not csv_files:
            result["openface_error"] = f"No CSVs produced by OpenFace in {out_dir}"
            return result

        csv_path = csv_files[0]
        result["openface_csv"] = str(csv_path)

        # parse and aggregate numeric columns
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                accum: Dict[str, List[float]] = {}
                for row in reader:
                    for k, v in row.items():
                        try:
                            fv = float(v)
                            accum.setdefault(k, []).append(fv)
                        except Exception:
                            continue
                aggregates = {k: float(statistics.mean(v)) for k, v in accum.items() if v}
                result["openface_aggregates"] = aggregates
        except Exception as e:
            result["openface_error"] = f"Failed to parse OpenFace CSV: {e}"
            return result

        # create a receipt for openface extraction (non-fatal)
        try:
            result["openface_receipt"] = self.receipts.create_receipt(
                operation="openface_extraction",
                input_meta={"frames_dir": str(frames_dir)},
                output_uri=str(csv_path)
            )
        except Exception:
            result["openface_receipt"] = None

        return result

    def process_video(self, video_path: str) -> tuple[str, str, Dict[str, Any]]:
        """
        Process video:
          - Create an anonymized (faces-blurred) copy using Haar.
          - If OpenFace enabled: detect + crop per-frame faces, run OpenFace on frames, aggregate features.
        Returns: (processed_video_path, receipt_path, features_dict)
        """
        self._ensure_deps()
        cv2 = self.cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video {video_path}")

        out_path = self.output_dir / f"processed_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Load Haar cascade
        haar_path = self._haar_cascade_path()
        face_cascade = None
        if haar_path:
            face_cascade = cv2.CascadeClassifier(haar_path)
            if face_cascade.empty():
                face_cascade = None

        # Read frames and blur faces with Haar detection
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot re-open video for processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = []
            try:
                if face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in detections:
                        faces.append((int(x), int(y), int(w), int(h)))
            except Exception as e:
                log.debug("Haar detect error during anonymization: %s", e)
                faces = []

            # blur faces (largest-first)
            for (x, y, w, h) in sorted(faces, key=lambda b: -(b[2] * b[3])):
                pad_x = int(0.1 * w)
                pad_y = int(0.15 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)
                try:
                    region = frame[y1:y2, x1:x2]
                    blurred = cv2.GaussianBlur(region, (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred
                except Exception:
                    pass

            writer.write(frame)

        cap.release()
        writer.release()

        # receipt for anonymization
        receipt_path = self.receipts.create_receipt(
            operation="video_preprocessing",
            input_meta={"source": video_path, "fps": fps, "resolution": (width, height)},
            output_uri=str(out_path)
        )

        # OpenFace integration
        of_result: Dict[str, Any] = {"openface_error": "not_run"}
        cfg = self.openface_cfg or {}
        if cfg.get("enabled") and cfg.get("binary_path"):
            tmp_root = Path(tempfile.mkdtemp(prefix=f"openface_{self.session_id or 'unsess'}_"))
            frames_dir = tmp_root / "frames"
            out_dir = tmp_root / "openface_out"
            try:
                saved = self._detect_and_save_cropped_frames(str(video_path), frames_dir)
                if saved > 0:
                    of_result = self._run_openface_on_frames(frames_dir, out_dir, cfg["binary_path"])
                else:
                    # if no cropped frames were saved (rare), try running OpenFace on full video by invoking FeatureExtraction -f
                    # note: running on full video may produce a different CSV structure but often works
                    try:
                        # run openface on full video as fallback
                        cmd = [str(cfg["binary_path"]), "-f", str(video_path), "-out_dir", str(out_dir), "-pose", "-gaze", "-aus"]
                        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                        if proc.returncode == 0:
                            of_result = self._run_openface_on_frames(frames_dir, out_dir, cfg["binary_path"])  # will check CSV
                        else:
                            of_result = {"openface_error": f"OpenFace on full video fallback failed: {proc.stderr[:200]}"}
                    except Exception as e:
                        of_result = {"openface_error": f"OpenFace fallback exception: {e}"}
            finally:
                # cleanup tmp folder
                try:
                    shutil.rmtree(tmp_root)
                except Exception:
                    pass
        else:
            of_result = {"openface_error": "disabled"}

        features: Dict[str, Any] = {}
        features.update(of_result)

        return str(out_path), receipt_path, features


# Adapter for main pipeline
def process_video_file(video_path: str, cfg: dict, session_id: str) -> List[Dict[str, Any]]:
    out_dir = cfg.get("ingest", {}).get("video", {}).get("output_dir", "./processed/video")
    receipt_dir = cfg.get("ingest", {}).get("video", {}).get("receipt_dir", "./receipts")
    vp = VideoPreprocessor(out_dir, receipt_dir)
    vp.openface_cfg = cfg.get("video_pipe", {}).get("openface", {})
    vp.session_id = session_id

    out_path, receipt_path, features = vp.process_video(video_path)

    return [{
        "session_id": session_id,
        "modality": "video",
        "source": str(video_path),
        "filename": Path(video_path).name,
        "processed_uri": out_path,
        "receipt_path": receipt_path,
        "features": features
    }]
