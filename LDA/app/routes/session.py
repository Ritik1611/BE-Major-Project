# app/routes/session.py
from fastapi import APIRouter, Body
from typing import Dict, Any, Optional
from pathlib import Path
from app.pipelines.session_processor import process_session_file

router = APIRouter(prefix="/session")

@router.post("/process")
def process_session(body: Dict[str, Any] = Body(...)):
    """
    Example body:
    {
      "session_id": "sess-123",
      "config_uri": "file://configs/local_config.yaml",
      "video_path": "/path/to/video.mp4",
      "audio_path": null,
      "text_input": null,
      "mode": "session",
      "roles": {"spk0": "counsellor", "spk1": "patient"}
    }
    """
    session_id = body.get("session_id")
    cfg_uri = body.get("config_uri")
    video_path = body.get("video_path")
    audio_path = body.get("audio_path")
    text_input = body.get("text_input")
    mode = body.get("mode", "session")
    roles = body.get("roles", None)

    # Load config
    assert cfg_uri.startswith("file://"), "config_uri must be file://"
    import yaml, json
    cfg_path = Path(cfg_uri[len("file://"):])
    cfg = yaml.safe_load(cfg_path.read_text())

    rows, artifacts, receipts = process_session_file(
        session_id=session_id,
        cfg=cfg,
        work_dir=Path("./work") / session_id,
        video_path=video_path,
        audio_path=audio_path,
        text_input=text_input,
        mode=mode,
        roles=roles
    )
    return {"rows": rows, "artifacts": artifacts, "receipts": receipts}
