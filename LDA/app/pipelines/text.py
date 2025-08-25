# app/pipelines/text.py
import re
from pathlib import Path
import hashlib
from typing import Tuple
from app.utils.receipts import ReceiptManager

try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

class TextPreprocessor:
    def __init__(self, output_dir: str, receipt_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.receipts = ReceiptManager(receipt_dir)
        self.nlp = None
        if _HAS_SPACY:
            try:
                # don't fail if model is missing; user can install en_core_web_sm
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

    def _remove_pii_spacy(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.ent_type_ in ["PERSON", "ORG", "GPE", "DATE", "TIME", "LOC", "MONEY", "EMAIL", "PHONE"]:
                tokens.append(f"<{token.ent_type_}>")
            else:
                tokens.append(token.text)
        return " ".join(tokens)

    def _remove_pii_regex(self, text: str) -> str:
        # remove emails
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<EMAIL>", text)
        # remove phone numbers (simple)
        text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", "<PHONE>", text)
        # remove urls
        text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
        return text

    def _basic_cleaning(self, text: str) -> str:
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process_text(self, text_input: str, file_id: str) -> Tuple[str, str]:
        cleaned = self._basic_cleaning(text_input)
        if self.nlp:
            try:
                anonymized = self._remove_pii_spacy(cleaned)
            except Exception:
                anonymized = self._remove_pii_regex(cleaned)
        else:
            anonymized = self._remove_pii_regex(cleaned)

        out_path = self.output_dir / f"processed_{file_id}.txt"
        out_path.write_text(anonymized, encoding="utf-8")

        receipt_path = self.receipts.create_receipt(
            operation="text_preprocessing",
            input_meta={
                "original_length": len(text_input),
                "anonymized_length": len(anonymized),
                "hash": hashlib.sha256(anonymized.encode()).hexdigest()
            },
            output_uri=str(out_path)
        )
        return str(out_path), receipt_path


# Adapter expected by main: process_text_sources(tdir: Path, cfg: dict, session_id: str) -> List[Dict]
def process_text_sources(tdir, cfg: dict, session_id: str):
    """
    Read all text files in tdir and produce a row per file (simple adapter).
    """
    from pathlib import Path
    tdir = Path(tdir)
    out = []
    tp = TextPreprocessor(cfg.get("text", {}).get("output_dir", "./processed/text"),
                          cfg.get("text", {}).get("receipt_dir", "./receipts"))
    for f in sorted(tdir.glob("*.txt")):
        p, r = tp.process_text(f.read_text(encoding="utf-8"), f.stem)
        out.append({
            "session_id": session_id,
            "modality": "text",
            "source": str(f),
            "processed_uri": p,
            "receipt_path": r
        })
    return out
