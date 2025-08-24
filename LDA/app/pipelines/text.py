# app/pipelines/text.py
import re
from pathlib import Path
import hashlib
from app.utils.receipts import ReceiptManager
from typing import List, Dict, Any

class TextPreprocessor:
    def __init__(self, output_dir: str, receipt_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.receipts = ReceiptManager(receipt_dir)

        # Lazy-load spaCy model here so importing this module doesn't require spaCy
        self.nlp = None
        try:
            # import inside try so missing spaCy doesn't break imports
            import spacy  # type: ignore
            try:
                self.nlp = spacy.load("en_core_web_sm")  # For NER-based PII removal
            except Exception:
                # model not downloaded; leave nlp None (we'll skip NER)
                self.nlp = None
        except Exception:
            # spaCy (or its dependencies) not installed — we continue without NER
            self.nlp = None

    def _remove_pii(self, text: str) -> str:
        """Use NER to remove Personally Identifiable Information (PII) from transcripts."""
        if self.nlp is None:
            return text
        doc = self.nlp(text)
        anonymized_tokens = []

        for token in doc:
            if token.ent_type_ in ["PERSON", "ORG", "GPE", "DATE", "TIME", "LOC", "MONEY", "EMAIL", "PHONE"]:
                anonymized_tokens.append(f"<{token.ent_type_}>")
            else:
                anonymized_tokens.append(token.text)

        return " ".join(anonymized_tokens)

    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning: remove special chars, extra spaces."""
        # allow angle brackets (for anonymized tags) and alnum + whitespace
        text = re.sub(r"[^a-zA-Z0-9\s\<\>]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process_text(self, text_input: str, file_id: str) -> tuple[str, str]:
        """Process text input (from transcript or live session). Returns (out_path, receipt_path)."""
        cleaned_text = self._basic_cleaning(text_input)
        anonymized_text = self._remove_pii(cleaned_text)

        # Save to output
        out_path = self.output_dir / f"processed_{file_id}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(anonymized_text)

        # Generate receipt for audit
        receipt_path = self.receipts.create_receipt(
            operation="text_preprocessing",
            input_meta={
                "original_length": len(text_input),
                "anonymized_length": len(anonymized_text),
                "hash": hashlib.sha256(anonymized_text.encode()).hexdigest()
            },
            output_uri=str(out_path)
        )

        return str(out_path), receipt_path


# Adapter expected by app.main: process_text_sources(tdir, cfg, session_id) -> List[Dict]
def process_text_sources(text_dir: Path | str, cfg: dict, session_id: str) -> List[Dict[str, Any]]:
    """
    Walk a directory of text files and return rows suitable for parquet:
    each row is a dict with session_id, source, filename, text (anonymized) and receipt_path.
    """
    text_dir = Path(text_dir)
    out_dir = cfg.get("ingest", {}).get("text", {}).get("output_dir", "./processed/text")
    receipt_dir = cfg.get("ingest", {}).get("text", {}).get("receipt_dir", "./receipts")
    tp = TextPreprocessor(out_dir, receipt_dir)
    rows: List[Dict[str, Any]] = []
    for p in sorted(text_dir.glob("*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        out_path, receipt_path = tp.process_text(raw, p.stem + f"_{session_id}")
        # read anonymized content to include in rows (optional)
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                anonymized = f.read()
        except Exception:
            anonymized = ""
        rows.append({
            "session_id": session_id,
            "source": str(p),
            "filename": p.name,
            "anonymized_text": anonymized,
            "processed_uri": out_path,
            "receipt_path": receipt_path
        })
    return rows
