"""Prepare licensed/plain-text corpora for Kaptia's byte-level scratch LM.

Input files: .txt, .md, .jsonl. JSONL records may use "text" or "content".
Documents are split before concatenation to reduce train/validation leakage.
No network access, downloads, MNIST, or pretrained tokenizer are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


SUPPORTED = {".txt", ".md", ".jsonl"}


def read_documents(path: Path) -> Iterable[str]:
    if path.is_file():
        files = [path] if path.suffix.lower() in SUPPORTED else []
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise ValueError(f"No supported .txt/.md/.jsonl files found under {path}")
    for file in files:
        if file.suffix.lower() in {".txt", ".md"}:
            text = file.read_text(encoding="utf-8-sig", errors="strict").strip()
            if text:
                yield text
        else:
            with file.open("r", encoding="utf-8-sig") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {file}:{line_no}: {exc}") from exc
                    if isinstance(item, str):
                        text = item.strip()
                    elif isinstance(item, dict):
                        value = item.get("text", item.get("content"))
                        if not isinstance(value, str):
                            raise ValueError(f"{file}:{line_no}: expected string field 'text' or 'content'")
                        text = value.strip()
                    else:
                        raise ValueError(f"{file}:{line_no}: expected a JSON object or string")
                    if text:
                        yield text


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local text documents for Kaptia scratch pretraining.")
    parser.add_argument("--input", type=Path, required=True, help="A .txt/.md/.jsonl file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("training/scratch_corpus"))
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--min-document-bytes", type=int, default=32)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be greater than 0 and less than 0.5")
    if args.min_document_bytes < 1:
        parser.error("--min-document-bytes must be positive")

    unique: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for raw in read_documents(args.input):
        scanned += 1
        doc = normalize(raw)
        if len(doc.encode("utf-8")) < args.min_document_bytes:
            continue
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(doc)
    if len(unique) < 2:
        raise ValueError("Need at least two unique documents after filtering to create train and validation sets.")

    random.Random(args.seed).shuffle(unique)
    val_count = max(1, round(len(unique) * args.val_fraction))
    val_count = min(val_count, len(unique) - 1)
    validation_docs = unique[:val_count]
    train_docs = unique[val_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # UTF-8 bytes are token IDs 0..255 for ByteTokenizer.
    train_bytes = b"\n\n".join(doc.encode("utf-8") for doc in train_docs)
    val_bytes = b"\n\n".join(doc.encode("utf-8") for doc in validation_docs)
    if len(train_bytes) < 2 or len(val_bytes) < 2:
        raise ValueError("Encoded train and validation corpora must each contain at least two bytes.")
    (args.output_dir / "train.bin").write_bytes(train_bytes)
    (args.output_dir / "val.bin").write_bytes(val_bytes)
    metadata = {
        "format": "raw UTF-8 bytes; each byte is a token ID 0..255",
        "tokenizer": "Kaptia ByteTokenizer; vocab_size=256",
        "source_path": str(args.input),
        "documents_scanned": scanned,
        "documents_after_deduplication_and_filter": len(unique),
        "train_documents": len(train_docs),
        "validation_documents": len(validation_docs),
        "train_bytes": len(train_bytes),
        "validation_bytes": len(val_bytes),
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "validation_sha256": hashlib.sha256(val_bytes).hexdigest(),
        "split_seed": args.seed,
        "validation_fraction_requested": args.val_fraction,
        "warning": "Only use text you have permission to use. Small corpora are for pipeline tests, not useful general-purpose pretraining.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[PASS] Text corpus prepared")
    print(f"Documents: scanned={scanned}, unique/usable={len(unique)}, train={len(train_docs)}, validation={len(validation_docs)}")
    print(f"Train bytes: {len(train_bytes):,}; validation bytes: {len(val_bytes):,}")
    print(f"Output: {args.output_dir.resolve()}")
    print("No network downloads or pretrained tokenizer used.")


if __name__ == "__main__":
    main()
