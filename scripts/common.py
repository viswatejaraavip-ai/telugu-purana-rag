"""Shared helpers for the Purana corpus build (open sources only).

Run every script from the repo root:  .venv/bin/python scripts/<script>.py
"""
import hashlib, json, os, re, unicodedata
from indic_transliteration import sanscript

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCES = os.path.join(ROOT, "sources")
OUT = os.path.join(ROOT, "corpus")
CARDS = os.path.join(ROOT, "cards")
os.makedirs(OUT, exist_ok=True); os.makedirs(CARDS, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s):
    return unicodedata.normalize("NFC", s or "").strip()


def iast_to_telugu(s):
    return sanscript.transliterate(s, sanscript.IAST, sanscript.TELUGU)


def iast_to_devanagari(s):
    return sanscript.transliterate(s, sanscript.IAST, sanscript.DEVANAGARI)


GRETIL_LICENSE = ("GRETIL e-text: CC BY-NC-SA 4.0 (NON-COMMERCIAL). The underlying Sanskrit "
                  "is public domain; the digitisation licence still binds redistribution. "
                  "Re-source from Ambuda/Wikisource before any commercial release.")
