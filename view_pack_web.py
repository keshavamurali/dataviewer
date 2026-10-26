"""Web viewer for a sample pack. Same logic as view_pack.py, in the browser.

Replaces the terminal viewer with a single-page web app (Python standard
library only -- http.server, no web framework). It reuses the exact
invisible-character and hex-dump logic from view_pack.py so the diagnostic
view is byte-for-byte faithful to the script.

Features
  * Diagnostic pane: each document rendered with hidden characters made
    visible (same VISIBLE map and control-char rule as view_pack.py).
  * Character offset on cursor: click or move the cursor in the diagnostic
    pane and the status bar shows the CODE-POINT offset into the raw text
    (identical to Python text[offset]) plus the code point and Unicode name.
  * Selection -> evidence + hex: select a span to auto-fill the evidence text
    (with "at offset N") and the space-separated "U+XXXX U+XXXX" hex string,
    in the exact format of labels_TEMPLATE.csv.
  * Markdown preview: a rendered preview of the document (toggle).
  * Hex dump pane: same offset / code-point / Unicode-name list as the script.
  * Disease highlights: candidates for DISEASES.md sections A (characters) and
    B (markup and ghost tokens) are highlighted in the diagnostic pane and
    listed with one-click "+" to tick the code and append evidence/hex.
    B3 ghost markers get a red banner. Toggle highlights with "g".
  * Per-row annotation: textboxes for every CSV column (read, verdict,
    diseases, evidence, hex, note). Save keeps the row; Export downloads a CSV
    matching labels_TEMPLATE.csv exactly.

Usage
    python view_pack_web.py <pack.parquet> [--start N] [--port 8000]
    python view_pack_web.py <pack.parquet> --out labels.csv   # server-side save

Then open http://127.0.0.1:8000 in a browser.
"""
import argparse
import csv
import html
import io
import json
import os
import re
import unicodedata
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Shared logic (identical to view_pack.py).
# ---------------------------------------------------------------------------
VISIBLE = {
    "\u200b": "\u27e6ZWSP\u27e7", "\u200c": "\u27e6ZWNJ\u27e7",
    "\u200d": "\u27e6ZWJ\u27e7", "\ufeff": "\u27e6BOM\u27e7",
    "\u00ad": "\u27e6SHY\u27e7", "\u202a": "\u27e6LRE\u27e7",
    "\u202b": "\u27e6RLE\u27e7", "\u202c": "\u27e6PDF\u27e7",
    "\u202d": "\u27e6LRO\u27e7", "\u202e": "\u27e6RLO\u27e7",
    "\u2066": "\u27e6LRI\u27e7", "\u2067": "\u27e6RLI\u27e7",
    "\u2068": "\u27e6FSI\u27e7", "\u2069": "\u27e6PDI\u27e7",
    "\ufffd": "\u27e6U+FFFD\u27e7", "\t": "\u27e6TAB\u27e7",
}


def token_for(ch: str):
    """Return the visible token for ch, or None if ch is shown as-is.

    Mirrors view_pack.py show(): mapped specials, then non-newline control
    chars as U+XXXX, else the character itself (including newline).
    """
    if ch in VISIBLE:
        return VISIBLE[ch]
    if ch != "\n" and unicodedata.category(ch) == "Cc":
        return f"\u27e6U+{ord(ch):04X}\u27e7"
    return None


def show(text: str) -> str:
    """Identical to view_pack.py show()."""
    out = []
    for ch in text:
        tok = token_for(ch)
        out.append(tok if tok is not None else ch)
    return "".join(out)


def hexdump_rows(text: str, limit: int = 2000):
    """Same format as view_pack.py hexdump(), as structured rows, but listing
    every character (ASCII letters included), not just non-ASCII/control/separator.

    Each row: {offset, cp (U+XXXX), name}. Offsets are code-point indices.
    """
    rows = []
    for i, ch in enumerate(text[:limit]):
        rows.append({
            "offset": i,
            "cp": f"U+{ord(ch):04X}",
            "name": unicodedata.name(ch, "?"),
        })
    return rows


def segments(text: str, marks=None):
    """Split text into display segments carrying raw code-point offsets.

    Each segment is one of:
      {"t": "text", "s": "...", "off": start_offset}            normal run
      {"t": "tok",  "s": "token", "raw": ch, "cp": "U+XXXX",
       "name": "...", "off": offset}                            invisible/ctrl

    A newline is emitted as its own text segment so the frontend can break
    lines while keeping exact offsets. Offsets are code-point indices into the
    raw text (what Python text[off] returns).

    marks (optional, from detect()) is a per-character list of finding dicts or
    None; a run is split wherever the finding changes, and segments inside a
    finding carry "dx" (disease code) and "dxl" (label) for highlighting.
    """
    segs = []
    buf = []
    buf_start = 0

    def mark_of(i):
        return marks[i] if marks else None

    def flush():
        if buf:
            seg = {"t": "text", "s": "".join(buf), "off": buf_start}
            m = mark_of(buf_start)
            if m:
                seg["dx"], seg["dxl"] = m["code"], m["label"]
            segs.append(seg)
            buf.clear()

    for i, ch in enumerate(text):
        tok = token_for(ch)
        if tok is None and ch != "\n":
            if buf and mark_of(i) is not mark_of(i - 1):
                flush()
            if not buf:
                buf_start = i
            buf.append(ch)
            continue
        flush()
        if ch == "\n":
            segs.append({"t": "text", "s": "\n", "off": i})
        else:
            seg = {
                "t": "tok", "s": tok, "raw": ch, "off": i,
                "cp": f"U+{ord(ch):04X}",
                "name": unicodedata.name(ch, "?"),
            }
            m = mark_of(i)
            if m:
                seg["dx"], seg["dxl"] = m["code"], m["label"]
            segs.append(seg)
    flush()
    return segs


# ---------------------------------------------------------------------------
# Disease detectors for sections A (characters) and B (markup and ghost
# tokens) of DISEASES.md. They only highlight candidates; the reader decides.
# ---------------------------------------------------------------------------
# A1: invisible junk. ZWJ U+200D / ZWNJ U+200C are deliberately absent: they
# are part of Indic script, not a disease.
A1_CHARS = {
    "\u200b": "zero-width space", "\ufeff": "BOM / zero-width no-break space",
    "\u00ad": "soft hyphen", "\u2060": "word joiner",
    "\u200e": "bidi mark (LRM)", "\u200f": "bidi mark (RLM)", "\u061c": "bidi mark (ALM)",
    **{chr(c): "bidi control" for c in range(0x202A, 0x202F)},
    **{chr(c): "bidi isolate" for c in range(0x2066, 0x206A)},
}


def _is_a2(ch):
    o = ord(ch)
    return (o < 0x20 and ch not in "\t\n") or 0x7F <= o <= 0x9F


# A3: UTF-8 bytes shown as cp1252/latin-1, e.g. "â€™", "Ã¤", "à¤¹". A
# continuation byte 0x80-0xBF appears as its cp1252 char (or raw latin-1 for
# the bytes cp1252 leaves undefined).
def _cp1252_char(b):
    try:
        return bytes([b]).decode("cp1252")
    except UnicodeDecodeError:
        return bytes([b]).decode("latin-1")


_BYTE_OF = {_cp1252_char(b): b for b in range(0x80, 0x100)}
_CONT = "[" + "".join(sorted({_cp1252_char(b) for b in range(0x80, 0xC0)}
                             | {chr(b) for b in range(0x80, 0xC0)})) + "]"
MOJIBAKE_RE = re.compile(
    "(?:[\u00c2-\u00df]" + _CONT + "|[\u00e0-\u00ef]" + _CONT + "{2}"
    "|[\u00f0-\u00f4]" + _CONT + "{3})+")


def _unmojibake(s):
    """Return the intended text if s is UTF-8 decoded as cp1252, else None."""
    try:
        return bytes(_BYTE_OF.get(c, ord(c)) for c in s).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


_HTML_TAGS = (
    "a|abbr|article|aside|audio|b|blockquote|body|br|button|center|code|div|em|"
    "figcaption|figure|font|footer|form|h[1-6]|head|header|hr|html|i|iframe|img|"
    "input|label|li|link|main|meta|nav|noscript|ol|option|p|path|pre|script|"
    "section|select|small|source|span|strong|style|sub|sup|svg|table|tbody|td|"
    "th|thead|title|tr|u|ul|video")

# (code, regex, label). B3 first: it is the most dangerous disease.
PATTERNS = [
    ("B3", re.compile(r"<\|[A-Za-z0-9_.\- ]{1,40}\|>"), "ghost special token"),
    ("B3", re.compile(r"\[/?INST\]|<</?SYS>>"), "ghost chat marker"),
    ("B3", re.compile(r"</?s>"), "ghost BOS/EOS marker <s>"),
    ("B3", re.compile(r"</?think(?:ing)?>"), "ghost reasoning tag"),
    ("B3", re.compile(r"<(?:bos|eos|unk|pad|mask|start_of_turn|end_of_turn)>"
                      r"|\[(?:CLS|SEP|MASK|PAD|UNK)\]"), "ghost special token"),
    ("B3", re.compile(r"(?mi)^[ \t]*#{1,4}[ \t]*(?:assistant|user|human|system|"
                      r"instruction|response)[ \t]*:"), "ghost role header"),
    ("B3", re.compile(r"(?m)^[ \t]*(?:Human|Assistant)[ \t]*:"), "ghost role prefix"),
    ("B2", re.compile(r"<!--.{0,500}?-->|<!DOCTYPE[^>]{0,200}>|<\?xml[^>]{0,200}\?>",
                      re.S | re.I), "HTML/XML comment or declaration"),
    # attributes must have values, so math like "a<b and c>d" is not a tag
    ("B2", re.compile(r"</?(?:" + _HTML_TAGS + r")(?:\s+[\w:-]+\s*=\s*(?:\"[^\"<>]*\"|'[^'<>]*'|[^\s<>]+))*"
                      r"\s*/?>", re.I), "HTML tag"),
    ("B1", re.compile(r"&(?:#\d{1,7};?|#[xX][0-9A-Fa-f]{1,6};?|[A-Za-z][A-Za-z0-9]{1,31};"
                      r"|(?:nbsp|amp|lt|gt|quot|apos)\b)"), "HTML entity"),
    ("B4", re.compile(r"\{\{.{0,200}?\}\}|\{%.{0,200}?%\}"), "template placeholder"),
    ("B4", re.compile(r"\$\{[^}\n]{0,100}\}"), "template placeholder ${...}"),
    ("B4", re.compile(r"\b(?:undefined|NaN)\b|\[object Object\]"), "code junk value"),
]

# lang_label script suffix (e.g. kan_Knda) -> Unicode name prefix
SCRIPT_OF_LABEL = {
    "Knda": "KANNADA", "Deva": "DEVANAGARI", "Beng": "BENGALI", "Taml": "TAMIL",
    "Telu": "TELUGU", "Mlym": "MALAYALAM", "Gujr": "GUJARATI", "Guru": "GURMUKHI",
    "Orya": "ORIYA", "Latn": "LATIN", "Arab": "ARABIC", "Cyrl": "CYRILLIC",
}
# A code's priority when findings overlap: earlier wins the highlight.
PRIORITY = ["B3", "A4", "A1", "A3", "A2", "B5", "B2", "B1", "B4", "A5", "A6"]


def _script(ch):
    """Unicode script name prefix for a letter or mark, else None."""
    cat = unicodedata.category(ch)
    if cat[0] not in "LM" or cat == "Lm":   # Lm: modifier letters act like punctuation
        return None
    first = unicodedata.name(ch, "").split(" ")[0]
    return None if first in ("", "COMBINING") else first


def _char_runs(text, classify):
    """Yield (start, end, label) for runs of consecutive chars with the same label."""
    start, label = None, None
    for i, ch in enumerate(text):
        lab = classify(ch)
        if lab != label:
            if label is not None:
                yield start, i, label
            start, label = i, lab
    if label is not None:
        yield start, len(text), label


def _unclosed_latex(text):
    """B5: \\cmd{ whose brace never closes (any of its {..}{..} arguments),
    \\begin{env} with no \\end{env}, odd $$."""
    out = []
    for m in re.finditer(r"\\[A-Za-z]+\s*\{", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            if text[j] == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
            if depth == 0 and j < len(text) and text[j] == "{":   # next argument group
                depth, j = 1, j + 1
        if depth:
            out.append((m.start(), m.end(), "unclosed " + m.group(0).strip()))
    for env in set(re.findall(r"\\begin\{([^}\n]{1,40})\}", text)):
        begins = list(re.finditer(r"\\begin\{" + re.escape(env) + r"\}", text))
        extra = len(begins) - len(re.findall(r"\\end\{" + re.escape(env) + r"\}", text))
        for m in begins[len(begins) - extra:] if extra > 0 else []:
            out.append((m.start(), m.end(), f"\\begin{{{env}}} with no \\end"))
    dd = [m.start() for m in re.finditer(r"\$\$", text)]
    if len(dd) % 2:
        out.append((dd[-1], dd[-1] + 2, "unpaired $$"))
    return out


def _unclosed_fence(text):
    """B5: a ``` / ~~~ code fence that is opened but never closed."""
    open_at = None
    for m in re.finditer(r"(?m)^[ \t]*(```|~~~)", text):
        open_at = None if open_at is not None else (m.start(1), m.end(1))
    return [(*open_at, "unclosed code fence")] if open_at else []


def _hex(s, cap=64):
    cps = [f"U+{ord(c):04X}" for c in s[:cap]]
    return " ".join(cps) + (" …" if len(s) > cap else "")


def detect(text, lang_label=""):
    """Find section A/B disease candidates in text.

    Returns (findings, marks): findings is a list of
    {code, start, end, label, evidence, hex} sorted by offset (hex only for the
    A group, as the label sheet asks); marks is a per-character list holding the
    finding that highlights that character (or None).
    """
    raw = []
    for s, e, lab in _char_runs(text, lambda ch: A1_CHARS.get(ch)):
        raw.append(("A1", s, e, lab))
    for s, e, lab in _char_runs(text, lambda ch: "control character" if _is_a2(ch) else None):
        raw.append(("A2", s, e, lab))
    for s, e, lab in _char_runs(text, lambda ch: "replacement character" if ch == "\ufffd" else None):
        raw.append(("A4", s, e, f"{lab} x{e - s}" if e - s > 1 else lab))
    for m in MOJIBAKE_RE.finditer(text):
        fixed = _unmojibake(m.group(0))
        if fixed is not None:
            raw.append(("A3", m.start(), m.end(), f"mojibake, should be {fixed!r}"))
    for code, rx, lab in PATTERNS:
        for m in rx.finditer(text):
            raw.append((code, m.start(), m.end(), lab))
    for s, e, lab in _unclosed_fence(text) + _unclosed_latex(text):
        raw.append(("B5", s, e, lab))

    expected = SCRIPT_OF_LABEL.get(lang_label.rpartition("_")[2])
    nfc_ok = unicodedata.is_normalized("NFC", text)
    if not nfc_ok:
        for m in re.finditer(r"\S+", text):
            tok = m.group(0)
            if not unicodedata.is_normalized("NFC", tok):
                raw.append(("A5", m.start(), m.end(),
                            f"not NFC; NFC form is {unicodedata.normalize('NFC', tok)!r}"))
    # A6 words: runs of letters/marks (ZWJ/ZWNJ included), so punctuation splits words
    is_wordch = lambda ch: "w" if ch in "\u200c\u200d" or _script(ch) else None
    for s, e, _ in _char_runs(text, is_wordch):
        scripts = {sc for sc in map(_script, text[s:e]) if sc}
        if len(scripts) > 1:
            raw.append(("A6", s, e, "mixed scripts in one word: " + "+".join(sorted(scripts))))
        elif expected and scripts and scripts - {expected, "LATIN"}:
            raw.append(("A6", s, e, f"{scripts.pop()} word in {lang_label} text"))

    # keep a finding unless higher-priority findings already cover all of it
    raw.sort(key=lambda f: (PRIORITY.index(f[0]), f[1]))
    marks = [None] * len(text)
    findings = []
    for code, s, e, lab in raw:
        if all(marks[i] is not None for i in range(s, e)):
            continue
        seg = text[s:e]
        ev = show(seg)
        f = {
            "code": code, "start": s, "end": e, "label": lab,
            "evidence": (ev if len(ev) <= 60 else ev[:57] + "…") + f" at offset {s}",
            "hex": _hex(seg) if code.startswith("A") else "",
        }
        findings.append(f)
        for i in range(s, e):
            if marks[i] is None:
                marks[i] = f
    findings.sort(key=lambda f: (f["start"], PRIORITY.index(f["code"])))
    return findings, marks


# ---------------------------------------------------------------------------
# Server.
# ---------------------------------------------------------------------------
class State:
    docs = []          # list of row dicts
    out_path = None    # CSV save path (default labels.csv in CWD)
    doc_index = {}     # doc_id -> pack position, for stable CSV ordering


def doc_payload(i: int):
    d = State.docs[i]
    text = d.get("text", "") or ""
    findings, marks = detect(text, d.get("lang_label", "") or "")
    return {
        "i": i,
        "count": len(State.docs),
        "doc_id": d.get("doc_id", ""),
        "source": d.get("source", ""),
        "lang_label": d.get("lang_label", ""),
        "url": d.get("url", "") or "",
        "chars": len(text),
        "raw_text": text,
        "segments": segments(text, marks),
        "hexdump": hexdump_rows(text),
        "dx_counts": dict(Counter(f["code"] for f in findings)),
        "findings": findings[:MAX_FINDINGS],
    }


MAX_FINDINGS = 1000  # rows sent to the browser per document; counts are always complete


FIELDNAMES = ["doc_id", "read", "verdict", "diseases", "evidence", "hex", "note"]


def rows_to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDNAMES, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (r.get(k, "") or "") for k in FIELDNAMES})
    return buf.getvalue()


def read_csv_rows(path):
    """Read labels.csv into {doc_id: rowdict}. Missing/empty file -> {}."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            did = (row.get("doc_id") or "").strip()
            if did:
                out[did] = {k: (row.get(k, "") or "") for k in FIELDNAMES}
    return out


def ordered_rows(rowmap):
    """Rows sorted by pack position; unknown doc_ids go last, stable."""
    def key(did):
        return (0, State.doc_index[did]) if did in State.doc_index else (1, 0)
    return [rowmap[d] for d in sorted(rowmap, key=lambda d: key(d))]


def merge_save(incoming):
    """Upsert incoming rows into the on-disk CSV by doc_id; never delete.

    Re-reads the current file so rows the browser never touched are preserved.
    Returns (abs_path, total_rows).
    """
    disk = read_csv_rows(State.out_path)
    for r in incoming:
        did = (r.get("doc_id") or "").strip()
        if did:
            disk[did] = {k: (r.get(k, "") or "") for k in FIELDNAMES}
    csv_text = rows_to_csv(ordered_rows(disk))
    with open(State.out_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    return os.path.abspath(State.out_path), len(disk)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = dict(
            (k, v) for k, _, v in (p.partition("=") for p in query.split("&") if p)
        )
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/count":
            self._json({"count": len(State.docs)})
        elif path == "/api/rows":
            # current on-disk labels.csv, keyed by doc_id (source of truth)
            self._json({"rows": read_csv_rows(State.out_path)})
        elif path == "/api/doc":
            try:
                i = int(params.get("i", "0"))
            except ValueError:
                return self._json({"error": "bad index"}, 400)
            if not (0 <= i < len(State.docs)):
                return self._json({"error": "out of range"}, 404)
            self._json(doc_payload(i))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path, _, _ = self.path.partition("?")
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        if path in ("/api/save", "/api/export"):
            rows = data.get("rows", [])
            saved_path, total = merge_save(rows)
            self._json({"saved": saved_path, "rows": total})
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--out", default="labels.csv",
                    help="CSV save path (default: labels.csv in the current directory)")
    a = ap.parse_args()

    State.docs = pq.read_table(a.pack).to_pylist()
    State.out_path = a.out
    State.doc_index = {d.get("doc_id", ""): i for i, d in enumerate(State.docs)}
    start = max(0, min(a.start, len(State.docs) - 1)) if State.docs else 0

    global PAGE
    PAGE = PAGE.replace("__START__", str(start)).replace("__VISIBLE__", json.dumps(VISIBLE))

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    shown_host = "127.0.0.1" if a.host == "0.0.0.0" else a.host
    url = f"http://{shown_host}:{a.port}"
    print(f"Loaded {len(State.docs)} documents from {a.pack}")
    print(f"Serving on {a.host}:{a.port}  ->  open {url}  (Ctrl-C to stop)")
    if State.out_path:
        print(f"Save button writes: {os.path.abspath(State.out_path)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


# ---------------------------------------------------------------------------
# Frontend (single inlined page, offline, no external assets).
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pack viewer</title>
<style>
  :root { --bg:#1e1e1e; --panel:#252526; --fg:#d4d4d4; --muted:#888;
          --accent:#4ec9b0; --tok:#f0a500; --tokbg:#4a3a00; --border:#3a3a3a; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg);
         color:var(--fg); font-size:14px; }
  header { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
           padding:8px 12px; background:var(--panel);
           border-bottom:1px solid var(--border); position:sticky; top:0; z-index:5; }
  header .meta { color:var(--muted); font-size:12px; }
  button, input, select, textarea { font-family:inherit; font-size:13px; }
  button { background:#333; color:var(--fg); border:1px solid var(--border);
           border-radius:4px; padding:4px 10px; cursor:pointer; }
  button:hover { background:#3d3d3d; }
  button.active { background:var(--accent); color:#000; }
  .wrap { display:flex; height:calc(100vh - 100px); }
  .left { flex:1 1 60%; overflow:auto; padding:12px; }
  .right { flex:0 0 360px; overflow:auto; padding:12px; background:var(--panel);
           border-left:1px solid var(--border); }
  #diag { white-space:pre-wrap; word-break:break-word; line-height:1.55;
          font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
          font-size:13px; }
  .tok { color:var(--tok); background:var(--tokbg); border-radius:3px;
         padding:0 2px; cursor:help; }
  /* disease highlights: A = characters (orange), B = markup (purple), B3 = ghost markers (red) */
  .dx { border-bottom:2px solid; cursor:pointer; }
  .dxA { background:rgba(255,165,0,.22); border-color:#ffa500; }
  .dxB { background:rgba(170,120,255,.25); border-color:#aa78ff; }
  .dxB3 { background:rgba(255,70,70,.45); border-color:#ff4646; font-weight:bold; }
  #diag.nohl .dx { background:transparent; border-bottom:none; font-weight:inherit; }
  #dxBanner { display:none; background:#5a1a1a; color:#ffd0d0; padding:6px 10px;
              border-radius:4px; margin-bottom:8px; font-size:12px; }
  #dxList { max-height:280px; overflow:auto; font-size:12px; }
  .dxrow { display:flex; gap:6px; align-items:baseline; padding:3px 2px; cursor:pointer;
           border-bottom:1px solid #2c2c2c; }
  .dxrow:hover { background:#2d2d2d; }
  .dxrow.added { opacity:.5; }
  .dxrow .dxev { font-family:monospace; word-break:break-all; }
  .dxrow small { color:var(--muted); }
  .dxc { font-family:monospace; padding:0 3px; border-radius:3px; border-bottom:none; }
  .dxadd { padding:0 7px; }
  .chip { display:inline-block; margin:0 4px 4px 0; padding:1px 7px; border-radius:10px;
          font-size:11px; cursor:pointer; border:1px solid var(--border); }
  .chip.on { outline:1px solid var(--accent); }
  #md { line-height:1.6; max-width:52em; }
  #md pre { background:#111; padding:10px; overflow:auto; border-radius:4px; }
  #md code { background:#111; padding:1px 4px; border-radius:3px; }
  #md table { border-collapse:collapse; } #md td,#md th { border:1px solid var(--border); padding:4px 8px; }
  #md blockquote { border-left:3px solid var(--border); margin:0; padding-left:10px; color:var(--muted); }
  footer { position:sticky; bottom:0; background:var(--panel);
           border-top:1px solid var(--border); padding:6px 12px; font-size:12px;
           display:flex; gap:16px; align-items:center; z-index:5; }
  .status b { color:var(--accent); }
  .hexrow { font-family:monospace; font-size:12px; white-space:pre; }
  .field { margin-bottom:8px; }
  .field label { display:block; color:var(--muted); font-size:11px;
                 text-transform:uppercase; letter-spacing:.04em; margin-bottom:2px; }
  .field input, .field textarea, .field select { width:100%; background:#1b1b1b;
     color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:5px; }
  .field textarea { resize:vertical; min-height:38px; }
  .row-actions { display:flex; gap:8px; margin:8px 0; }
  #dxPicked { width:100%; background:#1b1b1b; color:var(--accent); border:1px solid var(--border);
              border-radius:4px; padding:5px; font-family:monospace; }
  .dxsections { border:1px solid var(--border); border-radius:4px; margin-top:4px; }
  details.dxsec { border-bottom:1px solid var(--border); }
  details.dxsec:last-child { border-bottom:none; }
  details.dxsec > summary { cursor:pointer; padding:5px 8px; font-size:12px; color:var(--fg);
                            list-style:none; user-select:none; background:#1b1b1b; }
  details.dxsec > summary::-webkit-details-marker { display:none; }
  details.dxsec > summary:hover { background:#262626; }
  details.dxsec[open] > summary { color:var(--accent); }
  .dxitems { padding:4px 8px 6px; }
  .dxitem { display:flex; align-items:flex-start; gap:6px; padding:2px 0; font-size:12px; }
  .dxitem input { width:auto; margin-top:2px; }
  .dxitem .code { color:var(--tok); font-family:monospace; min-width:26px; }
  .dxitem .desc { color:var(--fg); }
  .dxitem .desc small { color:var(--muted); }
  h3 { margin:14px 0 6px; font-size:12px; color:var(--muted);
       text-transform:uppercase; letter-spacing:.05em; }
  .hint { color:var(--muted); font-size:11px; margin-top:2px; }
  .warn { color:#e0a0a0; font-size:11px; }
  #exportBox { width:100%; height:120px; background:#111; color:var(--fg);
               border:1px solid var(--border); font-family:monospace; font-size:11px; }
  .saved { color:var(--accent); }
</style>
</head>
<body>
<header>
  <button id="prev">\u25c0 back</button>
  <button id="next">next \u25b6</button>
  <span>#<input id="jump" type="number" min="1" style="width:70px"> </span>
  <input id="jumpId" placeholder="jump to doc_id" style="width:160px">
  <button id="goId">go</button>
  <button id="toggleMd">markdown preview</button>
  <button id="toggleHex" class="active">hex dump</button>
  <button id="toggleDx" class="active">highlights</button>
  <span class="meta" id="meta"></span>
</header>

<div class="wrap">
  <div class="left">
    <div id="dxBanner"></div>
    <div id="diag"></div>
    <div id="md" style="display:none"></div>
  </div>
  <div class="right">
    <h3>Annotation (this document)</h3>
    <div class="field"><label>doc_id</label><input id="f_doc_id" readonly></div>
    <div class="field"><label>read</label>
      <select id="f_read"><option value=""></option><option>full</option><option>partial</option></select></div>
    <div class="field"><label>verdict</label>
      <select id="f_verdict"><option value=""></option><option>clean</option><option>fix</option><option>drop</option></select></div>
    <div class="field"><label>diseases (tick below &rarr; e.g. A1;B3)</label>
      <input id="f_diseases" placeholder="codes appear here; you can also type">
      <div class="dxsections" id="dxSections"></div></div>
    <div class="field"><label>evidence</label><textarea id="f_evidence"></textarea>
      <div class="hint">Select text in the left pane to auto-fill "chars at offset N".</div></div>
    <div class="field"><label>hex (U+XXXX ...)</label><input id="f_hex">
      <div class="warn" id="hexWarn" style="display:none">F1/E3: leave hex empty. Never paste the value.</div></div>
    <div class="field"><label>note</label><input id="f_note"></div>
    <div class="row-actions">
      <button id="clearRow">clear this row (UI only)</button>
      <span id="rowState" class="hint"></span>
    </div>

    <h3>Save (<span id="savedCount">0</span> rows in file)</h3>
    <div class="row-actions">
      <button id="exportBtn">save labels.csv</button>
    </div>
    <textarea id="exportBox" readonly placeholder="current row (CSV) appears here"></textarea>
    <div class="hint" id="exportMsg"></div>

    <h3>Detected A/B candidates <span id="dxTotal"></span></h3>
    <div class="hint" style="margin-bottom:6px">Click a row to select it in the text; + ticks the code and appends evidence (and hex for A).</div>
    <div id="dxCounts"></div>
    <div id="dxList"></div>

    <h3 id="hexHead">Hex dump (all characters)</h3>
    <div id="hexList"></div>
  </div>
</div>

<footer>
  <span class="status" id="status">offset: \u2013</span>
  <span class="status" id="selstatus"></span>
</footer>

<script>
// ---- tiny markdown renderer (bundled JS, no deps) ----
function escapeHtml(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function mdInline(s){
  s = escapeHtml(s);
  s = s.replace(/`([^`]+)`/g,'<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*]+)\*/g,'$1<em>$2</em>');
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function renderMarkdown(text){
  const lines = text.split('\n');
  let out = [], i = 0;
  while(i < lines.length){
    let line = lines[i];
    if(/^```/.test(line)){
      let code=[]; i++;
      while(i<lines.length && !/^```/.test(lines[i])){ code.push(lines[i]); i++; }
      i++;
      out.push('<pre><code>'+escapeHtml(code.join('\n'))+'</code></pre>'); continue;
    }
    let h = line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ const n=h[1].length; out.push('<h'+n+'>'+mdInline(h[2])+'</h'+n+'>'); i++; continue; }
    if(/^\s*>/.test(line)){
      let q=[]; while(i<lines.length && /^\s*>/.test(lines[i])){ q.push(lines[i].replace(/^\s*>\s?/,'')); i++; }
      out.push('<blockquote>'+mdInline(q.join('<br>'))+'</blockquote>'); continue;
    }
    if(/^\s*([-*+]|\d+\.)\s+/.test(line)){
      const ordered=/^\s*\d+\./.test(line); let items=[];
      while(i<lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])){
        items.push('<li>'+mdInline(lines[i].replace(/^\s*([-*+]|\d+\.)\s+/,''))+'</li>'); i++; }
      out.push((ordered?'<ol>':'<ul>')+items.join('')+(ordered?'</ol>':'</ul>')); continue;
    }
    if(/^\s*\|.*\|\s*$/.test(line) && i+1<lines.length && /^\s*\|[-:\s|]+\|\s*$/.test(lines[i+1])){
      const parse=r=>r.trim().replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
      const head=parse(line); i+=2; let body=[];
      while(i<lines.length && /^\s*\|.*\|\s*$/.test(lines[i])){ body.push(parse(lines[i])); i++; }
      let t='<table><thead><tr>'+head.map(c=>'<th>'+mdInline(c)+'</th>').join('')+'</tr></thead><tbody>';
      t+=body.map(r=>'<tr>'+r.map(c=>'<td>'+mdInline(c)+'</td>').join('')+'</tr>').join('');
      out.push(t+'</tbody></table>'); continue;
    }
    if(line.trim()===''){ i++; continue; }
    let para=[]; while(i<lines.length && lines[i].trim()!=='' && !/^(#{1,6}\s|```|\s*>|\s*([-*+]|\d+\.)\s)/.test(lines[i])){ para.push(lines[i]); i++; }
    out.push('<p>'+mdInline(para.join(' '))+'</p>');
  }
  return out.join('\n');
}

// ---- state ----
let cur = null;           // current doc payload
let idx = parseInt("__START__",10) || 0;
const VISIBLE = __VISIBLE__;   // same hidden-char tokens as the Python show()
const saved = {};         // doc_id -> row object (what the UI shows)
let disk = {};            // doc_id -> row as last read from labels.csv
const LS_KEY = 'view_pack_web_rows';
// NOTE: seeding happens in bootstrap(): CSV (source of truth) first, then
// localStorage overlaid on top. localStorage holds ONLY unsaved edits (rows that
// differ from the CSV), and save sends only those rows, so rows changed in
// labels.csv by someone else are never overwritten by a stale browser copy.
const ROW_KEYS = ['doc_id','read','verdict','diseases','evidence','hex','note'];
function sameRow(a, b){ return !!a && !!b && ROW_KEYS.every(k=>(a[k]||'')===(b[k]||'')); }
function dirtyRows(){
  const out = {};
  for(const [k, r] of Object.entries(saved)) if(!sameRow(r, disk[k])) out[k] = r;
  return out;
}
function persistLocal(){
  try { localStorage.setItem(LS_KEY, JSON.stringify(dirtyRows())); } catch(e){}
}

const $ = id => document.getElementById(id);

// ---- disease catalog (verbatim from DISEASES.md) ----
const DISEASES = [
 ["A","Characters",[
   ["A1","invisible junk characters","ZWSP U+200B, BOM U+FEFF, bidi marks, soft hyphen (ZWJ/ZWNJ are NOT diseases in Indic)"],
   ["A2","control characters","U+0000-U+001F except tab/newline, U+007F-U+009F"],
   ["A3","mojibake","UTF-8 decoded twice, e.g. Ã¤ à¤¹ â€™"],
   ["A4","replacement characters","U+FFFD \ufffd ; count them"],
   ["A5","wrong normal form","same word in two byte forms; nukta two ways (we use NFC)"],
   ["A6","wrong or mixed script","Hindi with Latin look-alikes; Bengali word in Assamese letters"]]],
 ["B","Markup and ghost tokens",[
   ["B1","HTML entities","&amp; &nbsp; &#2361;"],
   ["B2","HTML or XML tags","<div> <br/> </p>"],
   ["B3","ghost chat markers","<|im_start|> <|endoftext|> [INST] <s> ### Assistant: <think> (most dangerous)"],
   ["B4","template/code junk in prose","{{ variable }} ${...} undefined NaN"],
   ["B5","broken markdown or LaTeX","unclosed code fence, \\frac{ with no end"]]],
 ["C","Document structure",[
   ["C1","boilerplate","menus, cookie banners, Share on WhatsApp, footer links"],
   ["C2","repeated lines","same line many times in one document"],
   ["C3","truncated document","text stops mid-sentence or mid-code-block"],
   ["C4","two documents joined","two unrelated texts, no boundary"],
   ["C5","lists of keywords or links","SEO spam, tag clouds"]]],
 ["D","Language",[
   ["D1","wrong language label","label says Marathi, text is Hindi"],
   ["D2","romanised Indic","Hindi written in Latin letters"],
   ["D3","mixed language","Hindi + English in one sentence (can be natural; label, do not judge)"],
   ["D4","bad machine translation","wrong meaning, broken grammar, untranslated words"],
   ["D5","translationese","grammatical, but no native speaker writes like this"]]],
 ["E","Code",[
   ["E1","minified or generated code","one very long line; auto-generated do not edit"],
   ["E2","data blobs","base64, long number arrays, embedded images"],
   ["E3","secrets","API keys, passwords, tokens, private keys (hex empty, position only)"],
   ["E4","repeated licence headers","same 30-line licence at top of many files"],
   ["E5","broken encoding in code","mojibake inside strings or comments"],
   ["E6","not code","README, log file or CSV labelled as code"]]],
 ["F","Safety, privacy and contamination",[
   ["F1","personal data","emails, phones, addresses, IDs of private people (hex empty, position only)"],
   ["F2","benchmark leak","a question/solution from a benchmark test set"],
   ["F3","harmful content","content we must not train on"]]],
 ["G","Damage from earlier cleaning",[
   ["G1","cut at the vowel sign","word split before a matra or virama, e.g. \u0915 \u093f"],
   ["G2","lost ZWJ or ZWNJ","a conjunct that renders wrong"],
   ["G3","Indic dropped by an English rule","good doc with low quality score; sections missing"],
   ["G4","over-normalised","NFKC changed chars; Indic digits changed to Latin"]]],
 ["H","Duplicates",[
   ["H1","exact duplicate in the pack",""],
   ["H2","near duplicate","same text with small changes"],
   ["H3","same content in another source or language","translation of a doc seen in English"]]],
];
// all catalog codes, to tell ticked codes from typed-in unknown ones
const DX_KNOWN = new Set();
DISEASES.forEach(([,,items])=>items.forEach(([code])=>DX_KNOWN.add(code)));

function buildDiseasePicker(){
  const box = $('dxSections'); box.innerHTML='';
  for(const [sec, title, items] of DISEASES){
    const det=document.createElement('details'); det.className='dxsec';
    const sum=document.createElement('summary'); sum.textContent=sec+'. '+title;
    det.appendChild(sum);
    const wrap=document.createElement('div'); wrap.className='dxitems';
    for(const [code, name, how] of items){
      const row=document.createElement('label'); row.className='dxitem';
      const cb=document.createElement('input'); cb.type='checkbox'; cb.value=code;
      cb.dataset.dx='1'; cb.addEventListener('change', onDxCheck);
      const desc=document.createElement('span'); desc.className='desc';
      desc.innerHTML='<span class="code">'+code+'</span> '+name + (how?' <small>\u2014 '+how.replace(/</g,'&lt;')+'</small>':'');
      row.appendChild(cb); row.appendChild(desc);
      if(how) row.title=how;
      wrap.appendChild(row);
    }
    det.appendChild(wrap); box.appendChild(det);
  }
}
function dxCheckboxes(){ return Array.from($('dxSections').querySelectorAll('input[data-dx]')); }
// checkboxes -> diseases string in the order picked (so it lines up with evidence/hex):
// existing codes keep their place, unticked ones drop out, newly ticked ones go last;
// typed-in unknown codes are preserved
function onDxCheck(){
  const ticked = new Set(dxCheckboxes().filter(c=>c.checked).map(c=>c.value));
  const existing = ($('f_diseases').value||'').split(';').map(s=>s.trim()).filter(Boolean);
  const kept = existing.map(c=>DX_KNOWN.has(c.toUpperCase()) ? c.toUpperCase() : c)
                       .filter(c=>!DX_KNOWN.has(c) || ticked.has(c));
  const all = [...new Set([...kept, ...ticked])];
  $('f_diseases').value = all.join(';');
  updateHexWarn();
  renderRowPreview();
}
// diseases string -> checkbox states (for load / manual typing)
function syncDiseaseChecks(){
  const codes = new Set(($('f_diseases').value||'').toUpperCase().split(';').map(s=>s.trim()).filter(Boolean));
  dxCheckboxes().forEach(c=>{ c.checked = codes.has(c.value); });
}

function cpLen(s){ return Array.from(s).length; }           // code-point length
function cpAt(s, n){ return Array.from(s)[n]; }

// Build diagnostic DOM; each node carries data-off (code-point offset of its
// first char) and data-len (code-point length of its raw content).
function renderDiag(payload){
  const diag = $('diag'); diag.innerHTML='';
  for(const seg of payload.segments){
    if(seg.t==='text'){
      if(seg.s==='\n'){ diag.appendChild(document.createTextNode('\n')); continue; }
      const span=document.createElement('span');
      span.textContent=seg.s; span.dataset.off=seg.off; span.dataset.len=cpLen(seg.s);
      span.dataset.kind='text'; markDx(span, seg); diag.appendChild(span);
    } else {
      const span=document.createElement('span');
      span.className='tok'; span.textContent=seg.s;
      span.dataset.off=seg.off; span.dataset.len=1; span.dataset.kind='tok';
      span.dataset.raw=seg.raw; span.dataset.cp=seg.cp; span.dataset.name=seg.name;
      span.title=seg.cp+' '+seg.name+'  (offset '+seg.off+')';
      markDx(span, seg); diag.appendChild(span);
    }
  }
}

function dxClass(code){ return 'dx'+code[0]+' dx'+code; }
function markDx(span, seg){
  if(!seg.dx) return;
  span.className += ' dx '+dxClass(seg.dx);
  span.title = seg.dx+': '+seg.dxl + (span.title ? '  |  '+span.title : '  (offset '+seg.off+')');
}

// DOM position for a raw code-point offset; isEnd places it just after char off-1.
function domPos(off, isEnd){
  const target = isEnd ? off-1 : off;
  for(const span of $('diag').querySelectorAll('span[data-off]')){
    const o=+span.dataset.off, n=+span.dataset.len;
    if(target>=o && target<o+n){
      const k = target-o + (isEnd?1:0);
      if(span.dataset.kind==='tok') return [span, k];
      return [span.firstChild, Array.from(span.textContent).slice(0,k).join('').length];
    }
  }
  return null;
}
// Select raw range [s,e) in the diagnostic pane, as if the reader had dragged over it.
function selectRange(s, e){
  if($('diag').style.display==='none') $('toggleMd').click();
  const a=domPos(s,false), b=domPos(e,true); if(!a||!b) return;
  const r=document.createRange(); r.setStart(a[0],a[1]); r.setEnd(b[0],b[1]);
  const sel=window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
  (a[0].nodeType===3?a[0].parentElement:a[0]).scrollIntoView({block:'center'});
  onCursor();
}

// ---- detected A/B candidates panel ----
let dxFilter = null;
function markFinding(f){
  const cb = dxCheckboxes().find(c=>c.value===f.code);
  if(cb && !cb.checked){ cb.checked=true; onDxCheck(); }
  appendValue('f_evidence', f.evidence);
  if(f.hex) appendValue('f_hex', f.hex);
  renderRowPreview();
}
function renderFindings(){
  const counts = cur.dx_counts||{}, codes = Object.keys(counts).sort();
  const total = codes.reduce((a,c)=>a+counts[c],0);
  $('dxTotal').textContent = total ? '('+total+')' : '';
  const banner=$('dxBanner');
  banner.style.display = counts.B3 ? 'block' : 'none';
  banner.textContent = counts.B3 ? '\u26a0 B3 ghost chat markers: '+counts.B3+'. Most dangerous disease: a ghost marker teaches the model to write our special tokens as plain text.' : '';
  const cbox=$('dxCounts'); cbox.innerHTML='';
  for(const c of codes){
    const chip=document.createElement('span');
    chip.className='chip '+dxClass(c)+(dxFilter===c?' on':'');
    chip.textContent=c+' \u00d7'+counts[c]; chip.title='show only '+c+' (click again for all)';
    chip.onclick=()=>{ dxFilter = dxFilter===c ? null : c; renderFindings(); };
    cbox.appendChild(chip);
  }
  const list=$('dxList'); list.innerHTML='';
  if(!total){ list.innerHTML='<div class="hint">(none found)</div>'; return; }
  const have = new Set(($('f_evidence').value||'').split('; '));
  for(const f of cur.findings.filter(f=>!dxFilter||f.code===dxFilter)){
    const row=document.createElement('div'); row.className='dxrow'+(have.has(f.evidence)?' added':'');
    const add=document.createElement('button'); add.className='dxadd'; add.textContent='+';
    add.title='tick '+f.code+', append evidence'+(f.hex?' and hex':'');
    add.onclick=ev=>{ ev.stopPropagation(); markFinding(f); row.classList.add('added'); };
    const code=document.createElement('span'); code.className='dxc '+dxClass(f.code); code.textContent=f.code;
    const body=document.createElement('div');
    const evs=document.createElement('span'); evs.className='dxev'; evs.textContent=f.evidence;
    const lab=document.createElement('small'); lab.textContent=f.label;
    body.append(evs, document.createElement('br'), lab);
    row.append(add, code, body);
    row.onclick=()=>selectRange(f.start, f.end);
    list.appendChild(row);
  }
  if(cur.findings.length < total){
    const more=document.createElement('div'); more.className='hint';
    more.textContent='(list shows the first '+cur.findings.length+' of '+total+')';
    list.appendChild(more);
  }
}

// Map a DOM (node, domOffset) to a raw code-point offset in the text.
function rawOffsetOf(node, domOffset){
  let span = node.nodeType===3 ? node.parentElement : node;
  if(!span || span.id==='diag'){
    // cursor between spans: approximate via previous element
    const el = node.childNodes && node.childNodes[domOffset-1];
    if(el && el.dataset) return parseInt(el.dataset.off)+parseInt(el.dataset.len);
    return null;
  }
  if(span.dataset.off===undefined) return null;
  const base = parseInt(span.dataset.off);
  if(span.dataset.kind==='tok') return base + (domOffset>0?1:0);
  // text span: domOffset counts UTF-16 units; convert to code points
  const txt = span.textContent.slice(0, domOffset);
  return base + cpLen(txt);
}

function describeOffset(off){
  if(off===null || off===undefined || !cur) return 'offset: \u2013';
  const ch = cpAt(cur.raw_text, off);
  if(ch===undefined) return 'offset: '+off+' (end)';
  const cp = ch.codePointAt(0);
  const hex = 'U+'+cp.toString(16).toUpperCase().padStart(4,'0');
  let name=''; // name not shipped per-char; show cp. Tokens carry names via title.
  return 'offset: <b>'+off+'</b> \u00b7 '+hex;
}

function onCursor(){
  const sel = window.getSelection();
  if(!sel.rangeCount){ return; }
  const r = sel.getRangeAt(0);
  const off = rawOffsetOf(r.startContainer, r.startOffset);
  $('status').innerHTML = describeOffset(off);

  if(!sel.isCollapsed){
    const startOff = rawOffsetOf(r.startContainer, r.startOffset);
    const endOff = rawOffsetOf(r.endContainer, r.endOffset);
    if(startOff!==null && endOff!==null && endOff>startOff){
      const arr = Array.from(cur.raw_text).slice(startOff, endOff);
      const raw = arr.join('');
      const hex = arr.map(c=>'U+'+c.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')).join(' ');
      // visible form of selection for evidence
      const vis = arr.map(c=>{ if(VISIBLE[c]) return VISIBLE[c]; const n=c.codePointAt(0); return (n<32||(n>=127&&n<=159))?('\u27e6U+'+n.toString(16).toUpperCase().padStart(4,'0')+'\u27e7'):c; }).join('');
      $('selstatus').innerHTML = 'selected ['+startOff+'\u2013'+endOff+'] \u00b7 '+hex;
      window._selFill = { evidence: vis+' at offset '+startOff, hex: hex };
    }
  } else {
    $('selstatus').innerHTML=''; window._selFill=null;
    if(off!==null && off!==undefined){
      navigator.clipboard && navigator.clipboard.writeText(String(off)).then(()=>{
        $('status').innerHTML = describeOffset(off) + ' <span class="saved">\u2713 copied</span>';
      }).catch(()=>{});
    }
  }
}

function fillFields(docId){
  const row = saved[docId] || {};
  $('f_doc_id').value = docId;
  $('f_read').value = row.read||'';
  $('f_verdict').value = row.verdict||'';
  $('f_diseases').value = row.diseases||'';
  $('f_evidence').value = row.evidence||'';
  $('f_hex').value = row.hex||'';
  $('f_note').value = row.note||'';
  $('rowState').textContent = saved[docId] ? 'saved' : '';
  syncDiseaseChecks();
  updateHexWarn();
  renderRowPreview();
}

function updateHexWarn(){
  const dis=($('f_diseases').value||'').toUpperCase();
  $('hexWarn').style.display = (dis.includes('F1')||dis.includes('E3')) ? 'block':'none';
}

// CSV-quote one field to match Python csv (quote if it has , " or newline; double quotes)
function csvField(s){
  s = (s==null)?'':String(s);
  return /[",\n\r]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s;
}
const CSV_HEADER = 'doc_id,read,verdict,diseases,evidence,hex,note';
function currentRowValues(){
  return {
    doc_id: $('f_doc_id').value, read: $('f_read').value, verdict: $('f_verdict').value,
    diseases: $('f_diseases').value, evidence: $('f_evidence').value,
    hex: $('f_hex').value, note: $('f_note').value,
  };
}
// Live preview of the row being built, in the CSV box below the save button.
function renderRowPreview(){
  const r = currentRowValues();
  const line = ['doc_id','read','verdict','diseases','evidence','hex','note']
                 .map(k=>csvField(r[k])).join(',');
  $('exportBox').value = CSV_HEADER + '\n' + line;
}

async function load(i){
  commitCurrent();            // keep in-progress edits when traversing
  const res = await fetch('/api/doc?i='+i);
  if(!res.ok){ return; }
  cur = await res.json(); idx = cur.i;
  $('meta').textContent = '['+(cur.i+1)+'/'+cur.count+']  doc_id='+cur.doc_id
     +'  source='+cur.source+'  lang='+cur.lang_label+'  chars='+cur.chars
     + (cur.url? '  '+cur.url : '');
  $('jump').value = cur.i+1;
  renderDiag(cur);
  // markdown
  $('md').innerHTML = renderMarkdown(cur.raw_text);
  // hex dump
  const hl=$('hexList'); hl.innerHTML = cur.hexdump.length
    ? cur.hexdump.map(h=>'<div class="hexrow">'+String(h.offset).padStart(6)+'  '+h.cp+'  '+escapeHtml(h.name)+'</div>').join('')
    : '<div class="hint">(empty document)</div>';
  fillFields(cur.doc_id);
  dxFilter = null; renderFindings();
  $('status').innerHTML='offset: \u2013'; $('selstatus').innerHTML='';
}

// ---- events ----
$('prev').onclick = ()=> load(Math.max(0, idx-1));
$('next').onclick = ()=> load(Math.min((cur?cur.count:1)-1, idx+1));
$('jump').onchange = e => { const n=parseInt(e.target.value,10)-1; if(!isNaN(n)) load(n); };
$('goId').onclick = async ()=>{
  const id=$('jumpId').value.trim(); if(!id) return;
  const res=await fetch('/api/count'); const {count}=await res.json();
  for(let k=0;k<count;k++){ const r=await (await fetch('/api/doc?i='+k)).json(); if(r.doc_id===id){ load(k); return; } }
  alert('doc_id not found');
};
$('toggleMd').onclick = e => {
  const md=$('md'), dg=$('diag');
  const on = md.style.display==='none';
  md.style.display = on?'block':'none'; dg.style.display = on?'none':'block';
  e.target.classList.toggle('active', on);
};
$('toggleDx').onclick = e => {
  const on = $('diag').classList.toggle('nohl');
  e.target.classList.toggle('active', !on);
};
$('toggleHex').onclick = e => {
  const on = $('hexList').style.display==='none';
  $('hexList').style.display = on?'block':'none';
  $('hexHead').style.display = on?'block':'none';
  e.target.classList.toggle('active', on);
};
document.getElementById('diag').addEventListener('mouseup', onCursor);
document.getElementById('diag').addEventListener('keyup', onCursor);
document.addEventListener('selectionchange', ()=>{ if(document.activeElement && document.activeElement.tagName==='TEXTAREA') return; onCursor(); });

// use current selection to fill evidence+hex
// each new selection is appended once ("; "-separated); re-focusing without a new selection adds nothing
function appendSelFill(fieldId, key){
  const f = window._selFill;
  const used = key+'Used';
  if(!f || f[used]) return;
  f[used] = true;
  appendValue(fieldId, f[key]);
  renderRowPreview();
}
function appendValue(fieldId, value){
  const el = $(fieldId);
  if(el.value.split('; ').includes(value)) return;   // same span already recorded
  el.value = el.value ? el.value+'; '+value : value;
}
$('f_evidence').addEventListener('focus', ()=>appendSelFill('f_evidence', 'evidence'));
$('f_hex').addEventListener('focus', ()=>appendSelFill('f_hex', 'hex'));
$('f_diseases').addEventListener('input', ()=>{ syncDiseaseChecks(); updateHexWarn(); renderRowPreview(); });
['f_read','f_verdict','f_evidence','f_hex','f_note'].forEach(id=>{
  $(id).addEventListener('input', renderRowPreview);
  $(id).addEventListener('change', renderRowPreview);
});

// Commit the current doc's field values into the in-memory store + localStorage
// buffer (does NOT touch disk). Called on navigation and before saving.
function commitCurrent(){
  if(!cur) return;
  const r = currentRowValues();
  // keep a row only if something was entered
  const nonEmpty = ['read','verdict','diseases','evidence','hex','note'].some(k=>r[k]);
  // an existing CSV row that is edited down to empty stays in the UI, so it is not lost
  if(nonEmpty || disk[cur.doc_id]){ saved[cur.doc_id] = r; }
  persistLocal();
  $('savedCount').textContent = Object.keys(saved).length;
}

$('clearRow').onclick = ()=>{
  if(!cur) return; delete saved[cur.doc_id];
  persistLocal();
  fillFields(cur.doc_id); $('savedCount').textContent = Object.keys(saved).length;
  $('exportMsg').textContent = 'cleared in UI (labels.csv unchanged; delete rows there manually)';
};

$('exportBtn').onclick = async ()=>{
  commitCurrent();
  const rows = Object.values(dirtyRows());   // only edited rows; the server merges by doc_id
  const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({rows})});
  const data = await res.json();
  if(data.saved){
    // re-read the merged file: it is now the baseline, and picks up rows changed elsewhere
    try {
      disk = (await (await fetch('/api/rows')).json()).rows || {};
      Object.assign(saved, disk);
    } catch(e){}
    persistLocal();
  }
  $('exportMsg').innerHTML = data.saved
     ? '<span class="saved">saved \u2192 '+data.saved+' ('+data.rows+' rows in file)</span>'
     : 'save failed';
  $('rowState').textContent = 'saved';
  $('savedCount').textContent = data.rows;
  renderRowPreview();
};

document.addEventListener('keydown', e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
  if(e.key==='ArrowRight' || e.key==='Enter') $('next').click();
  else if(e.key==='ArrowLeft' || e.key==='b') $('prev').click();
  else if(e.key==='m') $('toggleMd').click();
  else if(e.key==='h') $('toggleHex').click();
  else if(e.key==='g') $('toggleDx').click();
});

async function bootstrap(){
  // 1) CSV is the source of truth
  try {
    const data = await (await fetch('/api/rows')).json();
    disk = data.rows || {};
    Object.assign(saved, disk);
  } catch(e){}
  // 2) overlay localStorage (unsaved edits win)
  try {
    const ls = JSON.parse(localStorage.getItem(LS_KEY)||'{}');
    Object.assign(saved, ls);
  } catch(e){}
  persistLocal();   // drop buffered rows that already match the CSV
  $('savedCount').textContent = Object.keys(saved).length;
  buildDiseasePicker();
  load(idx);
}
bootstrap();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
