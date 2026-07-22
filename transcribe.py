#!/usr/bin/env python3
"""VLM-assisted transcription pipeline for handwritten SEA monastic manuscript scripts.

Scripts are CONFIG, not code: every profile lives in scripts.yaml. Adding a new
script requires editing that file only — never this one.

Pipeline: preprocess -> identify -> transcribe (multi-pass) -> review.

    python transcribe.py <dir> [--model claude-sonnet-4-6] [--passes 2]
                                [--force-script id] [--dry-run]

Requires ANTHROPIC_API_KEY in the environment for live runs (not for --dry-run).
"""
from __future__ import annotations

import base64
import datetime as _dt
import difflib
import hashlib
import io
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import click
import yaml
from PIL import Image, ImageOps

try:
    import anthropic
except ImportError:  # allow --dry-run / --help without the SDK installed
    anthropic = None

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "scripts.yaml"
PENDING_PATH = HERE / "scripts.pending.yaml"
CACHE_DIR = Path(".cache")

MODEL_DEFAULT = "claude-sonnet-4-6"
MAX_EDGE = 1568          # Anthropic vision downscale / tile threshold
BAND_OVERLAP = 0.20      # 20% overlap between adjacent bands
IDENT_THRESHOLD = 0.45   # below this top-confidence => unknown mode
MAX_TOKENS = 8000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}

# Sonnet 4.6 pricing ($/1M tokens) — used only for the dry-run estimate.
PRICE_IN = 3.0
PRICE_OUT = 15.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def load_registry(path: Path = REGISTRY_PATH) -> "OrderedDict[str, dict]":
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reg = OrderedDict()
    for entry in data.get("scripts", []):
        reg[entry["id"]] = entry
    if not reg:
        raise click.ClickException(f"No scripts found in {path}")
    return reg


def registry_reference(reg: dict) -> str:
    """Full distinguishing-feature block for every script — a stable, cacheable prefix."""
    out = []
    for sid, e in reg.items():
        out.append(
            f"### {sid}\n"
            f"names: {', '.join(e.get('names', []))}\n"
            f"unicode_block: {e.get('unicode_block', 'n/a')}\n"
            f"regions: {', '.join(e.get('regions', []))}\n"
            f"traditions: {', '.join(e.get('traditions', []))}\n"
            f"distinguishing features:\n{e.get('prompt_notes', '').strip()}\n"
        )
    return "\n".join(out)


def script_notes(reg: dict, ids: list[str]) -> str:
    out = []
    for sid in ids:
        e = reg.get(sid)
        if not e:
            continue
        out.append(
            f"### {sid}  ({', '.join(e.get('names', []))})\n"
            f"unicode_block: {e.get('unicode_block', 'n/a')}\n"
            f"romanization: {e.get('romanization', 'n/a')}\n"
            f"{e.get('prompt_notes', '').strip()}\n"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
def preprocess(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)          # honour camera rotation
    img = ImageOps.grayscale(img)               # L mode
    img = ImageOps.autocontrast(img)
    return img


def _starts(length: int, max_edge: int, overlap: float) -> list[tuple[int, int]]:
    if length <= max_edge:
        return [(0, length)]
    step = max(1, int(max_edge * (1 - overlap)))
    spans, s = [], 0
    while True:
        e = min(s + max_edge, length)
        spans.append((s, e))
        if e >= length:
            break
        s += step
    return spans


def tile_bands(img: Image.Image, max_edge: int = MAX_EDGE,
               overlap: float = BAND_OVERLAP) -> list[tuple[Image.Image, dict]]:
    """Split into overlapping bands, each <= max_edge on both axes. Offsets tracked."""
    w, h = img.size
    xs, ys = _starts(w, max_edge, overlap), _starts(h, max_edge, overlap)
    bands, idx = [], 0
    for (y0, y1) in ys:
        for (x0, x1) in xs:
            crop = img.crop((x0, y0, x1, y1))
            bands.append((crop, {"index": idx, "x": x0, "y": y0,
                                 "w": x1 - x0, "h": y1 - y0}))
            idx += 1
    return bands


def downscale(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_edge:
        return img
    r = max_edge / m
    return img.resize((max(1, int(w * r)), max(1, int(h * r))))


def encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def image_block(img: Image.Image) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": encode_png(img)}}


def est_image_tokens(img: Image.Image) -> int:
    w, h = img.size
    return int((w * h) / 750)


# ---------------------------------------------------------------------------
# Local result cache — key by (stage, file hash, band, pass, script_id, model)
# ---------------------------------------------------------------------------
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cache_key(**parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_get(key: str):
    f = CACHE_DIR / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def cache_put(key: str, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# API client (backoff + structured output)
# ---------------------------------------------------------------------------
class Model:
    def __init__(self, model: str):
        if anthropic is None:
            raise click.ClickException(
                "The 'anthropic' package is not installed. `pip install anthropic`.")
        self.model = model
        self.client = anthropic.Anthropic()   # ANTHROPIC_API_KEY from env
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

    def json_call(self, system_blocks: list[dict], content: list[dict],
                  schema: dict, max_tokens: int = MAX_TOKENS) -> dict:
        delay, last = 1.0, None
        for attempt in range(6):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_blocks,
                    messages=[{"role": "user", "content": content}],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                )
                u = resp.usage
                self.usage["in"] += getattr(u, "input_tokens", 0) or 0
                self.usage["out"] += getattr(u, "output_tokens", 0) or 0
                self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                text = next((b.text for b in resp.content if b.type == "text"), "")
                return json.loads(text)
            except anthropic.RateLimitError as e:
                last = e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last = e
                else:
                    raise
            except anthropic.APIConnectionError as e:
                last = e
            sleep = min(delay * (2 ** attempt) + random.uniform(0, 1), 60.0)
            click.echo(f"    retry {attempt + 1}/6 in {sleep:.1f}s ({type(last).__name__})",
                       err=True)
            time.sleep(sleep)
        raise click.ClickException(f"API failed after retries: {last}")


# ---------------------------------------------------------------------------
# Prompts + schemas
# ---------------------------------------------------------------------------
def cached(text: str) -> dict:
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


IDENT_BASE = (
    "You are an expert palaeographer of Southeast Asian monastic manuscript "
    "scripts (Tham-family, Khom, Mon-Burmese, Shan, and rarer sacred scripts).\n"
    "Given a manuscript image, identify which registry script(s) it is written in.\n"
    "- Rank the TOP-3 candidate scripts by confidence (0.0-1.0) with a short rationale.\n"
    "- Mixed-script pages are NORMAL. If different regions/lines use different "
    "scripts, set mixed=true and list them under 'regions' (describe where).\n"
    "- If NOTHING in the registry matches, or you are genuinely unsure, set "
    "no_registry_match=true and fill 'unknown_description' with what you see "
    "(letterforms, stacking, provenance cues). Otherwise leave it \"\".\n"
    "Use ONLY registry ids for script_id. Output JSON matching the schema."
)

TRANSCRIBE_BASE = (
    "You are an expert transcriber of Southeast Asian monastic manuscripts.\n"
    "Transcribe the manuscript band EXACTLY as written. Rules:\n"
    "- Work LINE BY LINE, preserving the grid: give integer row (top=0) and col "
    "(left=0) for each line/cell.\n"
    "- For an uncertain glyph, write [?] in the text and add an entry to "
    "'uncertain' with your candidate alternatives.\n"
    "- NEVER normalise an ambiguous or archaic manuscript form to a modern print "
    "glyph. Preserve it and flag it as uncertain instead.\n"
    "- Preserve stacked/subscript consonant clusters; do not silently unstack.\n"
    "- Provide 'romanization' for every line: REQUIRED for unencoded scripts "
    "(give a Pali/Sanskrit or language romanization); for encoded scripts it is a "
    "helpful parallel — include it when you can, else \"\".\n"
    "- Set each line's script_id to the registry id it is written in (or 'unknown').\n"
    "- TRANSCRIPTION ONLY. Do NOT translate. Output JSON matching the schema."
)

UNKNOWN_BASE = (
    "The script could not be confidently identified from the registry. "
    "Transcribe in UNKNOWN mode:\n"
    "- In the first line (row 0) put a short VISUAL DESCRIPTION of the script "
    "(letterforms, stacking, ductus) as the 'text'.\n"
    "- In its 'romanization' name the CLOSEST registry script and why.\n"
    "- Then transcribe the remaining lines as a ROMANIZED attempt (best-effort "
    "phonetic reading), script_id 'unknown', flagging uncertainty liberally.\n"
    "Output JSON matching the schema."
)

IDENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"script_id": {"type": "string"},
                           "confidence": {"type": "number"},
                           "rationale": {"type": "string"}},
            "required": ["script_id", "confidence", "rationale"]}},
        "regions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"where": {"type": "string"},
                           "script_id": {"type": "string"},
                           "confidence": {"type": "number"}},
            "required": ["where", "script_id", "confidence"]}},
        "mixed": {"type": "boolean"},
        "no_registry_match": {"type": "boolean"},
        "unknown_description": {"type": "string"},
    },
    "required": ["candidates", "regions", "mixed", "no_registry_match",
                 "unknown_description"],
}

TRANSCRIBE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"lines": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "row": {"type": "integer"}, "col": {"type": "integer"},
            "script_id": {"type": "string"},
            "text": {"type": "string"},
            "romanization": {"type": "string"},
            "uncertain": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"glyph": {"type": "string"},
                               "alternatives": {"type": "array",
                                                "items": {"type": "string"}},
                               "note": {"type": "string"}},
                "required": ["glyph", "alternatives", "note"]}},
        },
        "required": ["row", "col", "script_id", "text", "romanization",
                     "uncertain"]}}},
    "required": ["lines"],
}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def identify(model: Model, reg: dict, full_img: Image.Image, fhash: str) -> dict:
    key = cache_key(stage="ident", fhash=fhash, model=model.model)
    hit = cache_get(key)
    if hit is not None:
        return hit
    system = [cached(IDENT_BASE + "\n\n=== REGISTRY ===\n" + registry_reference(reg))]
    content = [image_block(downscale(full_img)),
               {"type": "text",
                "text": "Identify the script(s) in this manuscript page."}]
    result = model.json_call(system, content, IDENT_SCHEMA, max_tokens=1500)
    cache_put(key, result)
    return result


def transcribe_band(model: Model, reg: dict, band_img: Image.Image, offset: dict,
                    script_ids: list[str], fhash: str, pass_no: int,
                    unknown: bool) -> dict:
    key = cache_key(stage="transcribe", fhash=fhash, band=offset["index"],
                    pass_no=pass_no, scripts=script_ids, unknown=unknown,
                    model=model.model)
    hit = cache_get(key)
    if hit is not None:
        return hit
    if unknown:
        system = [cached(TRANSCRIBE_BASE), cached(UNKNOWN_BASE + "\n\n"
                  "Registry scripts you may reference as the closest match:\n"
                  + registry_reference(reg))]
    else:
        system = [cached(TRANSCRIBE_BASE),
                  cached("=== SCRIPT GUIDANCE (identified + likely alternates) ===\n"
                         + script_notes(reg, script_ids))]
    content = [image_block(band_img),
               {"type": "text",
                "text": (f"Transcribe band index {offset['index']} "
                         f"(pixel offset x={offset['x']}, y={offset['y']}, "
                         f"w={offset['w']}, h={offset['h']}). "
                         "Give row/col per line.")}]
    result = model.json_call(system, content, TRANSCRIBE_SCHEMA)
    cache_put(key, result)
    return result


# ---------------------------------------------------------------------------
# Multi-pass merge (glyph-level diff)
# ---------------------------------------------------------------------------
def _annotate_two(a: str, b: str) -> tuple[str, list[dict]]:
    sm = difflib.SequenceMatcher(None, a, b)
    out, dis = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(a[i1:i2])
        else:
            sa, sb = a[i1:i2], b[j1:j2]
            out.append(f"⚠[{sa or '∅'}|{sb or '∅'}]")
            dis.append({"a": sa, "b": sb})
    return "".join(out), dis


def merge_passes(passes: list[list[dict]]) -> list[dict]:
    """passes = list (one per pass) of line-dict lists. Aligns by (row,col)."""
    n = len(passes)
    keyed: "OrderedDict[tuple, list]" = OrderedDict()
    for pi, lines in enumerate(passes):
        for ln in lines:
            k = (ln.get("row"), ln.get("col"))
            keyed.setdefault(k, [None] * n)[pi] = ln
    merged = []
    for (row, col), perpass in keyed.items():
        present = [l for l in perpass if l]
        base = present[0]
        rec = {"row": row, "col": col,
               "script_id": base.get("script_id"),
               "romanization": base.get("romanization", ""),
               "uncertain": list(base.get("uncertain", []) or []),
               "disagreements": []}
        if len(present) < n:
            rec["disagreements"].append({"type": "missing_in_some_pass"})
        texts = [l.get("text", "") for l in present]
        distinct = list(dict.fromkeys(texts))
        if len(distinct) == 1:
            rec["text"] = distinct[0]
        elif len(distinct) == 2 and n == 2:
            text, dis = _annotate_two(distinct[0], distinct[1])
            rec["text"] = text
            rec["disagreements"] += dis
        else:
            rec["text"] = "⚠[" + " | ".join(distinct) + "]"
            rec["disagreements"].append({"readings": distinct})
        merged.append(rec)
    return merged


def line_needs_review(ln: dict) -> bool:
    return ("⚠" in (ln.get("text") or "") or "[?]" in (ln.get("text") or "")
            or bool(ln.get("uncertain")) or bool(ln.get("disagreements")))


# ---------------------------------------------------------------------------
# Unknown-script pending stub
# ---------------------------------------------------------------------------
def append_pending(image_name: str, description: str, closest: str) -> None:
    data = {}
    if PENDING_PATH.exists():
        data = yaml.safe_load(PENDING_PATH.read_text(encoding="utf-8")) or {}
    pend = data.get("pending", [])
    if any(p.get("from_image") == image_name for p in pend):
        return
    pend.append({
        "id": f"pending_{Path(image_name).stem}",
        "from_image": image_name,
        "closest_script": closest,
        "seen": _dt.date.today().isoformat(),
        "model_description": description or "(no description returned)",
    })
    data["pending"] = pend
    PENDING_PATH.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")


# ---------------------------------------------------------------------------
# Review markdown
# ---------------------------------------------------------------------------
def write_review(target: Path, results: list[dict], reg: dict) -> Path:
    lines = [f"# Transcription review — {target.name}",
             f"_{len(results)} image(s) · generated {_dt.datetime.now():%Y-%m-%d %H:%M}_",
             ""]

    # per-script summary
    summary: "OrderedDict[str, dict]" = OrderedDict()
    for r in results:
        for sid in r["scripts_used"]:
            summary.setdefault(sid, {"images": 0, "lines": 0, "flags": 0})
    for r in results:
        seen = set()
        for ln in r["lines"]:
            sid = ln.get("script_id") or (r["scripts_used"][0] if r["scripts_used"] else "unknown")
            s = summary.setdefault(sid, {"images": 0, "lines": 0, "flags": 0})
            s["lines"] += 1
            if line_needs_review(ln):
                s["flags"] += 1
            seen.add(sid)
        for sid in seen:
            summary[sid]["images"] += 1

    lines += ["## Per-script summary", "",
              "| script | images | lines | flagged | error density |",
              "|---|---:|---:|---:|---:|"]
    for sid, s in summary.items():
        dens = (s["flags"] / s["lines"]) if s["lines"] else 0.0
        lines.append(f"| {sid} | {s['images']} | {s['lines']} | {s['flags']} | {dens:.0%} |")
    lines.append("")

    for r in results:
        cand = ", ".join(f"{c['script_id']} ({c['confidence']:.2f})"
                         for c in r["ident"].get("candidates", [])[:3]) or "—"
        tags = []
        if r["ident"].get("mixed"):
            tags.append("MIXED")
        if r["unknown"]:
            tags.append("UNKNOWN")
        tagstr = f"  **[{' · '.join(tags)}]**" if tags else ""
        lines += [f"## {r['image']}{tagstr}",
                  f"- Identified: {cand}",
                  f"- Passes: {r['passes']}", ""]
        if r["ident"].get("regions"):
            lines.append("- Regions: " + "; ".join(
                f"{rg['where']}→{rg['script_id']} ({rg['confidence']:.2f})"
                for rg in r["ident"]["regions"]))
            lines.append("")

        lines += ["### Transcription", "",
                  "| band | row | col | script | text | romanization |",
                  "|---:|---:|---:|---|---|---|"]
        for ln in r["lines"]:
            txt = (ln.get("text") or "").replace("|", "\\|")
            rom = (ln.get("romanization") or "").replace("|", "\\|")
            lines.append(f"| {ln.get('band')} | {ln.get('row')} | {ln.get('col')} "
                         f"| {ln.get('script_id')} | {txt} | {rom} |")
        lines.append("")

        flagged = [ln for ln in r["lines"] if line_needs_review(ln)]
        lines += ["### Needs human review", ""]
        if not flagged:
            lines += ["_none flagged_", ""]
        else:
            for ln in flagged:
                bits = []
                if ln.get("uncertain"):
                    bits.append("; ".join(
                        f"{u.get('glyph', '?')}→{'/'.join(u.get('alternatives', []))}"
                        for u in ln["uncertain"]))
                if ln.get("disagreements"):
                    bits.append(f"{len(ln['disagreements'])} pass-disagreement(s)")
                bo = ln.get("band_offset", {})
                coord = f"band {ln.get('band')} (x={bo.get('x')}, y={bo.get('y')})"
                lines.append(f"- {coord} · row {ln.get('row')} col {ln.get('col')}: "
                             f"`{ln.get('text')}`" + (f" — {' | '.join(bits)}" if bits else ""))
            lines.append("")

    path = target / "review.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def gather_images(target: Path) -> list[Path]:
    return sorted(p for p in target.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def choose_scripts(ident: dict, force: str | None) -> tuple[list[str], bool]:
    if force:
        return [force], False
    cands = ident.get("candidates", [])
    top_conf = cands[0]["confidence"] if cands else 0.0
    unknown = ident.get("no_registry_match", False) or top_conf < IDENT_THRESHOLD
    # union of region scripts + top-3 candidates, primary first, de-duped
    ids: list[str] = []
    for rg in ident.get("regions", []):
        if rg.get("script_id") and rg["script_id"] not in ids:
            ids.append(rg["script_id"])
    for c in cands[:3]:
        if c["script_id"] not in ids:
            ids.append(c["script_id"])
    return ids, unknown


def run_pipeline(target: Path, model_name: str, passes: int, force: str | None):
    reg = load_registry()
    if force and force not in reg:
        raise click.ClickException(f"--force-script '{force}' not in registry. "
                                   f"Known: {', '.join(reg)}")
    images = gather_images(target)
    if not images:
        raise click.ClickException(f"No images ({', '.join(sorted(IMAGE_EXTS))}) in {target}")

    model = Model(model_name)
    results = []
    for img_path in images:
        click.echo(f"• {img_path.name}")
        fhash = file_hash(img_path)
        proc = preprocess(img_path)

        if force:
            ident = {"candidates": [{"script_id": force, "confidence": 1.0,
                                     "rationale": "forced via --force-script"}],
                     "regions": [], "mixed": False, "no_registry_match": False,
                     "unknown_description": ""}
        else:
            ident = identify(model, reg, proc, fhash)
        (img_path.with_suffix(img_path.suffix + ".ident.json")).write_text(
            json.dumps(ident, ensure_ascii=False, indent=2), encoding="utf-8")

        script_ids, unknown = choose_scripts(ident, force)
        if not script_ids:
            script_ids = ["unknown"]
        click.echo(f"    scripts: {', '.join(script_ids)}"
                   + ("  [UNKNOWN mode]" if unknown else ""))
        if unknown:
            append_pending(img_path.name, ident.get("unknown_description", ""),
                           script_ids[0] if script_ids else "n/a")

        bands = tile_bands(proc)
        img_lines = []
        for band_img, offset in bands:
            per_pass = []
            for p in range(passes):
                res = transcribe_band(model, reg, band_img, offset, script_ids,
                                      fhash, p, unknown)
                per_pass.append(res.get("lines", []))
            for ln in merge_passes(per_pass):
                ln["band"] = offset["index"]
                ln["band_offset"] = offset
                img_lines.append(ln)

        (img_path.with_suffix(img_path.suffix + ".transcription.json")).write_text(
            json.dumps({"image": img_path.name, "scripts_used": script_ids,
                        "unknown": unknown, "passes": passes, "lines": img_lines},
                       ensure_ascii=False, indent=2), encoding="utf-8")

        results.append({"image": img_path.name, "ident": ident,
                        "scripts_used": script_ids, "unknown": unknown,
                        "passes": passes, "lines": img_lines})

    review = write_review(target, results, reg)
    u = model.usage
    cost = (u["in"] + u["cache_write"]) * PRICE_IN / 1e6 + u["out"] * PRICE_OUT / 1e6 \
        + u["cache_read"] * PRICE_IN * 0.1 / 1e6
    click.echo(f"\n✓ {len(results)} image(s) → {review}")
    click.echo(f"  tokens: in={u['in']} out={u['out']} "
               f"cache_read={u['cache_read']} cache_write={u['cache_write']}  "
               f"≈ ${cost:.2f}")


def dry_run_estimate(target: Path, model_name: str, passes: int, force: str | None):
    reg = load_registry()
    images = gather_images(target)
    if not images:
        raise click.ClickException(f"No images in {target}")
    ident_sys = len(IDENT_BASE + registry_reference(reg)) // 4
    trans_sys = len(TRANSCRIBE_BASE + registry_reference(reg)) // 4

    tot_in = tot_out = n_ident = n_trans = 0
    click.echo(f"DRY RUN — {len(images)} image(s), model={model_name}, passes={passes}\n")
    for p in images:
        proc = preprocess(p)
        bands = tile_bands(proc)
        band_tok = sum(est_image_tokens(b) for b, _ in bands)
        if not force:
            n_ident += 1
            tot_in += ident_sys + est_image_tokens(downscale(proc)) + 40
            tot_out += 300
        nt = len(bands) * passes
        n_trans += nt
        tot_in += passes * (band_tok + len(bands) * (trans_sys + 60))
        tot_out += nt * 700
        click.echo(f"  {p.name}: {len(bands)} band(s) → "
                   f"{(0 if force else 1)} ident + {nt} transcribe call(s)")

    cost = tot_in * PRICE_IN / 1e6 + tot_out * PRICE_OUT / 1e6
    click.echo(f"\n  calls: {n_ident} identify + {n_trans} transcribe")
    click.echo(f"  est. input≈{tot_in:,} tok  output≈{tot_out:,} tok")
    click.echo(f"  est. cost ≈ ${cost:.2f}  (upper bound; prompt caching lowers "
               "repeated system cost, local cache skips repeats)")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("directory", type=click.Path(exists=True, file_okay=False,
                                              path_type=Path))
@click.option("--model", default=MODEL_DEFAULT, show_default=True,
              help="Anthropic model id.")
@click.option("--passes", default=2, show_default=True, type=click.IntRange(1, 5),
              help="Independent transcription passes per band (glyph-diffed).")
@click.option("--force-script", "force", default=None,
              help="Skip identification; force this registry script id.")
@click.option("--dry-run", is_flag=True, help="Estimate cost without calling the API.")
def main(directory: Path, model: str, passes: int, force: str | None, dry_run: bool):
    """Transcribe every manuscript image in DIRECTORY."""
    if dry_run:
        dry_run_estimate(directory, model, passes, force)
    else:
        run_pipeline(directory, model, passes, force)


if __name__ == "__main__":
    main()
