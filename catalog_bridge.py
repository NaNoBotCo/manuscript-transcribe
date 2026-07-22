#!/usr/bin/env python3
"""catalog_bridge — connect the transcription pipeline to the crawler's catalog.

transcribe.py is deliberately provenance-blind: it reads a folder of images and
writes JSON beside them, knowing nothing about manuscript ids. This bridge is the
seam that carries provenance across that gap, in two file-based steps:

    stage   MSID [outdir]     content store -> folder + manifest.json
    ingest  FOLDER            manifest + *.transcription.json -> catalog pages

`stage` is read-only. It copies a manuscript's page images out of the crawler's
content-addressed store (keyed by sha256) into a working folder named p0001.jpg…,
and records filename -> (manuscript_id, page_no, sha256) in manifest.json.

`ingest` reads that manifest plus the transcription JSON transcribe.py produced,
and writes the flattened text into catalog.db `pages` — CREATING rows for the
1349 image-only manuscripts the OCR digester never reached. It never touches
ocr_text / vision_desc: the VLM transcription lands in its own additive columns,
so the seam between tesseract-OCR text and VLM-transcribed text is provenance,
not structure. It also drops a display copy into store_pages/<mid>/ so the page
shows in the wiki plate gallery, exactly like the digester does.

Idempotent: re-running ingest upserts. Coexists with a running crawl via a long
busy_timeout.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent

# The catalog + stores live in the sibling crawler project. Paths are overridable
# via env (CATALOG_DB / PAGES_STORE / CONTENT_STORE) — handy for testing against a
# throwaway copy without touching the real catalog.
CRAWLER = HERE.parent / "manuscript-crawler"
CATALOG_DB = Path(os.environ.get("CATALOG_DB", CRAWLER / "crawler" / "catalog.db"))
PAGES_STORE = Path(os.environ.get("PAGES_STORE", CRAWLER / "store_pages"))
CONTENT_STORE_DEFAULT = os.environ.get(
    "CONTENT_STORE", "/Volumes/Passport5TB/catalog-data/store")  # sha256 CAS

MANIFEST = "manifest.json"
TRANSCRIPTION_COLUMNS = {
    "transcription": "TEXT",
    "transcription_script": "TEXT",
    "transcription_conf": "REAL",
    "transcribed_at": "TEXT",
}


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def connect(db: Path = CATALOG_DB):
    import sqlite3
    if not db.is_file():
        raise click.ClickException(f"catalog.db not found at {db}")
    con = sqlite3.connect(str(db), timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=120000")
    return con


# ---------------------------------------------------------------------------
# stage — content store -> working folder + manifest
# ---------------------------------------------------------------------------
def blob_path(store: Path, sha: str) -> Path:
    return store / sha[:2] / sha


def do_stage(msid: int, outdir: Path | None, store: Path) -> Path:
    con = connect()
    try:
        m = con.execute(
            "SELECT id, source_identifier, title_thai, title_english, "
            "genre_normalized, script FROM manuscripts WHERE id=?", (msid,)).fetchone()
        if not m:
            raise click.ClickException(f"no manuscript id={msid}")
        rows = con.execute(
            "SELECT sequence, sha256, bytes FROM images "
            "WHERE manuscript_id=? AND sha256 IS NOT NULL AND sha256<>'' "
            "ORDER BY sequence", (msid,)).fetchall()
    finally:
        con.close()
    if not rows:
        raise click.ClickException(f"manuscript {msid} has no downloaded images")

    title = m["title_thai"] or m["title_english"] or m["source_identifier"]
    slug = (m["source_identifier"] or f"ms{msid}").replace("/", "_")
    outdir = outdir or (HERE / "samples" / f"ms{msid}_{m['genre_normalized'] or 'x'}")
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {"manuscript_id": msid, "source_identifier": m["source_identifier"],
                "title": title, "script": m["script"],
                "genre": m["genre_normalized"], "pages": []}
    staged = missing = 0
    for r in rows:
        page_no = r["sequence"] + 1          # pages.page_no is 1-indexed
        src = blob_path(store, r["sha256"])
        fname = f"p{page_no:04}.jpg"
        if src.is_file():
            (outdir / fname).write_bytes(src.read_bytes())
            staged += 1
        else:
            missing += 1
            click.echo(f"    ! missing blob for page {page_no} ({r['sha256'][:12]})")
        manifest["pages"].append({"file": fname, "page_no": page_no,
                                  "sequence": r["sequence"], "sha256": r["sha256"]})

    (outdir / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"  staged {staged} page(s)"
               + (f", {missing} missing" if missing else "")
               + f" -> {outdir}")
    click.echo(f"  {title}  [{m['script']} · {m['genre_normalized']}]")
    click.echo(f"  next: python transcribe.py '{outdir}'  then  "
               f"python catalog_bridge.py ingest '{outdir}'")
    return outdir


# ---------------------------------------------------------------------------
# ingest — transcription JSON -> catalog pages
# ---------------------------------------------------------------------------
def ensure_columns(con):
    have = {r["name"] for r in con.execute("PRAGMA table_info(pages)")}
    for col, typ in TRANSCRIPTION_COLUMNS.items():
        if col not in have:
            con.execute(f"ALTER TABLE pages ADD COLUMN {col} {typ}")
    con.commit()


def flatten_lines(lines: list[dict]) -> str:
    """One text block: bands in order, lines joined, romanization kept inline when
    it is the only reading (unencoded scripts). Uncertainty markers are preserved
    verbatim — the point is that the flagged text stays honest, not tidy."""
    out = []
    for ln in sorted(lines, key=lambda l: (l.get("band", 0), l.get("row", 0),
                                           l.get("col", 0))):
        txt = (ln.get("text") or "").strip()
        rom = (ln.get("romanization") or "").strip()
        if txt and rom and rom != txt:
            out.append(f"{txt}  ({rom})")
        elif txt:
            out.append(txt)
        elif rom:
            out.append(rom)
    return "\n".join(out)


def png_copy(src_img: Path, mid: int, page_no: int) -> str | None:
    """Drop a display copy into store_pages/<mid>/pNNNN.png (wiki gallery store).
    Returns the image_path relative to the crawler dir, or None on failure."""
    try:
        from PIL import Image
    except ImportError:
        return None
    dest_dir = PAGES_STORE / str(mid)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"p{page_no:04}.png"
    rel = f"store_pages/{mid}/p{page_no:04}.png"
    if dest.is_file():
        return rel
    try:
        with Image.open(src_img) as im:
            im.convert("RGB").save(dest, "PNG")
        return rel
    except Exception as e:
        click.echo(f"    ! image copy failed p{page_no}: {e}")
        return None


def do_ingest(folder: Path):
    manifest_path = folder / MANIFEST
    if not manifest_path.is_file():
        raise click.ClickException(
            f"no {MANIFEST} in {folder} — was this folder produced by `stage`?")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mid = manifest["manuscript_id"]

    con = connect()
    try:
        if not con.execute("SELECT 1 FROM manuscripts WHERE id=?", (mid,)).fetchone():
            raise click.ClickException(f"manuscript {mid} absent from catalog")
        ensure_columns(con)

        wrote = skipped = 0
        for pg in manifest["pages"]:
            page_no = pg["page_no"]
            tpath = folder / (pg["file"] + ".transcription.json")
            if not tpath.is_file():
                skipped += 1
                continue
            t = json.loads(tpath.read_text(encoding="utf-8"))
            text = flatten_lines(t.get("lines", []))
            if not text:
                skipped += 1
                continue
            scripts = ",".join(t.get("scripts_used", [])) or None

            conf = None
            ipath = folder / (pg["file"] + ".ident.json")
            if ipath.is_file():
                cands = json.loads(ipath.read_text(encoding="utf-8")).get("candidates", [])
                if cands:
                    conf = cands[0].get("confidence")

            image_path = png_copy(folder / pg["file"], mid, page_no)

            con.execute(
                """INSERT INTO pages
                       (manuscript_id, page_no, image_path, kind,
                        transcription, transcription_script, transcription_conf,
                        transcribed_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(manuscript_id, page_no) DO UPDATE SET
                       transcription=excluded.transcription,
                       transcription_script=excluded.transcription_script,
                       transcription_conf=excluded.transcription_conf,
                       transcribed_at=excluded.transcribed_at,
                       image_path=COALESCE(pages.image_path, excluded.image_path),
                       kind=COALESCE(pages.kind, excluded.kind)""",
                (mid, page_no, image_path, "prose",
                 text, scripts, conf, now()))
            wrote += 1
        con.commit()
    finally:
        con.close()

    click.echo(f"  ✓ ingested {wrote} page(s) into catalog.db pages "
               f"(manuscript {mid})"
               + (f", {skipped} without transcriptions skipped" if skipped else ""))
    click.echo("  wiki search will pick these up on its next index rebuild.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Bridge the transcription pipeline to the crawler's catalog."""


@cli.command()
@click.argument("msid", type=int)
@click.argument("outdir", type=click.Path(file_okay=False, path_type=Path),
                required=False)
@click.option("--store", default=CONTENT_STORE_DEFAULT, show_default=True,
              help="Content-addressed image store root.")
def stage(msid, outdir, store):
    """Pull manuscript MSID's images from the store into a working folder."""
    sp = Path(store)
    if not sp.is_dir():
        raise click.ClickException(
            f"content store not found at {sp} — is the external drive mounted?")
    do_stage(msid, outdir, sp)


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False,
                                           path_type=Path))
def ingest(folder):
    """Write FOLDER's transcriptions back into catalog.db pages."""
    do_ingest(folder)


if __name__ == "__main__":
    cli()
