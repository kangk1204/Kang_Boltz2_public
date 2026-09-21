#!/usr/bin/env python
"""Build a batch of Boltz-2 jobs from a TSV/CSV table.

Table columns (header required, case-insensitive):
    name      job name (a-z, A-Z, 0-9, . _ -)
    antigen   FASTA path OR raw sequence. Several chains: separate with ';'
    nanobody  FASTA path OR raw sequence (single VHH)
    reference (optional) reference complex (cif/pdb) used for DockQ
    hotspot   (optional) 항원 epitope 잔기 (예: 45,67,101-103) -> pocket 제약
    notes     (optional) free text shown in the batch report

Example
-------
name        antigen                 nanobody          reference
nb01_RBD    inputs/RBD.fasta        inputs/nb01.fasta refs/RBD_nb01.cif
nb02_RBD    inputs/RBD.fasta        QVQLVESGGG...     -
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import re
from pathlib import Path

import prepare_input as P
from runtime_state import ensure_path_within, file_sha256, validate_run_name


def manifest_reusable(manifest_path: Path, batch_path: Path, msa: str) -> bool:
    """Fail closed for legacy, missing, altered inputs or stale generated jobs."""
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (m.get("schema_version") != 2 or m.get("errors") or not m.get("jobs")
                or m.get("batch_file") != str(batch_path.resolve()) or m.get("msa") != msa
                or m.get("batch_sha") != file_sha256(batch_path)):
            return False
        builder = {name: file_sha256(Path(__file__).parent / name)
                   for name in ("prepare_batch.py", "prepare_input.py", "runtime_state.py")}
        if m.get("builder_sources") != builder:
            return False
        for job in m["jobs"]:
            hashes = job.get("file_hashes") or {}
            if not hashes or any(str(Path(job[key]).resolve()) not in hashes for key in ("yaml", "meta")):
                return False
            if any(not Path(p).is_file() or file_sha256(p) != digest for p, digest in hashes.items()):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def resolve_reference(value: str, base_dirs) -> str | None:
    """A supplied reference must resolve to a readable file, never an inline sequence."""
    value = (value or "").strip()
    if not value or value == "-":
        return None
    for base in base_dirs:
        candidate = (base / Path(value).expanduser()).resolve()
        if candidate.is_file():
            file_sha256(candidate)
            return str(candidate)
    raise ValueError(f"reference file not found: {value}")


def resolve_entry(value: str, base_dirs):
    """Return ('file', Path) for a FASTA path or ('seq', str) for a raw sequence."""
    value = (value or "").strip()
    if not value or value == "-":
        return None
    candidate = value
    if "\n" not in value and len(value) < 400:
        # 공백이 있어도 실제 파일이면 경로로 인정한다 (R13)
        for base in base_dirs:
            p = (base / candidate).expanduser()
            if p.exists():
                return ("file", p)
        p = Path(candidate).expanduser()
        if p.exists():
            return ("file", p)
    seq = re.sub(r"\s+", "", value).upper()
    bad = sorted(set(seq) - P.VALID_AA)
    if bad:
        raise ValueError(f"'{value[:40]}...' is neither a readable FASTA path nor a valid "
                         f"sequence (bad characters: {bad})")
    # H-09: 단어(true/false/lysozyme 등)를 조용히 펩타이드로 받아들이지 않도록 최소 길이 요구
    if len(seq) < 20:
        raise ValueError(
            f"'{value[:40]}' 을 파일 경로로 찾지 못했고 서열로 보기에도 짧습니다(길이 {len(seq)}). "
            "FASTA 경로를 확인하거나 20잔기 이상 서열을 넣으세요")
    print(f"[batch] 경고: '{value[:40]}' 을 서열로 해석했습니다 (길이 {len(seq)})")
    return ("seq", seq)


def records_from(entry, base_dirs, label):
    if entry is None:
        raise ValueError(f"{label}: empty")
    kind, val = entry
    if kind == "file":
        return P.read_fasta(val), val
    return [{"header": f"inline {label}", "sequence": val}], None


def main():
    ap = argparse.ArgumentParser(description="batch TSV -> Boltz-2 jobs")
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--name", default=None, help="batch name (default: batch file stem)")
    ap.add_argument("--outdir", type=Path, default=Path("inputs/generated"))
    ap.add_argument("--msa", default="server", choices=["server", "empty", "cache"])
    args = ap.parse_args()

    batch_path = args.batch.expanduser()
    if not batch_path.exists():
        raise SystemExit(f"batch file not found: {batch_path}")
    try:
        batch_name = validate_run_name(args.name or batch_path.stem)
    except ValueError as exc:
        raise SystemExit(f"batch name 오류: {exc}") from None
    base_dirs = [batch_path.parent.resolve(), Path.cwd().resolve()]

    text = batch_path.read_text(encoding="utf-8-sig")   # M-19: BOM 허용
    if not text.strip():
        raise SystemExit(f"batch 파일이 비어 있습니다: {batch_path}")
    try:
        dialect = csv.Sniffer().sniff(text[:2000], delimiters="\t,;")
    except csv.Error:
        # 뒤쪽 선택 컬럼을 생략한 행이 섞이면 Sniffer 가 실패한다 -> 탭/쉼표 순으로 재시도
        dialect = None
        for cand in ("\t", ",", ";"):
            if cand in text.splitlines()[0]:
                dialect = csv.excel_tab if cand == "\t" else csv.excel
                if cand == ";":
                    dialect = csv.excel
                    dialect.delimiter = cand
                break
        if dialect is None:
            raise SystemExit("batch 파일의 구분자(탭/쉼표)를 판별하지 못했습니다") from None
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    field_map = {(f or "").strip().lower(): f for f in (reader.fieldnames or [])}
    for required in ("name", "antigen", "nanobody"):
        if required not in field_map:
            raise SystemExit(f"batch file needs a '{required}' column; found {list(field_map)}")

    root_outdir = args.outdir.expanduser()
    outdir = root_outdir / f"batch_{batch_name}"
    try:
        outdir = ensure_path_within(outdir, root_outdir, label="batch output path")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    outdir.mkdir(parents=True, exist_ok=True)
    jobs, seen, errors = [], set(), []
    for i, row in enumerate(reader, start=1):
        def cell(col, row=row):
            key = field_map.get(col)
            return (row.get(key) or "").strip() if key else ""

        name = cell("name") or f"job{i:02d}"
        try:
            name = validate_run_name(name)
        except ValueError as exc:
            errors.append(f"row {i}: invalid job name '{name}' ({exc})")
            continue
        if name in seen:
            errors.append(f"row {i}: duplicate job name '{name}'")
            continue
        seen.add(name)
        try:
            antigen_entries = [resolve_entry(v, base_dirs) for v in cell("antigen").split(";")]
            antigen_entries = [e for e in antigen_entries if e]
            if not antigen_entries:
                raise ValueError("antigen: empty")
            antigen_records, antigen_src = [], []
            for e in antigen_entries:
                recs, src = records_from(e, base_dirs, "antigen")
                antigen_records += recs
                antigen_src.append(src)
            nb_entry = resolve_entry(cell("nanobody"), base_dirs)
            nb_records, nb_src = records_from(nb_entry, base_dirs, "nanobody")
            if len(nb_records) > 1:
                raise ValueError("nanobody must be a single sequence")
            reference = resolve_reference(cell("reference"), base_dirs)
            # "-" 는 표에서 '없음'을 뜻하는 관례(reference 와 동일). hotspot 에서도 빈 값으로 취급한다.
            hotspot = cell("hotspot") or None
            if hotspot == "-":
                hotspot = None
            yaml_text, meta = P.build_job(antigen_records, nb_records, name, args.msa,
                                          antigen_src[0] if antigen_src else None, nb_src,
                                          hotspot_spec=hotspot)
            meta["batch"] = batch_name
            meta["reference"] = reference
            meta["notes"] = cell("notes")
            yaml_path = ensure_path_within(outdir / f"{name}.yaml", outdir, label="job YAML path")
            meta_path = ensure_path_within(outdir / f"{name}.meta.json", outdir, label="job meta path")
            yaml_path.write_text(yaml_text, encoding="utf-8")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            jobs.append({"name": name, "yaml": str(yaml_path), "meta": str(meta_path),
                         "reference": reference, "notes": meta["notes"],
                         "hotspot": meta.get("hotspot"),
                         "antigen_files": [str(s) for s in antigen_src if s],
                         "nanobody_file": str(nb_src) if nb_src else None,
                         "file_hashes": {str(Path(p).resolve()): file_sha256(p) for p in
                                         [yaml_path, meta_path, *antigen_src, nb_src, reference] if p},
                         "n_target_chains": len(antigen_records),
                         "nanobody_len": len(nb_records[0]["sequence"]),
                         "target_lens": [len(r["sequence"]) for r in antigen_records]})
            print(f"[batch] {name}: antigen {[len(r['sequence']) for r in antigen_records]} aa, "
                  f"nanobody {len(nb_records[0]['sequence'])} aa"
                  + (f", ref={Path(reference).name}" if reference else ""))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"row {i} ({name}): {exc}")

    manifest = {
        "schema_version": 2,
        "batch": batch_name,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_file": str(batch_path.resolve()),          # H-06: 절대 경로로 비교
        "batch_sha": file_sha256(batch_path),
        "builder_sources": {name: file_sha256(Path(__file__).parent / name)
                            for name in ("prepare_batch.py", "prepare_input.py", "runtime_state.py")},
        "outdir": str(outdir),
        "msa": args.msa,
        "jobs": jobs,
        "errors": errors,
    }
    manifest_path = outdir / "manifest.json"
    if errors:
        # H-06: 오류가 있으면 정식 manifest 를 덮어쓰지 않는다 (다음 실행에서 오류 행이 빠진 채
        # 조용히 진행하는 것을 방지)
        failed_path = outdir / "manifest.failed.json"
        failed_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n[batch] 오류 {len(errors)}건 -> {failed_path} (정식 manifest 는 갱신하지 않았습니다)")
        for e in errors:
            print("  -", e)
        return 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[batch] {len(jobs)} job(s) prepared -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
