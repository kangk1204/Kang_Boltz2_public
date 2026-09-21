"""Runtime validation helpers shared by the Boltz-2 shell workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_MODEL_RE = re.compile(r"^confidence_(?P<stem>.+_model_(?P<idx>\d+))\.json$")
_REQUIRED_CONFIDENCE_FIELDS = ("confidence_score", "ptm", "iptm")


def file_sha256(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_provenance(ipsae_script: Path | str | None = None) -> dict[str, Any]:
    """Identify the actual scoring implementation, not just a mutable version label."""
    scripts = Path(__file__).resolve().parent
    sources = {name: file_sha256(scripts / name) for name in
               ("analyze.py", "metrics.py", "figures.py", "msa_cache.py", "runtime_state.py")}
    vendor = Path(ipsae_script) if ipsae_script else scripts / "vendor" / "ipsae_official.py"
    sources["ipsae_script"] = file_sha256(vendor) if vendor.is_file() else None
    packages = {}
    for name in ("numpy", "gemmi", "DockQ", "anarci"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"schema_version": 2, "sources": sources, "packages": packages}


def msa_resource_hashes(yaml_path: Path | str, msa_mode: str, cache_dir: Path | str) -> dict:
    """Hash effective local MSA inputs, including cache files before YAML rewriting."""
    from msa_cache import DEFAULT_URL, cache_path, looks_like_msa_path, resolve_user_msa

    source = Path(yaml_path).resolve()
    data = yaml.safe_load(source.read_text())
    resources = {}
    for entry in data.get("sequences", []):
        spec = entry.get("protein")
        if not spec or spec.get("msa") == "empty":
            continue
        value = spec.get("msa")
        resource = resolve_user_msa(value, source.parent)
        if resource is None and looks_like_msa_path(value):
            raise ValueError(f"MSA file not found: {value}")
        if resource is None and msa_mode == "cache":
            resource = cache_path(Path(cache_dir), spec["sequence"], DEFAULT_URL, "greedy")
        if resource is not None:
            resource = resource.resolve()
            resources[str(resource)] = file_sha256(resource) if resource.is_file() else None
    return resources


def job_fingerprint(prediction_fingerprint: str, metadata_path: str = "", reference: str = "",
                    settings: dict[str, Any] | None = None) -> str:
    """Whole-job completion key; analysis/report changes need not invalidate predictions."""
    root = Path(__file__).resolve().parents[1]
    files = ("run.sh", "scripts/prepare_input.py", "scripts/prepare_batch.py",
             "scripts/make_report.py", "scripts/batch_report.py", "assets/report.js",
             "assets/molstar.js", "assets/molstar.css")
    payload = {
        "schema_version": 2,
        "prediction": prediction_fingerprint,
        "metadata": file_sha256(metadata_path) if metadata_path else None,
        "reference": file_sha256(reference) if reference else None,
        "settings": settings or {},
        "analysis": analysis_provenance(),
        "report_sources": {name: file_sha256(root / name) for name in files},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def runtime_provenance(boltz_executable: Path | str, kernel_mode: str) -> dict[str, Any]:
    """Return stable fields that can change prediction output compatibility."""

    executable = Path(boltz_executable)
    if not executable.is_file():
        raise ValueError(f"Boltz executable does not exist: {executable}")

    def package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "boltz_version": package_version("boltz"),
        "torch_version": package_version("torch"),
        "boltz_executable_sha256": file_sha256(executable),
        "kernel_mode": str(kernel_mode),
    }


def validate_run_name(name: str) -> str:
    """Return a safe run name or raise ValueError.

    Human-readable names, including Korean, are allowed. Path syntax is not.
    """

    if name is None:
        raise ValueError("run name is required")
    clean = str(name).strip()
    if not clean:
        raise ValueError("run name is empty")
    if clean in {".", ".."}:
        raise ValueError("run name cannot be '.' or '..'")
    if "\x00" in clean or any(ord(ch) < 32 for ch in clean):
        raise ValueError("run name contains a control character")
    p = Path(clean)
    if p.is_absolute() or "/" in clean or "\\" in clean:
        raise ValueError("run name cannot contain a path separator")
    if any(part in {".", ".."} for part in p.parts):
        raise ValueError("run name cannot contain path traversal")
    return clean


def confined_child(root: Path | str, name: str) -> Path:
    """Resolve root/name and ensure the result remains under root."""

    safe = validate_run_name(name)
    root_p = Path(root).resolve()
    child = (root_p / safe).resolve(strict=False)
    try:
        child.relative_to(root_p)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {child}") from exc
    return child


def ensure_path_within(path: Path | str, root: Path | str, label: str = "path") -> Path:
    """Resolve an arbitrary path and ensure it stays under root."""

    root_p = Path(root).resolve()
    child = Path(path).resolve(strict=False)
    try:
        child.relative_to(root_p)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {root_p}: {child}") from exc
    return child


def _finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} is not finite")
    elif value is None or isinstance(value, (int, str, bool)):
        return
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _finite_json(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{path}.{key}")
    else:
        raise ValueError(f"{path} has unsupported JSON value type {type(value).__name__}")


def _load_vector_npz(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"missing {path.name}")
    try:
        with np.load(path) as npz:
            arr = npz[key]
    except Exception as exc:
        raise ValueError(f"cannot read {path.name}:{key}: {exc}") from exc
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path.name}:{key} contains non-finite values")
    return arr


def _required_finite_number(data: dict[str, Any], key: str, source: str) -> None:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{source}.{key} must be a finite number")


def _validate_structure(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"missing readable structure for {path.stem} (.cif or .pdb)")
    try:
        import gemmi

        structure = gemmi.read_structure(str(path))
        n_models = len(structure)
        n_atoms = sum(1 for model in structure for chain in model for residue in chain for _atom in residue)
    except Exception as exc:
        raise ValueError(f"cannot parse structure {path.name}: {exc}") from exc
    if n_models < 1 or n_atoms < 1:
        raise ValueError(f"structure {path.name} contains no atoms")


def validate_prediction_outputs(pred_dir: Path | str, expected_samples: int | None = None) -> list[dict[str, Any]]:
    """Validate Boltz prediction files and return model records.

    Checks the requested model count, contiguous model ids, readable confidence
    JSON, matching CIF/PDB, square finite PAE, and finite pLDDT length matching
    the PAE token dimension. It intentionally does not assume all tokens are
    proteins; ligand-containing outputs can still pass if Boltz wrote matching
    token arrays.
    """

    pred = Path(pred_dir)
    if not pred.is_dir():
        raise ValueError(f"prediction directory does not exist: {pred}")

    records = []
    for conf in sorted(pred.glob("confidence_*_model_*.json")):
        m = _MODEL_RE.match(conf.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        stem = m.group("stem")
        records.append((idx, stem, conf))

    if not records:
        raise ValueError(f"no confidence_*_model_*.json files in {pred}")
    records.sort(key=lambda x: x[0])
    indexes = [r[0] for r in records]
    if indexes != list(range(len(indexes))):
        raise ValueError(f"model ids must be contiguous from 0, found {indexes}")
    if expected_samples is not None:
        if expected_samples < 1:
            raise ValueError("expected_samples must be >= 1")
        if len(records) != expected_samples:
            raise ValueError(f"expected {expected_samples} model(s), found {len(records)}")

    checked: list[dict[str, Any]] = []
    for idx, stem, conf in records:
        try:
            data = json.loads(conf.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"cannot read {conf.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{conf.name} must contain a JSON object")
        _finite_json(data, conf.name)
        for key in _REQUIRED_CONFIDENCE_FIELDS:
            _required_finite_number(data, key, conf.name)

        structure = pred / f"{stem}.cif"
        if not structure.exists():
            pdb = pred / f"{stem}.pdb"
            structure = pdb if pdb.exists() else structure
        _validate_structure(structure)

        pae = _load_vector_npz(pred / f"pae_{stem}.npz", "pae")
        plddt = _load_vector_npz(pred / f"plddt_{stem}.npz", "plddt")
        if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
            raise ValueError(f"pae_{stem}.npz:pae must be a square matrix, found {pae.shape}")
        if pae.shape[0] < 1:
            raise ValueError(f"pae_{stem}.npz:pae must contain at least one token")
        if plddt.ndim != 1:
            raise ValueError(f"plddt_{stem}.npz:plddt must be a vector, found {plddt.shape}")
        if plddt.shape[0] != pae.shape[0]:
            raise ValueError(
                f"plddt length {plddt.shape[0]} does not match PAE dimension {pae.shape[0]} for {stem}"
            )
        checked.append(
            {
                "index": idx,
                "stem": stem,
                "confidence": str(conf),
                "structure": str(structure),
                "pae": str(pred / f"pae_{stem}.npz"),
                "plddt": str(pred / f"plddt_{stem}.npz"),
                "tokens": int(pae.shape[0]),
            }
        )
    return checked


def _cmd_validate_name(args: argparse.Namespace) -> int:
    try:
        if args.root:
            print(confined_child(args.root, args.name))
        else:
            print(validate_run_name(args.name))
    except ValueError as exc:
        print(f"[runtime] {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_validate_predictions(args: argparse.Namespace) -> int:
    try:
        models = validate_prediction_outputs(args.pred_dir, args.expected_samples)
    except ValueError as exc:
        print(f"[runtime] {exc}", file=sys.stderr)
        return 2
    print(f"OK {len(models)} model(s)")
    return 0


def _cmd_runtime_provenance(args: argparse.Namespace) -> int:
    try:
        value = runtime_provenance(args.boltz_executable, args.kernel_mode)
    except (OSError, ValueError) as exc:
        print(f"[runtime] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


def _cmd_verify_sha256(args: argparse.Namespace) -> int:
    try:
        actual = file_sha256(args.path)
    except OSError as exc:
        print(f"[runtime] cannot hash {args.path}: {exc}", file=sys.stderr)
        return 2
    if actual.lower() != args.expected.lower():
        print(
            f"[runtime] SHA-256 mismatch for {args.path}: expected {args.expected}, got {actual}",
            file=sys.stderr,
        )
        return 2
    print(actual)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Boltz-2 runtime state checks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_name = sub.add_parser("validate-name")
    p_name.add_argument("name")
    p_name.add_argument("--root", type=Path, default=None)
    p_name.set_defaults(func=_cmd_validate_name)

    p_pred = sub.add_parser("validate-predictions")
    p_pred.add_argument("pred_dir", type=Path)
    p_pred.add_argument("--expected-samples", type=int, default=None)
    p_pred.set_defaults(func=_cmd_validate_predictions)

    p_prov = sub.add_parser("runtime-provenance")
    p_prov.add_argument("boltz_executable", type=Path)
    p_prov.add_argument("--kernel-mode", required=True)
    p_prov.set_defaults(func=_cmd_runtime_provenance)

    p_hash = sub.add_parser("verify-sha256")
    p_hash.add_argument("path", type=Path)
    p_hash.add_argument("expected")
    p_hash.set_defaults(func=_cmd_verify_sha256)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
