"""Read candidate libraries for the existing Boltz batch pipeline."""

import re
from pathlib import Path

from prepare_input import read_fasta

# The first token is always the candidate ID. Pipe-separated FASTA accessions
# remain intact unless the pipe starts a metadata field.
_FIELDS = re.compile(r"(?:\s+|\|)(?=(?:binder|ref|hotspot|notes|target|antigen|reference)=)")


def library_rows(source: Path, layout: str, target: Path | None, binder_chain: str,
                 reference: str, hotspot: str):
    if layout == "binders" and (target is None or not target.is_file()):
        raise ValueError("--binders requires --target with an existing antigen FASTA")
    if layout == "complex" and target is not None:
        raise ValueError("--complex-fasta already includes the target; omit --target")
    shared_target = read_fasta(target) if target else None
    records = read_fasta(source, allow_chain_separator=layout == "complex")
    for i, record in enumerate(records, 1):
        pieces = _FIELDS.split(record["header"])
        title = pieces[0].split(maxsplit=1)
        if not title:
            raise ValueError(f"FASTA entry {i}: missing candidate ID")
        name = title[0]
        options = {}
        for field in pieces[1:]:
            key, value = field.split("=", 1)
            if key in options or not value.strip():
                raise ValueError(f"{name}: duplicate or empty {key}")
            options[key] = value.strip()
        if "target" in options or "antigen" in options:
            raise ValueError(f"{name}: use --batch with an antigen column for per-job targets")
        parts = record["sequence"].split(":")
        if layout == "complex":
            if len(parts) < 2 or any(not part for part in parts):
                raise ValueError(f"{name}: expected VHH:target with nonempty chains")
            index = int(options.get("binder", 0 if binder_chain == "first" else len(parts) - 1))
            if not 0 <= index < len(parts):
                raise ValueError(f"{name}: invalid binder index {index}")
            binder = parts.pop(index)
            antigens = [{"sequence": seq, "header": name} for seq in parts]
        else:
            if "binder" in options:
                raise ValueError(f"{name}: binder index applies only to complex FASTA")
            binder = parts[0]
            antigens = shared_target
        if "ref" in options and "reference" in options:
            raise ValueError(f"{name}: specify the reference once")
        ref = options.get("ref", options.get("reference", reference))
        yield {"name": name, "target_records": antigens, "nanobody": binder,
               "reference": ref, "hotspot": options.get("hotspot", hotspot),
               "notes": options.get("notes", title[1] if len(title) > 1 else "")}
