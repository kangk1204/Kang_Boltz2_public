#!/usr/bin/env python
"""Per-chain MSA cache.

같은 항원을 여러 나노바디에 반복해서 쓰는 배치/재실행에서 MSA 서버 대기를 줄인다.
Boltz 자체 MSA 파이프라인(boltz.main.compute_msa)을 그대로 호출하므로 CSV 형식은
boltz 가 쓰는 것과 동일하다.

주의(H-10): 캐시는 **서열 하나씩** 받으므로 서버 모드와 달리 체인 간 페어링(paired) 행이
없다. 이형복합체에서 서버 모드와 MSA 구성이 달라질 수 있다(실측: 서버 6,281행 중 719행이
페어링, 캐시는 0행). 페어링이 중요한 입력은 `--msa server` 를 쓰라.
캐시 키에는 서버 url·pairing 전략·서열이 모두 들어간다.

  python scripts/msa_cache.py --yaml inputs/generated/job.yaml --cache-dir msa_cache
    -> job.cached.yaml 생성 (각 protein 체인에 msa: <cache csv> 추가)
  python scripts/msa_cache.py --stats --cache-dir msa_cache
  python scripts/msa_cache.py --yaml job.yaml --check     # 다운로드 없이 캐시 상태만 확인
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import yaml

DEFAULT_URL = "https://api.colabfold.com"


def seq_hash(sequence: str, url: str = "", pairing: str = "") -> str:
    """H-10: 서버 주소/페어링 전략까지 키에 포함한다 (같은 서열이라도 다른 DB/전략 구분)."""
    key = f"{url}|{pairing}|{sequence.strip().upper()}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def cache_path(cache_dir: Path, sequence: str, url: str = "", pairing: str = "") -> Path:
    return cache_dir / f"{seq_hash(sequence, url, pairing)}.csv"


def read_msa_summary(path: Path, query: str | None = None):
    """CSV 에서 (전체 행 수, 유효 여부) 반환.

    스키마(key,sequence)와 비어있지 않은 행을 확인하고, query 를 주면
    gap 을 제거한 서열이 MSA 안에 실제로 있는지까지 확인한다(R06).
    """
    try:
        with path.open() as fh:
            rows = list(csv.reader(fh))
    except Exception:  # noqa: BLE001 - 손상된 캐시 파일은 '유효하지 않음'으로 처리
        return 0, False
    if not rows or [c.strip().lower() for c in rows[0][:2]] != ["key", "sequence"]:
        return 0, False
    body = [r for r in rows[1:] if len(r) >= 2 and r[1].strip()]
    if not body:
        return 0, False
    if query:
        q = query.strip().upper()
        if not any(r[1].replace("-", "").strip().upper() == q for r in body):
            return len(body), False
    return len(body), True


def fetch_msa(sequence: str, dest: Path, url: str = DEFAULT_URL, pairing: str = "greedy",
              log=print, wait_lock_s: int = 600) -> Path:
    """boltz 내부 MSA 파이프라인으로 단일 서열 MSA를 받아 dest 에 저장.

    다른 프로세스가 같은 서열을 받는 중이면(lock 파일) 기다렸다가 결과를 재사용한다.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        _n, valid = read_msa_summary(dest, query=sequence)
        if valid:
            return dest
        log(f"  [경고] 캐시가 손상되었거나 서열과 불일치 -> 다시 받습니다: {dest.name}")
        dest.unlink()

    lock = dest.with_suffix(".lock")
    token = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"   # RR11: 소유자 토큰
    acquired = False
    waited = 0
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # 원자적 획득
            os.write(fd, token.encode())
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            # stale lock 회수: 30분 이상 + 소유자 프로세스가 살아있지 않을 때만
            try:
                content = lock.read_text().strip()
                pid = int(content.split(":")[0]) if content.split(":")[0].isdigit() else None
                alive = None   # None = 확인 불가(권한 없음) -> 보수적으로 회수하지 않음
                if pid:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        alive = None
                    except OSError:
                        alive = None
                if time.time() - lock.stat().st_mtime > 1800 and alive is False:
                    log(f"  [경고] 오래된 lock 회수 (owner pid {pid} 없음): {lock.name}")
                    lock.unlink()
                    continue
            except (FileNotFoundError, ValueError):
                continue
            if waited == 0:
                log(f"  다른 프로세스가 같은 MSA 를 받는 중... 대기 ({dest.name})")
            if waited >= wait_lock_s:
                raise SystemExit(
                    f"MSA lock 대기 시간 초과 ({wait_lock_s}s): {lock}\n"
                    "  다른 프로세스가 멈춘 것 같으면 lock 파일을 지우고 다시 실행하세요.") from None
            time.sleep(3)
            waited += 3
            if dest.exists() and dest.stat().st_size > 0:
                _n, valid = read_msa_summary(dest, query=sequence)
                if valid:
                    return dest
    try:
        return _fetch_msa_locked(sequence, dest, url, pairing, log)
    finally:
        if acquired:
            try:
                if lock.read_text().strip() == token:   # RR11: 내 lock 일 때만 해제
                    lock.unlink()
            except FileNotFoundError:
                pass


def _fetch_msa_locked(sequence: str, dest: Path, url: str, pairing: str, log) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        _n, valid = read_msa_summary(dest, query=sequence)
        if valid:
            return dest
    from boltz.main import compute_msa  # boltz 내부 함수 (서버 경로와 동일 코드)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        log(f"  MSA 서버 요청: {sequence[:24]}... ({len(sequence)} aa)")
        compute_msa(
            data={"cache_0": sequence},
            target_id="cache",
            msa_dir=tmp,
            msa_server_url=url,
            msa_pairing_strategy=pairing,
        )
        cands = sorted(tmp.glob("cache_0.csv")) + sorted(tmp.glob("*.csv"))
        if not cands:
            raise RuntimeError("MSA 서버 응답에서 CSV 를 찾지 못했습니다")
        tmp_dest = dest.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")  # RR11: writer 별 고유 임시 파일
        shutil.copy2(cands[0], tmp_dest)
        os.replace(tmp_dest, dest)  # 동시 실행 시에도 부분 파일이 읽히지 않도록
    n, valid = read_msa_summary(dest, query=sequence)
    if not valid:
        raise RuntimeError(f"받은 MSA 가 유효하지 않습니다 (query 미포함): {dest}")
    log(f"  -> 캐시 저장: {dest.name} ({n} 서열)")
    return dest


def resolve_user_msa(value, base_dir: Path | None = None) -> Path | None:
    """Resolve an explicit MSA path, relative to its YAML when available."""

    if not isinstance(value, str) or value.strip() in ("", "empty", "0"):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve() if path.exists() else None


def is_user_msa(value, base_dir: Path | None = None) -> bool:
    """사용자가 직접 넣은 MSA 경로인지 (H-10: 덮어쓰지 않는다)."""

    return resolve_user_msa(value, base_dir) is not None


def looks_like_msa_path(value) -> bool:
    """경로처럼 보이는 MSA 지정인지 (M-05: 존재하지 않으면 조용히 캐시로 대체하지 않는다)."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    if v in ("", "empty", "0"):
        return False
    return bool(Path(v).expanduser().suffix) or "/" in v or "\\" in v


def rewrite_yaml(yaml_path: Path, cache_dir: Path, url: str, pairing: str,
                 check_only: bool, out_path: Path | None = None, log=print,
                 lock_timeout: int = 600) -> tuple[Path, list]:
    """각 protein 체인의 서열에 대해 캐시를 채우고 원본 YAML 구조를 유지한 채 새 파일로 저장."""
    data = yaml.safe_load(yaml_path.read_text())
    results = []
    seqs = []
    for entry in data.get("sequences", []):
        for kind, spec in entry.items():
            if kind != "protein":
                continue
            ids = spec["id"]
            if isinstance(ids, list):
                ids = ids[0]
            seqs.append((ids, spec["sequence"], spec.get("msa")))

    for chain_id, seq, existing in seqs:
        if existing == "empty":
            results.append((chain_id, "empty (건너뜀)", 0))
            continue
        if is_user_msa(existing, yaml_path.parent):
            results.append((chain_id, "사용자 MSA 보존", 0))
            continue
        if looks_like_msa_path(existing):
            # M-05: 경로처럼 보이는데 파일이 없다 -> 오타/미생성. 캐시로 조용히 대체하지 않는다.
            raise SystemExit(
                f"[{chain_id}] 지정한 MSA 파일을 찾을 수 없습니다: '{existing}'\n"
                "  경로를 확인하거나, 캐시를 쓰려면 해당 msa 항목을 지우세요.")
        dest = cache_path(cache_dir, seq, url, pairing)
        if dest.exists():
            n, valid = read_msa_summary(dest, query=seq)
            if valid:
                results.append((chain_id, "캐시 사용", n))
                continue
            if check_only:
                results.append((chain_id, "캐시 손상/불일치", n))
                continue
            # 실제 실행 모드에서는 fetch_msa 가 검증 후 다시 받는다
            results.append((chain_id, "캐시 손상 -> 재다운로드", n))
            fetch_msa(seq, dest, url=url, pairing=pairing, log=log, wait_lock_s=lock_timeout)
            n2, _ = read_msa_summary(dest, query=seq)
            results[-1] = (chain_id, "새로 받음", n2)
            continue
        if check_only:
            results.append((chain_id, "캐시 없음", 0))
            continue
        fetch_msa(seq, dest, url=url, pairing=pairing, log=log, wait_lock_s=lock_timeout)
        n, _ = read_msa_summary(dest, query=seq)
        results.append((chain_id, "새로 받음", n))

    # YAML 재작성 (msa 키 추가)
    for entry in data.get("sequences", []):
        for kind, spec in entry.items():
            if kind != "protein":
                continue
            if spec.get("msa") == "empty":
                continue
            user_msa = resolve_user_msa(spec.get("msa"), yaml_path.parent)
            if user_msa is not None:
                spec["msa"] = str(user_msa)
                continue   # H-10: 사용자 MSA 는 절대 경로로 정규화해 그대로 보존한다
            dest = cache_path(cache_dir, spec["sequence"], url, pairing)
            spec["msa"] = str(dest)

    out_path = Path(out_path) if out_path else yaml_path.with_suffix(".cached.yaml")
    if not check_only:
        out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out_path, results


def main():
    ap = argparse.ArgumentParser(description="per-chain MSA cache")
    ap.add_argument("--yaml", type=Path, help="Boltz 입력 YAML")
    ap.add_argument("--cache-dir", type=Path, default=Path("msa_cache"))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--pairing", default="greedy", choices=["greedy", "complete"])
    ap.add_argument("--check", action="store_true",
                    help="다운로드/파일 생성 없이 캐시 상태만 확인")
    ap.add_argument("--out", type=Path, default=None,
                    help="캐시를 채운 YAML 출력 경로 (기본: <입력>.cached.yaml)")
    ap.add_argument("--stats", action="store_true", help="캐시 통계 출력")
    ap.add_argument("--lock-timeout", type=int, default=600,
                    help="다른 프로세스의 MSA 다운로드를 기다리는 최대 시간(초, 기본 600)")
    args = ap.parse_args()

    if args.stats:
        files = sorted(args.cache_dir.glob("*.csv")) if args.cache_dir.exists() else []
        total = sum(f.stat().st_size for f in files)
        print(f"MSA 캐시: {args.cache_dir} | {len(files)}개 | {total / 1e6:.1f} MB")
        for f in files[:20]:
            n, _ = read_msa_summary(f)
            print(f"  {f.name}  {n:6d} 서열  {f.stat().st_size / 1e3:8.1f} KB")
        return 0

    if not args.yaml:
        ap.error("--yaml 또는 --stats 가 필요합니다")
    yaml_path = args.yaml
    if not yaml_path.exists():
        raise SystemExit(f"YAML 이 없습니다: {yaml_path}")

    print(f"[msa-cache] {yaml_path}")
    out_path, results = rewrite_yaml(yaml_path, args.cache_dir, args.url, args.pairing,
                                     check_only=args.check, out_path=args.out,
                                     lock_timeout=args.lock_timeout)
    for chain_id, status, n in results:
        print(f"  체인 {chain_id}: {status}" + (f" ({n} 서열)" if n else ""))
    if args.check:
        missing = [r for r in results if r[1] == "캐시 없음"]
        print(f"[msa-cache] 캐시 누락 {len(missing)}개" + ("" if not missing else
              " -> --check 없이 다시 실행하면 받아옵니다"))
        print("[msa-cache] --check 모드: 파일을 생성하지 않았습니다")
        return 0
    print(f"[msa-cache] 완료 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
