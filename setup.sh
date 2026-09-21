#!/usr/bin/env bash
# Boltz-2 nanobody pipeline installer (existing env names are preserved).
#
#   bash setup.sh                 # env + boltz + CDR/DockQ 도구(ANARCI·HMMER·DockQ) + molstar 자산
#   bash setup.sh --verify        # 설치 후 two-chain 예측·분석·리포트 스모크까지
#   bash setup.sh --no-hmmer|--no-anarci|--no-dockq   # 해당 도구만 생략
#   bash setup.sh --reuse-env     # 새 이름 대신 기존 conda env 갱신
#   ENV_NAME=boltz2 bash setup.sh # 기본 이름 지정 (중복 시 _2, _3, ...)
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-boltz2}"
# 이 파이프라인은 boltz 2.2.x 에서 검증되었다. 다른 버전을 쓰려면 BOLTZ_VERSION=... 로 지정.
BOLTZ_VERSION="${BOLTZ_VERSION:-2.2.1}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0+cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PY_VER="3.12"
VERIFY=0
WITH_HMMER=1    # CDR 주석에 필요 -> 기본 설치
WITH_DOCKQ=1    # 참조 구조 DockQ 계산 -> 기본 설치
WITH_ANARCI=1   # CDR 자동 주석/변이 생성 -> 기본 설치
REUSE_ENV=0
for a in "$@"; do
  case "$a" in
    --verify) VERIFY=1;;
    --with-hmmer) WITH_HMMER=1;;
    --no-hmmer) WITH_HMMER=0;;
    --no-dockq) WITH_DOCKQ=0;;
    --no-anarci) WITH_ANARCI=0;;
    --reuse-env) REUSE_ENV=1;;
    -h|--help) sed -n '2,8p' "$0"; exit 0;;
    *) echo "unknown option: $a"; exit 1;;
  esac
done

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. conda/mamba ---------------------------------------------------------
CONDA=""
for c in mamba micromamba conda; do command -v "$c" >/dev/null 2>&1 && { CONDA="$c"; break; }; done
[[ -n "$CONDA" ]] || die "conda/mamba 를 찾을 수 없습니다. Miniforge 설치 후 다시 실행하세요."
log "conda: $CONDA"

# --- 2. environment ---------------------------------------------------------
ENV_LIST="$("$CONDA" env list)" || die "conda env 목록을 읽지 못했습니다 ($CONDA env list)"
if [[ "$REUSE_ENV" != "1" ]]; then
  BASE_ENV_NAME="$ENV_NAME"
  ENV_SUFFIX=2
  while printf '%s\n' "$ENV_LIST" | awk -v n="$ENV_NAME" '$1==n{found=1} END{exit !found}'; do
    ENV_NAME="${BASE_ENV_NAME}_${ENV_SUFFIX}"
    ENV_SUFFIX=$((ENV_SUFFIX + 1))
  done
  if [[ "$ENV_NAME" != "$BASE_ENV_NAME" ]]; then
    log "기존 env '$BASE_ENV_NAME' 보존 -> 새 env '$ENV_NAME' 자동 선택"
  fi
fi
ENV_PREFIX="$(printf '%s\n' "$ENV_LIST" | awk -v n="$ENV_NAME" '$1==n{print $NF; exit}')"
if [[ -n "$ENV_PREFIX" ]]; then
  ok "conda env '$ENV_NAME' 재사용 - --reuse-env 로 패키지를 갱신합니다"
else
  log "conda env '$ENV_NAME' 생성 (python $PY_VER)"
  "$CONDA" create -y -n "$ENV_NAME" "python=$PY_VER" pip || die "env 생성 실패"
  ENV_LIST="$("$CONDA" env list)" || die "생성 후 conda env 목록을 읽지 못했습니다"
  ENV_PREFIX="$(printf '%s\n' "$ENV_LIST" | awk -v n="$ENV_NAME" '$1==n{print $NF; exit}')"
fi
[[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX" ]] || die "env '$ENV_NAME' 의 실제 경로를 확인하지 못했습니다"
PIP="$ENV_PREFIX/bin/pip"
# PyTorch CDN 등이 느린 네트워크에서도 설치가 끝나도록 재시도/타임아웃을 넉넉히 준다
export PIP_RETRIES="${PIP_RETRIES:-10}"
export PIP_TIMEOUT="${PIP_TIMEOUT:-60}"
PY="$ENV_PREFIX/bin/python"
[[ -x "$PIP" ]] || die "pip 을 찾을 수 없습니다: $PIP"
log "env 경로: $ENV_PREFIX"
CONSTRAINTS="$ROOT/requirements/boltz2-cu128-constraints.txt"
[[ -s "$CONSTRAINTS" ]] || die "constraints 파일이 없습니다: $CONSTRAINTS"

# --- 3. GPU / driver 확인 ----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
  warn "nvidia-smi 없음 - CPU 실행은 매우 느립니다 (GPU 권장)"
fi

# --- 4. torch (CUDA 12.8 빌드: RTX 50 시리즈/sm_120 지원) --------------------
# PyTorch 2.7 official release notes announce Blackwell support and CUDA 12.8 wheels;
# NVIDIA lists RTX 50-series Blackwell parts as compute capability 12.0.
log "torch ${TORCH_VERSION} 설치 (cu128 wheel, no fallback)"
"$PIP" install -q --upgrade pip
"$PIP" install -q --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" \
  "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL" \
  || die "torch==${TORCH_VERSION} 설치 실패. 일반 PyPI torch 로 자동 대체하지 않습니다."

# --- 5. boltz + 옵션 패키지 --------------------------------------------------
log "boltz 설치"
"$PIP" install -q -c "$CONSTRAINTS" "boltz==${BOLTZ_VERSION}" || die "boltz==${BOLTZ_VERSION} 설치 실패 (버전 변경: BOLTZ_VERSION=2.2.3 bash setup.sh)"
log "GPU 커널(cuequivariance) 설치 - 없으면 자동으로 --no_kernels 로 동작"
"$PIP" install -q -c "$CONSTRAINTS" cuequivariance-ops-torch-cu12 cuequivariance-torch \
  || warn "cuequivariance 설치 실패: 실행 시 자동으로 --no-kernels 사용"
# 설치 성공 여부와 무관하게, 이 torch 빌드에서 실제 사용 모듈이 import 되는지 확인한다.
# (예: torch 2.7 + cuequivariance 0.11 은 is_fx_tracing_symbolic_tracing 미존재로 실패)
if "$PY" -c "import torch, cuequivariance_torch, cuequivariance_ops_torch.attention_pair_bias" >/dev/null 2>&1; then
  ok "cuequivariance 커널 import 확인 ($("$PY" -c 'import torch; print(torch.__version__)'))"
else
  warn "이 torch 빌드($("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null))에서는 cuequivariance 커널을 쓸 수 없습니다."
  warn "예측은 자동으로 --no-kernels 로 실행됩니다 (결과는 유효, 속도만 느림). 켜려면 torch/cuequivariance 버전을 맞추세요."
fi
log "필수 리포트 도구 설치 (matplotlib)"
"$PIP" install -q -c "$CONSTRAINTS" "matplotlib>=3.8,<3.12" || die "matplotlib 설치 실패 (분석/그림 생성에 필수)"
log "CDR/DockQ 도구 설치 (ANARCI, DockQ)"
if [[ "$WITH_DOCKQ" == "1" ]]; then
  "$PIP" install -q -c "$CONSTRAINTS" DockQ || warn "DockQ 설치 실패 (뒤에서 재시도; 참조 구조 DockQ 계산에 필요)"
else
  warn "--no-dockq: DockQ 생략 (참조 구조 DockQ 계산 불가)"
fi
if [[ "$WITH_ANARCI" == "1" ]]; then
  "$PIP" install -q -c "$CONSTRAINTS" anarci || warn "ANARCI 설치 실패 (뒤에서 재시도; CDR 자동 주석/변이 생성에 필요)"
else
  warn "--no-anarci: ANARCI 생략 (CDR 자동 주석/변이 생성 불가)"
fi
# anarci/matplotlib 가 numpy>=2 를 끌어오면 boltz 가 깨지므로 되돌린다
"$PIP" install -q -c "$CONSTRAINTS" "numpy<2.0" "contourpy<1.4" || true

# --- 5b. 필수 런타임 모듈 검증 (설치 누락을 예측/분석 전에 잡는다) -----------
log "필수 런타임 모듈 확인"
probe_missing() {
  "$PY" - "$WITH_DOCKQ" "$WITH_ANARCI" <<'PYEOF'
import importlib
import sys
need = ["torch", "boltz", "numpy", "gemmi", "matplotlib", "yaml"]
if sys.argv[1] == "1":
    need.append("DockQ")
if sys.argv[2] == "1":
    need.append("anarci")
missing = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print(" ".join(missing))
PYEOF
}
missing_mods="$(probe_missing)"
if [[ -n "$missing_mods" ]]; then
  warn "누락된 필수 모듈: $missing_mods — 자동 재설치를 시도합니다"
  case " $missing_mods " in *" torch "*) "$PIP" install -q --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL" || true;; esac
  case " $missing_mods " in *" boltz "*) "$PIP" install -q -c "$CONSTRAINTS" "boltz==${BOLTZ_VERSION}" || true;; esac
  case " $missing_mods " in *" numpy "*) "$PIP" install -q -c "$CONSTRAINTS" "numpy<2.0" || true;; esac
  case " $missing_mods " in *" gemmi "*) "$PIP" install -q -c "$CONSTRAINTS" gemmi || true;; esac
  case " $missing_mods " in *" matplotlib "*) "$PIP" install -q -c "$CONSTRAINTS" "matplotlib>=3.8,<3.12" || true;; esac
  case " $missing_mods " in *" yaml "*) "$PIP" install -q -c "$CONSTRAINTS" pyyaml || true;; esac
  case " $missing_mods " in *" DockQ "*) "$PIP" install -q -c "$CONSTRAINTS" DockQ || true;; esac
  case " $missing_mods " in *" anarci "*) "$PIP" install -q -c "$CONSTRAINTS" anarci || true;; esac
  missing_mods="$(probe_missing)"
fi
[[ -z "$missing_mods" ]] || die "필수 모듈이 아직 없습니다: $missing_mods (네트워크 확인 후 ENV_NAME='$ENV_NAME' bash setup.sh --reuse-env --verify 재실행)"
need_list="torch/boltz/numpy/gemmi/matplotlib/yaml"
[[ "$WITH_DOCKQ" == "1" ]] && need_list="$need_list/DockQ"
[[ "$WITH_ANARCI" == "1" ]] && need_list="$need_list/anarci"
ok "필수 모듈 확인 완료 ($need_list)"

# --- 6. hmmscan (CDR 주석; 기본 설치, --no-hmmer 로 생략) --------------------
if [[ -x "$ENV_PREFIX/bin/hmmscan" ]]; then
  ok "hmmscan 이미 설치됨"
elif [[ "$WITH_HMMER" == "1" ]]; then
  log "hmmer 설치 (bioconda) — CDR 주석(ANARCI)에 필요"
  "$CONDA" install -y -n "$ENV_NAME" -c bioconda -c conda-forge hmmer || die "hmmer 설치 실패 (CDR 주석에 필요; 생략하려면 --no-hmmer)"
  [[ -x "$ENV_PREFIX/bin/hmmscan" ]] || die "$ENV_PREFIX/bin/hmmscan 이 설치되지 않았습니다 (생략: --no-hmmer)"
  "$PIP" install -q -c "$CONSTRAINTS" "numpy<2.0" || true
else
  warn "--no-hmmer: hmmscan 이 없어 CDR 자동 주석이 생략됩니다."
fi

# --- 7. Mol* viewer assets --------------------------------------------------
mkdir -p "$ROOT/assets"
log "이 환경에서 설치를 재시도하려면: ENV_NAME='$ENV_NAME' bash setup.sh --reuse-env --verify"
declare -A MOLSTAR_SHA256=(
  [molstar.js]="7fad5561c74bc900930fb57d6ab028d1aafdda82223a901bf932b1098e84f1f3"
  [molstar.css]="5b68ceb6d3642549b4e9b2c071e58e41b98a5350ae269180587b39da86925d55"
)
download_molstar_asset() {
  local name="$1" expected="$2" tmp
  tmp="$(mktemp "$ROOT/assets/.${name}.XXXXXX")" || return 1
  if ! curl --fail --silent --show-error --location -o "$tmp" \
      "https://cdn.jsdelivr.net/npm/molstar@5.11.0/build/viewer/$name"; then
    rm -f "$tmp"
    return 1
  fi
  if ! "$PY" "$ROOT/scripts/runtime_state.py" verify-sha256 "$tmp" "$expected" >/dev/null; then
    rm -f "$tmp"
    return 1
  fi
  if ! mv -f "$tmp" "$ROOT/assets/$name"; then
    rm -f "$tmp"
    return 1
  fi
}
for f in molstar.js molstar.css; do
  if [[ ! -s "$ROOT/assets/$f" ]]; then
    log "Mol* 자산 다운로드: $f"
    download_molstar_asset "$f" "${MOLSTAR_SHA256[$f]}" \
      || die "Mol* 자산 다운로드/무결성 검증 실패 ($f)"
  elif ! "$PY" "$ROOT/scripts/runtime_state.py" verify-sha256 \
      "$ROOT/assets/$f" "${MOLSTAR_SHA256[$f]}" >/dev/null; then
    die "Mol* 자산 SHA-256 불일치 ($f). 파일을 삭제한 뒤 setup.sh 를 다시 실행하세요."
  fi
done
[[ -s "$ROOT/assets/report.js" ]] || die "assets/report.js 가 없습니다 (저장소에서 누락됨)"
ok "Mol* 자산 준비 완료 (오프라인 동작)"

# --- 8. 확인 -----------------------------------------------------------------
EXPECTED_TORCH_VERSION="$TORCH_VERSION" EXPECTED_TORCH_CUDA="12.8" ENV_PREFIX="$ENV_PREFIX" "$PY" - <<'PYEOF'
import os
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

required = ["torch", "boltz", "numpy", "gemmi", "matplotlib"]
optional = ["DockQ", "anarci"]
missing = []
for m in required:
    try:
        __import__(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m} ({e})")
for m in optional:
    try:
        __import__(m)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 선택 패키지 {m} 없음 -> 해당 지표/기능만 생략됩니다 ({e})")
if missing:
    print("[FAIL] 필수 패키지 import 실패:", ", ".join(missing))
    sys.exit(1)
import torch, numpy
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | numpy {numpy.__version__}")
expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
expected_cuda = os.environ["EXPECTED_TORCH_CUDA"]
if torch.__version__ != expected_torch:
    print(f"[FAIL] torch version mismatch: expected {expected_torch}, got {torch.__version__}")
    sys.exit(1)
if torch.version.cuda != expected_cuda:
    print(f"[FAIL] torch CUDA mismatch: expected {expected_cuda}/cu128, got {torch.version.cuda}")
    sys.exit(1)
if torch.cuda.is_available():
    x = torch.ones((128, 128), device="cuda")
    y = (x @ x).sum().item()
    if y <= 0:
        print("[FAIL] CUDA matmul validation failed")
        sys.exit(1)
    print(f"cuda kernel smoke ok | capability={torch.cuda.get_device_capability(0)}")
versions = {}
for pkg in required + optional:
    try:
        versions[pkg] = metadata.version(pkg)
    except metadata.PackageNotFoundError:
        versions[pkg] = None
driver = None
try:
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=10,
    ).splitlines()[0].strip()
except Exception:
    driver = None
provenance = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": versions,
    "torch": {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "driver": driver,
    },
}
prov_path = Path(os.environ["ENV_PREFIX"]) / "boltz2_install_provenance.json"
prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"install provenance: {prov_path}")
PYEOF
"$PIP" check || die "pip check 실패: 의존성 메타데이터 충돌"

if [[ "$VERIFY" == "1" ]]; then
  log "검증 실행: two-chain prediction + analysis/report (가중치 다운로드 포함, 수 분 소요)"
  VERIFY_TMP="$(mktemp -d)"
  trap '[[ -n "${VERIFY_TMP:-}" ]] && rm -rf -- "$VERIFY_TMP"' EXIT
  cat > "$VERIFY_TMP/twochain.yaml" <<'EOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGC
      msa: empty
  - protein:
      id: B
      sequence: VQLQASGGGSVQAGGSLRLSCAASGYTIGPYCMGWFRQAPGKEREGVAAINMGGGITYYADSVKGRFTISQDNAKNTVYLLMNSLEPEDTAIYYCAADSTIYASYYECGHGLSTGGYGYDSWGQGTQVTVSS
      msa: empty
EOF
  verify_name="setup_verify_$(date +%Y%m%d_%H%M%S)"
  BOLTZ_ENV="$ENV_PREFIX" SAMPLES=1 STEPS=20 RECYCLES=1 "$ROOT/run.sh" \
    --yaml "$VERIFY_TMP/twochain.yaml" --name "$verify_name" --nanobody-chain B \
    --antigen-chain A --reference "$ROOT/examples/demo_reference_1MEL_AB.cif" \
    --msa empty --no-kernels || die "two-chain pipeline 검증 실패"
  [[ -s "$ROOT/outputs/$verify_name/analysis/results.json" ]] || die "검증 results.json 누락"
  [[ -s "$ROOT/outputs/$verify_name/report/index.html" ]] || die "검증 HTML report 누락"
  ok "two-chain pipeline 검증 성공: outputs/$verify_name"
  rm -rf -- "$VERIFY_TMP"
  VERIFY_TMP=""
  trap - EXIT
fi

# run.sh 가 이 env 를 자동으로 찾도록 포인터를 남긴다.
# (conda/miniforge/micromamba 에 boltz2 env 가 여러 개 있어도 방금 설치한 env 를 쓰게 함)
printf '%s\n' "$ENV_PREFIX" > "$ROOT/.boltz_env"

cat <<EOF

$(ok "설치 완료")
설치 환경: $ENV_NAME ($ENV_PREFIX)
다음 단계:
  1) 입력 준비: inputs/target.fasta (항원), inputs/nanobody.fasta (나노바디)
  2) 실행:      ./run.sh --name mynb01
  3) 리포트:    ./run.sh --serve outputs/mynb01/report
  4) 환경 점검: ./run.sh --doctor
  이 env 를 자동 사용하도록 $ROOT/.boltz_env 에 기록했습니다.
  (다른 env 를 쓰려면 BOLTZ_ENV=<경로> ./run.sh ... 로 덮어쓰세요.)
EOF
