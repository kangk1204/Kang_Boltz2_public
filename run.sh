#!/usr/bin/env bash
# Boltz-2 nanobody prediction pipeline (실험 연구자용)
#   예측 -> 지표 분석(ipTM/ipSAE/DockQ/PAE) -> Mol* HTML 리포트 생성
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
ASSETS="$ROOT/assets"
OUT_ROOT="$ROOT/outputs"
GEN_ROOT="$ROOT/inputs/generated"

# ---------------------------------------------------------------------------
# 기본값 (환경변수로도 바꿀 수 있음)
# ---------------------------------------------------------------------------
SAMPLES="${SAMPLES:-3}"
SEED="${SEED:-42}"
STEPS="${STEPS:-200}"
RECYCLES="${RECYCLES:-3}"
WORKERS="${WORKERS:-4}"
RETRY_SLEEP="${RETRY_SLEEP:-15}"
PARALLEL_SAMPLES="${PARALLEL_SAMPLES:-1}"   # VRAM 절약 (16GB 카드 권장: 1~2)
CONCURRENCY="${CONCURRENCY:-1}"            # 배치 동시 job 수
DEVICES="${DEVICES:-1}"                    # boltz --devices (multi-GPU 단일 job)
GPUS="${GPUS:-}"                           # 배치에서 job 을 나눠 쓸 GPU 목록 (예: 0,1)
HOTSPOT="${HOTSPOT:-}"                     # 항원 epitope 잔기 (pocket 제약)
MSA_CACHE_DIR="${MSA_CACHE_DIR:-$ROOT/msa_cache}"  # MSA 캐시 폴더
MSA_MODE="${MSA_MODE:-server}"
MSA_SUBSAMPLE="${MSA_SUBSAMPLE:-0}"         # 0=전체 MSA, 양수=모델 내부 MSA 행 수 제한
OOM_RETRY="${OOM_RETRY:-1}"               # GPU OOM 시 메모리 절약 설정으로 1회 재시도
PORT="${PORT:-8765}"
ENV_DIR="${BOLTZ_ENV:-}"
GPU="${CUDA_VISIBLE_DEVICES:-}"
# CDR 변이 라이브러리 생성 (--make-cdr-library): env/conda 를 몰라도 쓰도록 run.sh 가 감싼다.
CDR_CDRS="${CDR_CDRS:-CDR1,CDR2,CDR3}"
CDR_N_MUTATIONS="${CDR_N_MUTATIONS:-2}"
CDR_N_VARIANTS="${CDR_N_VARIANTS:-12}"
CDR_MODE="${CDR_MODE:-random}"
CDR_SCHEME="${CDR_SCHEME:-imgt}"
CDR_PREFIX=""
CDR_OUTDIR=""
CDR_BATCH_OUT=""
CDR_EXHAUSTIVE=0
CDR_ALLOW_CYS=0
CDR_PARATOPE_FROM=""
CDR_PARATOPE_MIN_CONTACTS="1"
CDR_EXCLUDE_POSITIONS=""

MODE="single"
TARGET_ONLY=0
EXPLICIT_TARGET_ONLY=0
CHILD_PIDS=()
NO_EMBED=0
# M-08(정밀): 등록한 PID 는 백그라운드 작업 shell 이므로 TERM 만 보내면 실제 예측
# 자식(boltz/python)이 살아남는다. 자손을 재귀적으로 TERM -> KILL 한다.
kill_tree() {
  local pid="$1" sig="${2:-TERM}" k
  while read -r k; do
    [[ -n "$k" ]] && kill_tree "$k" "$sig"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill -"$sig" "$pid" 2>/dev/null || true
  return 0
}
cleanup_children() {
  local p
  for p in "${CHILD_PIDS[@]:-}"; do
    [[ -n "$p" ]] && kill_tree "$p" TERM
  done
  sleep 2
  for p in "${CHILD_PIDS[@]:-}"; do
    [[ -n "$p" ]] && kill_tree "$p" KILL
  done
}
trap 'echo; warn "중단 요청 -> 실행 중인 예측(자손 포함)을 종료합니다"; cleanup_children; exit 130' INT TERM
TARGET=""; NANOBODY=""; NAME=""; YAML=""; BATCH=""; JOBS_FILTER=""; FORCE=0
REFERENCE=""; DOCKQ_MAPPING=""
DOCKQ_MISMATCHES="${DOCKQ_MISMATCHES:-0}"; NB_CHAIN=""; AG_CHAIN=""; AG_CHAINS=""
EXPLICIT_REFERENCE=0; EXPLICIT_DOCKQ_MAPPING=0; EXPLICIT_DOCKQ_MISMATCHES=0
EXPLICIT_NB_CHAIN=0; EXPLICIT_AG_CHAIN=0; EXPLICIT_AG_CHAINS=0
RUN_DIR=""; SERVE_DIR=""; KERNELS="auto"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# 파일 경로를 클릭 가능한 링크로 출력. 대화형 터미널이면 OSC 8 하이퍼링크,
# 아니면(로그/파이프) 절대 file:// URL 텍스트로 출력한다.
print_link() {
  local path="$1" label="${2:-$1}" abs
  abs="$(cd "$(dirname "$path")" 2>/dev/null && pwd)/$(basename "$path")"
  if [[ -t 1 && "${TERM:-}" != "dumb" ]]; then
    printf '\033]8;;file://%s\033\\%s\033]8;;\033\\\n' "$abs" "$label"
  else
    printf 'file://%s\n' "$abs"
  fi
}

usage() {
cat <<'EOF'
Boltz-2 나노바디 예측 파이프라인
================================

[1] 단일 예측 (FASTA 두 개)
  ./run.sh                                     # inputs/target.fasta + inputs/nanobody.fasta
  ./run.sh --target ag.fasta --nanobody nb.fasta --name nb01
  ./run.sh --name nb01 --samples 5 --seed 1    # 옵션 조절

[1b] 타겟 단독/복합체 예측 (나노바디 없이)
  ./run.sh --target target.fasta --target-only --name target_only

[2] 직접 만든 Boltz YAML 사용 (ligand/수정잔기 등 고급 입력)
  ./run.sh --yaml my_input.yaml --name custom --nanobody-chain B

[3] 배치 스크리닝 (여러 나노바디/항원을 순차 실행 + 정렬 가능한 표 리포트)
  ./run.sh --batch inputs/batch.tsv --name screen1
  ./run.sh --batch inputs/batch.tsv --name screen1 --jobs nb03,nb07   # 일부만 재실행
  ./run.sh --batch inputs/batch.tsv --name screen1 --force            # 전체 재실행

[4] 리포트만 다시 만들기 (예측 결과 재사용)
  ./run.sh --report-only outputs/nb01 --reference ref.cif
  ./run.sh --batch-report outputs/batch_screen1

[4b] CDR 변이 라이브러리 만들기 (ANARCI 사용; 배치용 표까지 생성)
  ./run.sh --make-cdr-library --nanobody nb.fasta --target ag.fasta \
    --cdrs CDR3 --n-mutations 1-3 --n-variants 5 --seed 42 \
    --prefix scnE --outdir examples/E/library --batch-out examples/E/cdr_library.tsv
  ./run.sh --batch examples/E/cdr_library.tsv --name cdr_screen1   # 만든 표로 배치 실행

[5] 기타
  ./run.sh --list                              # 지금까지 돌린 실행 목록 + 핵심 점수
  ./run.sh --serve outputs/nb01/report         # 로컬 웹서버로 열기 (권장)
  ./run.sh --doctor                            # 설치/GPU/가중치 점검

주요 옵션
  --samples N      diffusion samples (기본 3): 많을수록 다양한 후보, 느려짐
  --seed N         랜덤 시드 (기본 42)
  --steps N        diffusion sampling steps (기본 200)
  --recycles N     recycling steps (기본 3)
  --msa server|empty   MSA 생성 방식 (기본 server = ColabFold MSA 서버 사용)
  --reference FILE 참조 복합체 구조(cif/pdb) -> DockQ 계산
  --dockq-mapping AB:AB  모델:참조 체인 매핑 (기본 자동)
  --dockq-mismatches N   CDR 변이체 등 참조와 서열이 다를 때 허용할 불일치 수 (기본 0)
  --nanobody-chain B  --antigen-chain A   YAML 입력에서 체인 역할 지정 (리포트 라벨 정확도)
  --antigen-chains A,D  항원 체인 집합을 콤마로 명시 (Fv 의 VL 등 다른 사슬을 항원에서 제외)
  --gpu N          사용할 GPU 번호 (CUDA_VISIBLE_DEVICES)
  --parallel-samples N  동시에 접는 샘플 수 (기본 1: VRAM 절약, 샘플이 많으면 2)
  --msa-subsample N MSA 행을 N개로 subsampling (0=기존 전체 MSA, 기본 0)
  --no-oom-retry    GPU OOM 시 설정을 변경하지 않고 실패로 종료
  --concurrency N  배치에서 동시에 돌릴 job 수 (기본 1; VRAM 여유가 부족하면 자동으로 낮춤)
  --devices N      한 job 에 쓸 GPU 수 (multi-GPU, 기본 1)
  --gpus 0,1       배치에서 job 을 GPU 별로 나눠 실행 (라운드로빈; 한 job 에는 GPU 1개)
                   한 job 에 여러 GPU 를 쓰려면 --devices N 을 쓰세요
  --hotspot 45,67  항원 epitope 잔기 지정 -> pocket 제약 (A:45-60 형식도 가능)
  --msa cache      서열별 MSA 를 msa_cache/ 에 저장해 재사용 (같은 항원 반복 배치에 유리)
  --no-kernels     cuequivariance 커널 없이 실행 (느리지만 호환성 높음)
  --cache DIR      Boltz 가중치/MSA 캐시 (기본 ~/.boltz)

CDR 변이 라이브러리 옵션 (--make-cdr-library 와 함께)
  --cdrs LIST      변이 넣을 CDR (기본 CDR1,CDR2,CDR3)
  --n-mutations S  variant 당 변이 수 (기본 2; '2', '1-3', '1,2,3')
  --n-variants N   variant 수 (기본 12; 야생형 대조 포함)
  --mode random|ala|conservative   변이 방식 (기본 random)
  --scheme imgt|kabat|chothia|martin
  --prefix P --outdir DIR --batch-out TSV   기본: FASTA 헤더 / examples/cdr_library
  --exhaustive --allow-cys                  조합 전수 / 시스테인 허용

환경변수: BOLTZ_ENV(conda env 경로), SAMPLES, SEED, STEPS, RECYCLES, WORKERS, MSA_MODE, MSA_SUBSAMPLE, OOM_RETRY, PORT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2;;
  --target-only) TARGET_ONLY=1; EXPLICIT_TARGET_ONLY=1; shift;;
    --nanobody) NANOBODY="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --yaml) YAML="$2"; MODE="yaml"; shift 2;;
    --batch) BATCH="$2"; MODE="batch"; shift 2;;
    --jobs) JOBS_FILTER="$2"; shift 2;;
    --report-only) RUN_DIR="$2"; MODE="report"; shift 2;;
    --batch-report) RUN_DIR="$2"; MODE="batch-report"; shift 2;;
    --serve) SERVE_DIR="$2"; MODE="serve"; shift 2;;
    --doctor) MODE="doctor"; shift;;
    --list) MODE="list"; shift;;
    --make-cdr-library|--cdr-library) MODE="cdr-library"; shift;;
    --cdrs) CDR_CDRS="$2"; shift 2;;
    --n-mutations) CDR_N_MUTATIONS="$2"; shift 2;;
    --n-variants) CDR_N_VARIANTS="$2"; shift 2;;
    --mode) CDR_MODE="$2"; shift 2;;
    --scheme) CDR_SCHEME="$2"; shift 2;;
    --prefix) CDR_PREFIX="$2"; shift 2;;
    --outdir) CDR_OUTDIR="$2"; shift 2;;
    --batch-out) CDR_BATCH_OUT="$2"; shift 2;;
    --exhaustive) CDR_EXHAUSTIVE=1; shift;;
    --allow-cys) CDR_ALLOW_CYS=1; shift;;
    --paratope-from) CDR_PARATOPE_FROM="$2"; shift 2;;
    --paratope-min-contacts) CDR_PARATOPE_MIN_CONTACTS="$2"; shift 2;;
    --exclude-positions) CDR_EXCLUDE_POSITIONS="$2"; shift 2;;
    --reference) REFERENCE="$2"; EXPLICIT_REFERENCE=1; shift 2;;
    --dockq-mapping) DOCKQ_MAPPING="$2"; EXPLICIT_DOCKQ_MAPPING=1; shift 2;;
    --dockq-mismatches) DOCKQ_MISMATCHES="$2"; EXPLICIT_DOCKQ_MISMATCHES=1; shift 2;;
    --nanobody-chain) NB_CHAIN="$2"; EXPLICIT_NB_CHAIN=1; shift 2;;
    --antigen-chain) AG_CHAIN="$2"; EXPLICIT_AG_CHAIN=1; shift 2;;
    --antigen-chains) AG_CHAINS="$2"; EXPLICIT_AG_CHAINS=1; shift 2;;
    --samples) SAMPLES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --recycles) RECYCLES="$2"; shift 2;;
    --msa) MSA_MODE="$2"; shift 2;;
    --msa-cache) MSA_MODE="cache"; shift;;
    --gpu) GPU="$2"; shift 2;;
    --cache) export BOLTZ_CACHE="$2"; shift 2;;
    --parallel-samples) PARALLEL_SAMPLES="$2"; shift 2;;
    --msa-subsample) MSA_SUBSAMPLE="$2"; shift 2;;
    --no-oom-retry) OOM_RETRY=0; shift;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --hotspot) HOTSPOT="$2"; shift 2;;
    --kernels) KERNELS="yes"; shift;;
    --no-kernels) KERNELS="no"; shift;;
    --no-embed) NO_EMBED=1; shift;;
    --force) FORCE=1; shift;;
    --port) PORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "알 수 없는 옵션: $1 (도움말: ./run.sh --help)";;
  esac
done

# ---------------------------------------------------------------------------
# 환경 탐지
# ---------------------------------------------------------------------------
detect_env() {
  local candidates=()
  # 1) BOLTZ_ENV 명시가 최우선
  if [[ -n "$ENV_DIR" ]]; then
    [[ -x "$ENV_DIR/bin/boltz" ]] && return 0
    die "BOLTZ_ENV 로 지정한 환경에서 boltz 실행 파일을 찾지 못했습니다: $ENV_DIR"
  fi
  # 2) setup.sh 가 기록한 env 포인터(.boltz_env). 다른 env(예: 옛 miniforge3 boltz2)가
  #    먼저 잡혀 의존성이 빠진 환경으로 도는 것을 막는다.
  if [[ -f "$ROOT/.boltz_env" ]]; then
    local pinned
    pinned="$(tr -d '[:space:]' < "$ROOT/.boltz_env" 2>/dev/null || true)"
    if [[ -n "$pinned" && -x "$pinned/bin/boltz" ]]; then ENV_DIR="$pinned"; return 0; fi
  fi
  # 3) 알려진 위치(conda/miniforge/micromamba/mambaforge)
  candidates+=(
    "$HOME/miniforge3/envs/boltz2"
    "$HOME/miniforge3/envs/boltz"
    "$HOME/miniforge3/envs/ncf-boltz2"
    "$HOME/miniconda3/envs/boltz2"
    "$HOME/mambaforge/envs/boltz2"
    "$HOME/micromamba/envs/boltz2"
    "$HOME/micromamba/envs/boltz"
  )
  for c in "${candidates[@]}"; do
    if [[ -x "$c/bin/boltz" ]]; then ENV_DIR="$c"; return 0; fi
  done
  # 4) conda/micromamba env list 폴백
  local mgr p
  for mgr in conda micromamba mamba; do
    command -v "$mgr" >/dev/null 2>&1 || continue
    p="$("$mgr" env list 2>/dev/null | awk '$1=="boltz2"||$1=="boltz"{print $NF; exit}')"
    if [[ -n "${p:-}" && -x "$p/bin/boltz" ]]; then ENV_DIR="$p"; return 0; fi
  done
  if command -v boltz >/dev/null 2>&1; then ENV_DIR="$(dirname "$(dirname "$(command -v boltz)")")"; return 0; fi
  return 1
}

# 실행 전 환경 점검: 필요한 모듈이 없으면 GPU를 쓰기 전에 멈추고 설치법을 안내한다.
require_runtime_deps() {
  local out core soft
  out="$("$PY" - <<'PYEOF'
import importlib
core = ("torch", "boltz", "numpy", "gemmi", "yaml")
soft = ("matplotlib", "anarci", "DockQ")

def missing(mods):
    out = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            out.append(m)
    return out

print("CORE:" + ",".join(missing(core)))
print("SOFT:" + ",".join(missing(soft)))
PYEOF
)"
  core="$(printf '%s\n' "$out" | awk -F: '/^CORE:/{print $2}')"
  soft="$(printf '%s\n' "$out" | awk -F: '/^SOFT:/{print $2}')"
  if [[ -n "$core" ]]; then
    die "선택된 환경에 필수 모듈이 없습니다: $core
  환경: $ENV_DIR
  설치: bash setup.sh --reuse-env --with-hmmer
  다른 환경을 쓰려면: BOLTZ_ENV=/path/to/env ./run.sh ..."
  fi
  if [[ -n "$soft" ]]; then
    warn "환경($ENV_DIR)에 선택 모듈이 없습니다: $soft"
    case " $soft " in *" matplotlib "*) warn "  matplotlib 없음 -> 그림 생략, 지표·리포트는 생성 (설치: bash setup.sh --reuse-env)";; esac
    case " $soft " in *" anarci "*) warn "  anarci 없음 -> CDR 주석/변이 생성 생략 (설치: bash setup.sh --reuse-env --with-hmmer)";; esac
    case " $soft " in *" DockQ "*) warn "  DockQ 없음 -> 참조 구조 DockQ 계산 생략 (설치: bash setup.sh --reuse-env)";; esac
  fi
  return 0
}

for pair in "SAMPLES:$SAMPLES" "SEED:$SEED" "STEPS:$STEPS" "RECYCLES:$RECYCLES" \
            "WORKERS:$WORKERS" "PARALLEL_SAMPLES:$PARALLEL_SAMPLES" \
            "CONCURRENCY:$CONCURRENCY" "DEVICES:$DEVICES" "MSA_SUBSAMPLE:$MSA_SUBSAMPLE"; do
  name="${pair%%:*}"; val="${pair#*:}"
  [[ "$val" =~ ^[0-9]+$ ]] || die "$name 값은 정수여야 합니다 (입력값: '$val')"
done
[[ "$CONCURRENCY" -ge 1 ]] || die "--concurrency 는 1 이상이어야 합니다"
[[ "$SAMPLES" -ge 1 ]] || die "--samples 는 1 이상이어야 합니다"
[[ "$STEPS" -ge 1 ]] || die "--steps 는 1 이상이어야 합니다"
[[ "$RECYCLES" -ge 1 ]] || die "--recycles 는 1 이상이어야 합니다"
[[ "$PARALLEL_SAMPLES" -ge 1 ]] || die "--parallel-samples 는 1 이상이어야 합니다"
[[ "$DEVICES" -ge 1 ]] || die "--devices 는 1 이상이어야 합니다"
[[ "$OOM_RETRY" == "0" || "$OOM_RETRY" == "1" ]] || die "OOM_RETRY 는 0 또는 1이어야 합니다"
REQUESTED_MSA_SUBSAMPLE="$MSA_SUBSAMPLE"
case "$MSA_MODE" in
  server|empty|cache) ;;
  *) die "--msa 는 server, empty, cache 중 하나여야 합니다 (입력값: $MSA_MODE)";;
esac

detect_env || die "Boltz 실행 환경을 찾지 못했습니다. 'bash setup.sh' 로 설치하거나 BOLTZ_ENV=/path/to/env 로 지정하세요."
PY="$ENV_DIR/bin/python"
BOLTZ="$ENV_DIR/bin/boltz"
[[ -x "$PY" ]] || PY="python3"
export PATH="$ENV_DIR/bin:$PATH"
[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"
log "실행 환경: $ENV_DIR ($("$PY" --version 2>&1))"
_other_envs=()
for _c in "$HOME/miniforge3/envs/boltz2" "$HOME/miniforge3/envs/boltz" "$HOME/miniforge3/envs/ncf-boltz2" \
          "$HOME/miniconda3/envs/boltz2" "$HOME/mambaforge/envs/boltz2"; do
  [[ -x "$_c/bin/boltz" && "$_c" != "$ENV_DIR" ]] && _other_envs+=("$_c")
done
if [[ ${#_other_envs[@]} -gt 0 ]]; then
  warn "boltz 가 설치된 다른 환경도 있습니다: ${_other_envs[*]}"
  warn "  지금 사용: $ENV_DIR | 다른 것을 쓰려면: BOLTZ_ENV=<경로> ./run.sh ..."
fi

KERNEL_ARGS=()
if [[ "$KERNELS" == "no" ]]; then
  KERNEL_ARGS=(--no_kernels)
elif [[ "$KERNELS" == "auto" ]]; then
  # cuequivariance 패키지의 존재 여부만 보면 부족하다. torch/cuequivariance 버전이
  # 맞지 않으면(예: torch 2.7 + cuequivariance 0.11 의 is_fx_tracing_symbolic_tracing)
  # 실제 사용 모듈 import 가 실패해 예측이 죽는다. Boltz 가 쓰는 실제 모듈을 검사해
  # 실패하면 --no_kernels 로 자동 대체한다.
  if ! "$PY" -c "import torch, cuequivariance_torch, cuequivariance_ops_torch.attention_pair_bias" >/dev/null 2>&1; then
    KERNEL_ARGS=(--no_kernels)
  fi
fi
EFFECTIVE_KERNEL_MODE="enabled"
if [[ ${#KERNEL_ARGS[@]} -gt 0 ]]; then EFFECTIVE_KERNEL_MODE="disabled"; fi
RUNTIME_PROVENANCE="$("$PY" "$SCRIPTS/runtime_state.py" runtime-provenance "$BOLTZ" \
  --kernel-mode "$EFFECTIVE_KERNEL_MODE")" || die "런타임 provenance 계산 실패"

# hmmscan (ANARCI CDR 주석용) 자동 탐색
if ! command -v hmmscan >/dev/null 2>&1; then
  for d in "$HOME"/miniforge3/envs/*/bin "$HOME"/miniconda3/envs/*/bin; do
    if [[ -x "$d/hmmscan" ]]; then export PATH="$PATH:$d"; break; fi
  done
fi

# ---------------------------------------------------------------------------
# 단계 함수
# ---------------------------------------------------------------------------
prepare_input() {
  local name="$1" target="$2" nb="$3"
  local args=(--target "$target" --name "$name" --outdir "$GEN_ROOT" --msa "$MSA_MODE")
  if [[ "$TARGET_ONLY" == "1" ]]; then
    args+=(--target-only)
  else
    args+=(--nanobody "$nb")
  fi
  [[ -n "$HOTSPOT" ]] && args+=(--hotspot "$HOTSPOT")
  "$PY" "$SCRIPTS/prepare_input.py" "${args[@]}" || die "입력 준비 실패"
}

run_fingerprint() {
  # 입력 YAML + 실행 파라미터 + 런타임/실행 파일 provenance 지문 (C-03)
  local yaml="$1" resources
  resources="$("$PY" - "$SCRIPTS" "$yaml" "$MSA_MODE" "$MSA_CACHE_DIR" <<'PYEOF'
import json, sys
sys.path.insert(0, sys.argv[1])
from runtime_state import msa_resource_hashes
print(json.dumps(msa_resource_hashes(*sys.argv[2:]), sort_keys=True))
PYEOF
  )" || return 1
  { cat "$yaml"
    printf 'msa_resources=%s\n' "$resources"
    printf 'samples=%s\nseed=%s\nsteps=%s\nrecycles=%s\nmsa=%s\nparallel=%s\ndevices=%s\nruntime=%s\n' \
      "$SAMPLES" "$SEED" "$STEPS" "$RECYCLES" "$MSA_MODE" "$PARALLEL_SAMPLES" "$DEVICES" \
      "$RUNTIME_PROVENANCE"
    printf 'msa_subsample=%s\noom_retry=%s\noom_fallback_msa=512\n' "$MSA_SUBSAMPLE" "$OOM_RETRY"
  } | "$PY" -c "import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:12])"
}

completion_fingerprint() {
  local yaml="$1" meta="$2" reference="$3" prediction_fp
  prediction_fp="$(run_fingerprint "$yaml")" || return 1
  "$PY" - "$SCRIPTS" "$prediction_fp" "$meta" "$reference" "$DOCKQ_MAPPING" \
    "${DOCKQ_MISMATCHES:-0}" "$TARGET_ONLY" "$NB_CHAIN" "$AG_CHAIN" "$AG_CHAINS" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from runtime_state import job_fingerprint
fp, meta, reference, mapping, mismatches, target_only, nb, ag, ags = sys.argv[2:]
print(job_fingerprint(fp, meta, reference, {
    "dockq_mapping": mapping, "dockq_allowed_mismatches": int(mismatches),
    "target_only": target_only, "nanobody_chain": nb, "antigen_chain": ag,
    "antigen_chains": ags,
}))
PYEOF
}

completed_artifacts_valid() {
  "$PY" - "$SCRIPTS" "$1" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from make_report import rebase_result_paths, validate_results
root = Path(sys.argv[2]).resolve()
try:
    path = root / "analysis/results.json"
    results = json.loads(path.read_text())
    validate_results(results, rebase_result_paths(results, path))
    for name in ("index.html", "summary.tsv", "results.json", "report.js", "molstar.js", "molstar.css"):
        p = root / "report" / name
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f"missing report artifact: {name}")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"[cache] incomplete completed job: {exc}", file=sys.stderr)
    raise SystemExit(1)
PYEOF
}

validate_run_name_or_die() {
  local name="$1" root="$2"
  "$PY" "$SCRIPTS/runtime_state.py" validate-name "$name" --root "$root" >/dev/null \
    || die "안전하지 않은 실행 이름입니다: $name"
}

prediction_dir_from_boltz_dir() {
  "$PY" - "$1" <<'PYEOF'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / "predictions"
for cand in sorted(root.glob("*")) if root.exists() else []:
    if cand.is_dir() and any(cand.glob("confidence_*_model_*.json")):
        print(cand)
        raise SystemExit(0)
raise SystemExit(1)
PYEOF
}

validate_active_predictions() {
  local out_dir="$1" expected="$2" active pred_dir
  active="$(record_active_boltz_dir "$out_dir")" || return 1
  pred_dir="$(prediction_dir_from_boltz_dir "$active")" || return 1
  "$PY" "$SCRIPTS/runtime_state.py" validate-predictions "$pred_dir" --expected-samples "$expected"
}

clear_partial_prediction() {
  local out_dir="$1"
  rm -rf "$out_dir"/boltz_results_* "$out_dir/.active_boltz_dir" "$out_dir/.run_fingerprint" \
    "$out_dir/.fingerprint.tmp" 2>/dev/null || true
}

prepare_boltz_dir() {
  # 지문이 바뀌었거나 --force 면 이전 예측 캐시(processed 포함)를 지운다.
  # boltz 는 processed/records/<stem>.json 이 있으면 --override 와 무관하게 입력 재처리를 건너뛴다.
  local yaml="$1" out_dir="$2" fp
  fp="$(run_fingerprint "$yaml")"
  mkdir -p "$out_dir"
  local has_cache=0
  if compgen -G "$out_dir/boltz_results_*" >/dev/null 2>&1; then has_cache=1; fi
  local old_fp=""
  [[ -f "$out_dir/.run_fingerprint" ]] && old_fp="$(cat "$out_dir/.run_fingerprint")"
  if [[ "$has_cache" == "1" && ( "$old_fp" != "$fp" || "$FORCE" == "1" ) ]]; then
    # 지문 파일이 없는 과거 실행도 '변경'으로 취급한다 (기존 processed/ 재사용 방지)
    log "입력/파라미터 변경(또는 --force) -> 이전 예측 캐시 삭제 (processed 포함)"
    rm -rf "$out_dir"/boltz_results_* "$out_dir/.active_boltz_dir" 2>/dev/null || true
  fi
  printf '%s' "$fp" > "$out_dir/.fingerprint.tmp" && mv "$out_dir/.fingerprint.tmp" "$out_dir/.run_fingerprint"
  "$PY" - "$out_dir" "$fp" "$SAMPLES" "$SEED" "$STEPS" "$RECYCLES" "$MSA_MODE" \
    "$RUNTIME_PROVENANCE" "$PARALLEL_SAMPLES" "$DEVICES" "$MSA_SUBSAMPLE" \
    "$REQUESTED_MSA_SUBSAMPLE" "${OOM_RETRY_APPLIED:-0}" <<'PYEOF'
import json, sys, datetime
out, fp, samples, seed, steps, recycles, msa, runtime, parallel, devices = sys.argv[1:11]
msa_subsample, requested_msa_subsample, oom_retry_applied = sys.argv[11:14]
json.dump({
    "fingerprint": fp, "samples": int(samples), "seed": int(seed), "steps": int(steps),
    "recycles": int(recycles), "msa": msa, "runtime": json.loads(runtime),
    "parallel_samples": int(parallel), "devices": int(devices),
    "msa_subsample": int(msa_subsample), "requested_msa_subsample": int(requested_msa_subsample),
    "oom_retry_applied": oom_retry_applied == "1",
    "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
}, open(f"{out}/.run_params.json", "w"), indent=1)
PYEOF
}

record_active_boltz_dir() {
  local out_dir="$1"
  local d
  d="$(ls -dt "$out_dir"/boltz_results_* 2>/dev/null | head -1 || true)"
  if [[ -n "$d" && -d "$d/predictions" ]]; then
    printf '%s' "$d" > "$out_dir/.active_boltz_dir"
    printf '%s' "$d"
    return 0
  fi
  return 1
}

run_boltz() {
  local yaml="$1" out_dir="$2" attempt_log="$3"
  local args=(predict "$yaml" --out_dir "$out_dir" --devices "$DEVICES"
              --diffusion_samples "$SAMPLES" --seed "$SEED" --sampling_steps "$STEPS"
              --recycling_steps "$RECYCLES" --output_format mmcif --write_full_pae
              --num_workers "$WORKERS" --max_parallel_samples "$PARALLEL_SAMPLES")
  [[ "$MSA_MODE" == "server" ]] && args+=(--use_msa_server)
  [[ "$FORCE" == "1" ]] && args+=(--override)
  [[ "$MSA_SUBSAMPLE" -gt 0 ]] && args+=(--subsample_msa --num_subsampled_msa "$MSA_SUBSAMPLE")
  if command -v nvidia-smi >/dev/null 2>&1; then
    local free_mb dev_idx
    dev_idx="${CUDA_VISIBLE_DEVICES:-}"      # M-09: 배정된 GPU 기준으로 확인 (없으면 첫 GPU)
    dev_idx="${dev_idx%%,*}"
    if [[ "$dev_idx" =~ ^[0-9]+$ ]]; then
      free_mb="$(nvidia-smi --id="$dev_idx" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    else
      free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    fi
    [[ "$free_mb" =~ ^[0-9]+$ ]] || free_mb=0   # H-07: 비수치 출력('No devices' 등) 방어
    if [[ "$free_mb" -lt 9000 ]]; then
      warn "GPU 여유 메모리가 ${free_mb}MB 입니다. 긴 입력·다중 체인은 더 필요할 수 있습니다. --msa-subsample 512 --parallel-samples 1 또는 다른 GPU 작업 종료를 검토하세요."
    fi
  fi
  prepare_boltz_dir "$yaml" "$out_dir"
  log "Boltz-2 예측 시작: $(basename "$yaml") (samples=$SAMPLES seed=$SEED steps=$STEPS msa_subsample=$MSA_SUBSAMPLE)"
  # H-04(정밀): 마지막 명령을 record_active_boltz_dir(... || true) 로 두면 Boltz 의
  # 비정상 종료 코드가 삼켜진다. 종료 코드를 보존해 그대로 반환한다.
  local rc=0
  local -a command_status=()
  if "$BOLTZ" "${args[@]}" "${KERNEL_ARGS[@]}" 2>&1 | tee "$attempt_log"; then
    command_status=("${PIPESTATUS[@]}")
  else
    command_status=("${PIPESTATUS[@]}")
  fi
  rc="${command_status[0]}"
  if [[ "$rc" == "0" && "${command_status[1]}" != "0" ]]; then
    warn "예측 로그 저장 실패: $attempt_log"
    rc="${command_status[1]}"
  fi
  record_active_boltz_dir "$out_dir" >/dev/null || true
  return "$rc"
}

cached_yaml_path() {
  # .yaml/.yml 모두에서 python 과 동일한 규칙(with_suffix)으로 캐시 YAML 경로를 얻는다
  "$PY" - "$1" <<'PYEOF'
import sys
from pathlib import Path
print(Path(sys.argv[1]).with_suffix(".cached.yaml"))
PYEOF
}

run_job_prediction() {
  # OOM은 Boltz가 exit 0으로 삼킬 수 있으므로 산출물과 로그를 함께 확인한다.
  local yaml="$1" out="$2" tries=2 i=1
  while [[ "$i" -le "$tries" ]]; do
    local rc=0 oom=0 attempt_log="$out/prediction_attempt_${i}.log"
    if run_boltz "$yaml" "$out" "$attempt_log"; then
      if validate_active_predictions "$out" "$SAMPLES"; then
        return 0
      fi
    else
      rc=$?
    fi
    if grep -Eiq 'out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$attempt_log"; then
      oom=1
      warn "GPU 메모리 부족(OOM)으로 예측이 실패했습니다. 로그: $attempt_log"
    elif [[ "$rc" == "0" ]]; then
      warn "Boltz 산출물 검증 실패: 요청 샘플 수($SAMPLES), 구조/PAE/pLDDT/신뢰도 파일을 확인합니다. 로그: $attempt_log"
    else
      if [[ "$rc" -ge 128 ]]; then
        # 139=SIGSEGV, 134=SIGABRT 등. 흔히 첫 CUDA 컨텍스트 초기화/드라이버 문제로
        # 한 번 죽었다가 재시도에서 성공한다. 재시도로 대개 해결되지만 반복되면 환경 점검 필요.
        warn "Boltz 가 시그널로 비정상 종료했습니다 (rc=$rc, signal $((rc - 128))). CUDA/드라이버 일시 문제일 수 있어 재시도합니다."
      fi
    fi
    i=$((i + 1))
    if [[ "$oom" == "1" ]]; then
      if [[ "$OOM_RETRY" != "1" || "$i" -gt "$tries" ]]; then
        warn "OOM 복구 종료. nvidia-smi로 다른 작업을 확인하거나 --msa-subsample 256 --parallel-samples 1로 새 실행을 시도하세요."
        return 1
      fi
      local reduced_msa=512
      if [[ "$MSA_SUBSAMPLE" -gt 0 && "$MSA_SUBSAMPLE" -lt "$reduced_msa" ]]; then
        reduced_msa="$MSA_SUBSAMPLE"
      fi
      if [[ "$MSA_SUBSAMPLE" == "$reduced_msa" && "$PARALLEL_SAMPLES" == "1" ]]; then
        warn "이미 MSA subsample=$MSA_SUBSAMPLE, parallel_samples=1입니다. 같은 설정으로 반복하지 않습니다."
        return 1
      fi
      MSA_SUBSAMPLE="$reduced_msa"
      PARALLEL_SAMPLES=1
      OOM_RETRY_APPLIED=1
      warn "메모리 절약 재시도: MSA subsample=$MSA_SUBSAMPLE, parallel_samples=1 (요청 모델 $SAMPLES개 유지). MSA 입력 행 수가 달라지므로 결과가 달라질 수 있으며 실제 설정을 기록합니다."
    fi
    if [[ "$i" -le "$tries" ]]; then
      warn "부분 산출물을 삭제하고 ${RETRY_SLEEP}초 후 재시도합니다 ($((i - 1))/$((tries - 1)))"
      clear_partial_prediction "$out"
      sleep "$RETRY_SLEEP"
    fi
  done
  return 1
}

run_analysis() {
  local out_dir="$1" yaml="$2" meta="$3" reference="$4"
  local args=(--run-dir "$out_dir")
  [[ -f "$yaml" ]] && args+=(--yaml "$yaml")
  [[ -f "$meta" ]] && args+=(--metadata "$meta")
  # C-03: 이번 실행이 쓴 boltz 디렉터리를 고정 (옛 캐시/비캐시 산출물 혼용 방지)
  if [[ -f "$out_dir/.active_boltz_dir" ]]; then
    args+=(--boltz-dir "$(cat "$out_dir/.active_boltz_dir")")
  fi
  if [[ -n "$reference" ]]; then
    [[ -f "$reference" ]] || die "--reference 경로를 찾을 수 없습니다: $reference"   # M-06
    args+=(--reference "$reference")
  fi
  [[ -n "$DOCKQ_MAPPING" ]] && args+=(--dockq-mapping "$DOCKQ_MAPPING")
  [[ "${DOCKQ_MISMATCHES:-0}" != "0" ]] && args+=(--dockq-allowed-mismatches "$DOCKQ_MISMATCHES")
  [[ "$TARGET_ONLY" == "1" ]] && args+=(--no-nanobody)
  [[ -n "$NB_CHAIN" ]] && args+=(--nanobody-chain "$NB_CHAIN")
  [[ -n "$AG_CHAIN" ]] && args+=(--antigen-chain "$AG_CHAIN")
  [[ -n "$AG_CHAINS" ]] && args+=(--antigen-chains "$AG_CHAINS")
  "$PY" "$SCRIPTS/analyze.py" "${args[@]}"
}

run_report() {
  local out_dir="$1"
  local args=(--results "$out_dir/analysis/results.json" --outdir "$out_dir/report")
  [[ "$NO_EMBED" == "1" ]] && args+=(--no-embed)
  "$PY" "$SCRIPTS/make_report.py" "${args[@]}"
}

print_job_summary() {
  local out_dir="$1"
  local txt
  txt="$("$PY" - "$out_dir/analysis/results.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
m = r["models"][0]; mx = (m.get("ipsae") or {}).get("max") or {}
primary = m.get("primary_interface") or {}
_dqd = m.get("dockq") or {}
dq = _dqd.get("headline", _dqd.get("best_dockq"))
iptm = primary.get("iptm", (m.get("boltz_pair_iptm") or {}).get("nanobody_in_antigen_frame"))
ipsae = primary.get("ipsae", mx.get("ipsae"))
pae = (m["interface_8A"] or {}).get("pae_mean_ab") if m.get("interface_8A") else None

def f(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"

if m.get("token_dims_ok") is False and not m.get("interface_8A", {}).get("n_contacts"):
    print(f"  best model_{m['index']}: 비단백질 토큰 포함(PAE 기반 지표 생략) | "
          f"pTM={f(m['boltz']['ptm'])} "
          + (f"DockQ={f(dq)}" if dq is not None else ""))
else:
    iptm_source = primary.get("iptm_source") or "legacy"
    ipsae_source = primary.get("ipsae_source") or "legacy"
    print(f"  best model_{m['index']}: ipTM={f(iptm)} [{iptm_source}] "
          f"ipSAE={f(ipsae,4) if ipsae is not None else '-'} [{ipsae_source}] "
          f"pDockQ2={f(mx.get('pdockq2'))} ifacePAE={f(pae,2)} "
          + (f"DockQ={f(dq)}" if dq is not None else ""))
PYEOF
)"
  printf '%s\n' "$txt"
}

do_single() {
  # Bash 동적 스코프: OOM 복구 설정을 이 job 안에서만 변경한다.
  local MSA_SUBSAMPLE="$MSA_SUBSAMPLE" PARALLEL_SAMPLES="$PARALLEL_SAMPLES" OOM_RETRY_APPLIED=0
  local name="${NAME:-$(date +%Y%m%d_%H%M%S)}"
  if [[ "$NO_EMBED" == "1" ]]; then
    die "--no-embed 는 현재 단일/YAML 리포트에서 지원하지 않습니다 (배치 리포트 재생성만 지원)"
  fi
  require_runtime_deps
  local target="${TARGET:-$ROOT/inputs/target.fasta}"
  local nb="${NANOBODY:-$ROOT/inputs/nanobody.fasta}"
  [[ -f "$target" || -f "$YAML" ]] || die "항원 FASTA가 없습니다: $target"
  local yaml="$YAML" meta=""
  if [[ "$MODE" == "yaml" ]]; then
    [[ -f "$yaml" ]] || die "YAML 파일이 없습니다: $yaml"
    name="${NAME:-$(basename "$yaml" .yaml)}"
    validate_run_name_or_die "$name" "$OUT_ROOT"
    validate_run_name_or_die "$name" "$GEN_ROOT"
    log "사용자 YAML 사용: $yaml"
  else
    validate_run_name_or_die "$name" "$OUT_ROOT"
    validate_run_name_or_die "$name" "$GEN_ROOT"
    if [[ "$TARGET_ONLY" != "1" ]]; then
      [[ -f "$nb" ]] || die "나노바디 FASTA가 없습니다: $nb (타겟 단독이면 --target-only)"
    fi
    yaml="$GEN_ROOT/$name.yaml"; meta="$GEN_ROOT/$name.meta.json"
    prepare_input "$name" "$target" "$nb"
  fi
  # 어떤 입력 파일이 쓰였는지 바로 보이도록 출력한다.
  log "입력 파일:"
  if [[ "$MODE" == "yaml" ]]; then
    echo "    YAML        : $yaml"
  else
    echo "    항원 FASTA  : $target"
    if [[ "$TARGET_ONLY" != "1" ]]; then
      echo "    나노바디    : $nb"
    fi
    echo "    생성 YAML   : $yaml"
    if [[ -n "$meta" ]]; then
      echo "    metadata    : $meta"
    fi
  fi
  if [[ -n "$REFERENCE" ]]; then
    echo "    참조 구조   : $REFERENCE"
  fi

  local base_fp original_yaml="$yaml"
  if [[ "$MSA_MODE" == "cache" ]]; then
    log "MSA 캐시 준비 (서열별 1회 다운로드 후 재사용)"
    cached_yaml="$(cached_yaml_path "$yaml")"
    "$PY" "$SCRIPTS/msa_cache.py" --yaml "$yaml" --cache-dir "$MSA_CACHE_DIR" \
      --out "$cached_yaml" || die "MSA 캐시 생성 실패"
    yaml="$cached_yaml"
    echo "    캐시 YAML   : $yaml"
  fi
  base_fp="$(completion_fingerprint "$original_yaml" "$meta" "$REFERENCE")" || die "완료 지문 계산 실패"
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  rm -f "$out_dir/.completed"
  rm -rf "$out_dir/analysis" "$out_dir/report"   # C-03/H-05: 옛 분석·리포트가 새 결과로 오인되지 않도록
  run_job_prediction "$yaml" "$out_dir" || die "Boltz 예측 실패 또는 산출물 불완전 (로그 확인)"
  # OOM 복구로 바뀐 실제 설정까지 완료 캐시에 반영한다.
  base_fp="$(completion_fingerprint "$original_yaml" "$meta" "$REFERENCE")" || die "완료 지문 계산 실패"
  run_analysis "$out_dir" "$yaml" "$meta" "$REFERENCE" || die "분석 실패"
  run_report "$out_dir" || die "리포트 생성 실패"
  printf '%s' "$base_fp" > "$out_dir/.completed"
  echo
  ok "완료: $name"
  print_job_summary "$out_dir"
  echo "  출력 파일:"
  echo "    실행 폴더 : $out_dir"
  echo "    분석 JSON : $out_dir/analysis/results.json"
  echo "    그림 폴더 : $out_dir/analysis/figures/"
  echo "    리포트    : $out_dir/report/index.html"
  printf '    바로 열기 : '; print_link "$out_dir/report/index.html" "report/index.html (클릭)"
  echo "    웹서버    : ./run.sh --serve $out_dir/report"
}

run_one_job() {
  local MSA_SUBSAMPLE="$MSA_SUBSAMPLE" PARALLEL_SAMPLES="$PARALLEL_SAMPLES" OOM_RETRY_APPLIED=0
  # $1=jobs_dir $2=jname $3=yaml $4=meta $5=ref  -> $out/.status 에 OK/FAIL 기록
  local jobs_dir="$1" jname="$2" jyaml="$3" jmeta="$4" jref="$5"
  local out="$jobs_dir/$jname" t0
  t0="$(date +%s)"
  # Completion includes reference, roles, scoring implementation and report sources.
  local base_fp original_yaml="$jyaml"
  mkdir -p "$jobs_dir/_logs" "$out"   # R14: 실패 상태 파일을 쓸 수 있도록 먼저 생성
  rm -f "$out/.status"                # H-08: 이전 배치의 상태가 남아 'OK'로 오집계되는 것 방지
  rm -f "$out/.completed"
  rm -rf "$out/analysis" "$out/report"  # C-03/H-05: 실패한 재실행이 옛 분석·리포트를 남기지 않도록
  {
    echo "===== $jname 시작 $(date +%H:%M:%S) ====="
    if [[ "$MSA_MODE" == "cache" ]]; then
      echo "[msa-cache] 캐시 준비"
      local cyaml
      cyaml="$(cached_yaml_path "$jyaml")"
      "$PY" "$SCRIPTS/msa_cache.py" --yaml "$jyaml" --cache-dir "$MSA_CACHE_DIR" \
        --out "$cyaml" \
        || { echo "RESULT: FAIL_MSA_CACHE"; echo "FAIL_MSA_CACHE" > "$out/.status"; return 1; }
      jyaml="$cyaml"
    fi
    base_fp="$(completion_fingerprint "$original_yaml" "$jmeta" "$jref")" \
      || { echo "FAIL_INPUT" > "$out/.status"; return 1; }
    if ! run_job_prediction "$jyaml" "$out"; then
      echo "RESULT: FAIL_PREDICT"; echo "FAIL_PREDICT" > "$out/.status"; return 1
    fi
    base_fp="$(completion_fingerprint "$original_yaml" "$jmeta" "$jref")" \
      || { echo "FAIL_INPUT" > "$out/.status"; return 1; }
    if ! run_analysis "$out" "$jyaml" "$jmeta" "$jref"; then
      echo "RESULT: FAIL_ANALYZE"; echo "FAIL_ANALYZE" > "$out/.status"; return 1
    fi
    if ! run_report "$out"; then
      echo "RESULT: FAIL_REPORT"; echo "FAIL_REPORT" > "$out/.status"; return 1
    fi
    echo "RESULT: OK ($(( $(date +%s) - t0 ))s)"
    echo "OK" > "$out/.status"
    printf '%s' "$base_fp" > "$out/.completed"
    return 0
  } >> "$jobs_dir/_logs/$jname.log" 2>&1
}

gpu_free_mb() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  local idx="${1:-${CUDA_VISIBLE_DEVICES:-}}"
  idx="${idx%%,*}"
  local v=""
  if [[ "$idx" =~ ^[0-9]+$ ]]; then
    v="$(nvidia-smi --id="$idx" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
  else
    v="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
  fi
  [[ "$v" =~ ^[0-9]+$ ]] || v=0
  printf '%s' "$v"
}

do_batch() {
  local batch_name="${NAME:-$(basename "$BATCH" .tsv)}"
  validate_run_name_or_die "$batch_name" "$OUT_ROOT"
  validate_run_name_or_die "$batch_name" "$GEN_ROOT"
  if [[ "$NO_EMBED" == "1" ]]; then
    die "--no-embed 는 현재 배치 실행 중 개별 job 리포트에서 지원하지 않습니다. 완료 후 --batch-report ... --no-embed 를 사용하세요"
  fi
  require_runtime_deps
  [[ -n "$HOTSPOT" ]] && warn "--batch 모드에서는 --hotspot 이 무시됩니다 (표의 hotspot 열을 쓰세요)"
  [[ -n "$REFERENCE" ]] && warn "--batch 모드에서는 전역 --reference 가 무시됩니다 (표의 reference 열을 쓰세요)"
  local gen_dir="$GEN_ROOT/batch_$batch_name"
  local jobs_dir="$OUT_ROOT/batch_$batch_name"
  echo "  배치 입력 표  : $BATCH"
  echo "  배치 출력 폴더: $jobs_dir"
  local reuse=0
  if [[ -f "$gen_dir/manifest.json" && "$FORCE" != "1" ]]; then
    reuse="$("$PY" - "$SCRIPTS" "$gen_dir/manifest.json" "$BATCH" "$MSA_MODE" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from prepare_batch import manifest_reusable
print(int(manifest_reusable(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])))
PYEOF
)"
  fi
  if [[ "$reuse" == "1" ]]; then
    log "기존 manifest 재사용: $gen_dir/manifest.json (표/파라미터가 같음)"
  else
    "$PY" "$SCRIPTS/prepare_batch.py" --batch "$BATCH" --name "$batch_name" \
          --outdir "$GEN_ROOT" --msa "$MSA_MODE" || die "배치 입력 준비 실패"
  fi

  mapfile -t job_lines < <("$PY" - "$gen_dir/manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
for j in m["jobs"]:
    print("\t".join([j["name"], j["yaml"], j["meta"],
                     j.get("reference") or "\x01", j.get("notes") or "\x01"]))
PYEOF
)
  if [[ -n "$JOBS_FILTER" ]]; then
    JOBS_FILTER="$("$PY" - "$gen_dir/manifest.json" "$JOBS_FILTER" <<'PYEOF'
import json
import sys

m = json.load(open(sys.argv[1]))
available = {j["name"] for j in m.get("jobs", [])}
requested = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
if not requested:
    print("[runtime] --jobs is empty", file=sys.stderr)
    raise SystemExit(1)
missing = [x for x in requested if x not in available]
if missing:
    print("[runtime] unknown --jobs: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print(",".join(requested))
PYEOF
    )" || die "--jobs 에 지정한 이름이 manifest 와 일치하지 않습니다"
  fi
  local manifest_total=${#job_lines[@]}
  local -a selected_job_lines=()
  local candidate candidate_name candidate_rest
  for candidate in "${job_lines[@]}"; do
    IFS=$'\t' read -r candidate_name candidate_rest <<< "$candidate"
    if [[ -z "$JOBS_FILTER" || ",$JOBS_FILTER," == *",$candidate_name,"* ]]; then
      selected_job_lines+=("$candidate")
    fi
  done
  mkdir -p "$jobs_dir"
  local n_total=${#selected_job_lines[@]} n_ok=0 n_skip=0 n_fail=0
  local fail_list=()
  local t_start; t_start="$(date +%s)"
  local -a pids=() pnames=()
  mkdir -p "$jobs_dir/_logs"
  local concurrency="$CONCURRENCY"

  local -a gpu_list=()
  if [[ -n "$GPUS" ]]; then IFS=',' read -r -a gpu_list <<< "$GPUS"; fi
  local launched=0 job_index=0
  for line in "${selected_job_lines[@]}"; do
    job_index=$((job_index + 1))
    local jname jyaml jmeta jref jnotes
    IFS=$'\t' read -r jname jyaml jmeta jref jnotes <<< "$line"
    # 빈 필드 sentinel 복원 + 참조 경로 검증 (노트가 reference 로 새는 문제 방지)
    [[ "$jref" == $'\x01' ]] && jref=""
    [[ "$jnotes" == $'\x01' ]] && jnotes=""
    if [[ -n "$jref" && ! -f "$jref" ]]; then
      die "$jname: reference 경로를 찾을 수 없습니다: $jref"
    fi
    if [[ "$FORCE" != "1" && -f "$jobs_dir/$jname/.completed" && -f "$jyaml" ]]; then
      local cur_fp old_fp
      cur_fp="$(completion_fingerprint "$jyaml" "$jmeta" "$jref")"
      old_fp="$(cat "$jobs_dir/$jname/.completed" 2>/dev/null)"
      if [[ -n "$old_fp" && "$cur_fp" == "$old_fp" ]] && completed_artifacts_valid "$jobs_dir/$jname"; then
        log "[SKIP] $jname (같은 입력/파라미터로 이미 완료 - 재실행은 --force)"
        n_skip=$((n_skip + 1)); continue
      fi
    fi

    # M-09(정밀): VRAM 은 이번 job 이 실제로 배정될 GPU 에서 확인해야 한다
    # (기존에는 항상 gpu_list[0] 을 봤지만 실제 배정은 round-robin 이다).
    local gpu_idx=""
    if [[ ${#gpu_list[@]} -gt 0 ]]; then
      gpu_idx="${gpu_list[$((launched % ${#gpu_list[@]}))]}"
    fi
    # 동시 실행 슬롯 대기 (VRAM 여유가 부족하면 순차 실행으로 강제)
    local eff="$concurrency"
    if [[ "$eff" -gt 1 ]]; then
      # 이 시점에 이미 돌고 있는 job 수를 세고, 각 job 이 6GB 안팎을 쓴다고 보고
      # 앞으로 띄울 여유분만 확인한다 (M-12: 이미 실행 중인 job 을 이중 계산하지 않음)
      local running_now=0 p_
      for p_ in "${pids[@]:-}"; do
        [[ -n "$p_" ]] && kill -0 "$p_" 2>/dev/null && running_now=$((running_now + 1))
      done
      local want=$((eff - running_now))
      if [[ "$want" -gt 0 ]]; then
        local need=$((want * 6000))
        local free
        free="$(gpu_free_mb "${gpu_idx:-${gpu_list[0]:-}}")"
        if [[ "${free:-0}" -lt "$need" ]]; then
          eff=$((running_now + 1))
          [[ "$eff" -lt 1 ]] && eff=1
          warn "GPU 여유 ${free}MB < ${need}MB - 이 job 부터는 동시 ${eff} 로 낮춥니다"
        fi
      fi
    fi
    while :; do
      local running=0 p
      for p in "${pids[@]:-}"; do
        [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && running=$((running + 1))
      done
      [[ "$running" -lt "$eff" ]] && break
      sleep 5
    done

    local gpu_note=""
    if [[ -n "$gpu_idx" ]]; then
      gpu_note=", GPU $gpu_idx"
      ( export CUDA_VISIBLE_DEVICES="$gpu_idx"
        run_one_job "$jobs_dir" "$jname" "$jyaml" "$jmeta" "$jref" ) &
    else
      run_one_job "$jobs_dir" "$jname" "$jyaml" "$jmeta" "$jref" &
    fi
    launched=$((launched + 1))
    log "===== JOB $job_index/$n_total : $jname 시작 (동시 $eff, 여유 VRAM $(gpu_free_mb "${gpu_idx:-}")MB$gpu_note) ====="
    pids+=("$!"); pnames+=("$jname"); CHILD_PIDS+=("$!")
  done

  # 종료 대기
  local i=0
  for p in "${pids[@]:-}"; do
    [[ -n "$p" ]] && wait "$p" 2>/dev/null || true
    local nm="${pnames[$i]:-}"
    if [[ -n "$nm" ]]; then
      if [[ "$(cat "$jobs_dir/$nm/.status" 2>/dev/null)" == "OK" ]]; then
        n_ok=$((n_ok + 1))
        ok "$nm 완료"
        print_job_summary "$jobs_dir/$nm" || true
        local done_n=$((i + 1 + n_skip))
        if [[ "$done_n" -lt "$n_total" ]]; then
          local elapsed=$(( $(date +%s) - t_start ))
          local eta=$(( elapsed * (n_total - done_n) / (done_n > 0 ? done_n : 1) ))
          log "진행 $done_n/$n_total | 경과 $((elapsed / 60))분 | 남은 예상 약 $((eta / 60))분"
        fi
      else
        n_fail=$((n_fail + 1)); fail_list+=("$nm")
        warn "$nm 실패 ($(cat "$jobs_dir/$nm/.status" 2>/dev/null || echo '원인 미상')) - 로그: $jobs_dir/_logs/$nm.log"
      fi
    fi
    i=$((i + 1))
  done
  rm -f "$jobs_dir"/*/.status 2>/dev/null || true

  local report_rc=0
  "$PY" "$SCRIPTS/batch_report.py" --manifest "$gen_dir/manifest.json" --jobs-dir "$jobs_dir" \
        --outdir "$jobs_dir/report" || report_rc=$?
  [[ "$report_rc" == "0" ]] || warn "배치 리포트에 불완전한 job 이 있습니다 (exit $report_rc)"
  echo
  if [[ -n "$JOBS_FILTER" ]]; then
    ok "배치 종료: 완료 $n_ok / 건너뜀 $n_skip / 실패 $n_fail (선택 $n_total / manifest $manifest_total)"
  else
    ok "배치 종료: 완료 $n_ok / 건너뜀 $n_skip / 실패 $n_fail (전체 $n_total)"
  fi
  if [[ ${#fail_list[@]} -gt 0 ]]; then
    warn "실패한 job: ${fail_list[*]}"
    warn "재실행: ./run.sh --batch $BATCH --name $batch_name --jobs $(IFS=,; echo "${fail_list[*]}")"
  fi
  echo "  출력 파일:"
  echo "    실행 폴더   : $jobs_dir"
  echo "    job 별 결과 : $jobs_dir/<job>/analysis/results.json"
  echo "    job 별 리포트: $jobs_dir/<job>/report/index.html"
  echo "    표 리포트   : $jobs_dir/report/index.html"
  echo "    요약 TSV    : $jobs_dir/report/summary.tsv"
  printf '    표 바로열기 : '; print_link "$jobs_dir/report/index.html" "report/index.html (클릭)"
  echo "    웹서버      : ./run.sh --serve $jobs_dir/report"
  if [[ "$n_fail" != "0" || "$report_rc" != "0" ]]; then
    warn "배치가 실패를 포함해 비정상 종료합니다 (재현: ./run.sh --batch $BATCH --name $batch_name --samples $SAMPLES --seed $SEED --msa $MSA_MODE)"
    return 1
  fi
}

do_report_only() {
  local out_dir="$RUN_DIR"
  if [[ "$NO_EMBED" == "1" ]]; then
    die "--no-embed 는 현재 단일 리포트 재생성에서 지원하지 않습니다 (배치 리포트 재생성만 지원)"
  fi
  require_runtime_deps
  local args=(--run-dir "$out_dir")
  if [[ -f "$out_dir/analysis/results.json" ]]; then
    args+=(--restore-settings "$out_dir/analysis/results.json")
  fi
  if [[ "$EXPLICIT_REFERENCE" == "1" ]]; then
    [[ -f "$REFERENCE" ]] || die "--reference 경로를 찾을 수 없습니다: $REFERENCE"
    args+=(--reference "$REFERENCE")
  fi
  [[ "$EXPLICIT_DOCKQ_MAPPING" == "1" ]] && args+=(--dockq-mapping "$DOCKQ_MAPPING")
  [[ "$EXPLICIT_DOCKQ_MISMATCHES" == "1" ]] && args+=(--dockq-allowed-mismatches "$DOCKQ_MISMATCHES")
  [[ "$EXPLICIT_TARGET_ONLY" == "1" ]] && args+=(--no-nanobody)
  [[ "$EXPLICIT_NB_CHAIN" == "1" ]] && args+=(--nanobody-chain "$NB_CHAIN")
  [[ "$EXPLICIT_AG_CHAIN" == "1" ]] && args+=(--antigen-chain "$AG_CHAIN")
  [[ "$EXPLICIT_AG_CHAINS" == "1" ]] && args+=(--antigen-chains "$AG_CHAINS")
  "$PY" "$SCRIPTS/analyze.py" "${args[@]}" || die "분석 실패"
  run_report "$out_dir" || die "리포트 생성 실패"
  ok "리포트 재생성 완료: $out_dir/report/index.html"
  printf '  바로 열기: '; print_link "$out_dir/report/index.html" "report/index.html (클릭)"
}

do_cdr_library() {
  # env/conda 를 몰라도 되도록, 감지된 env 의 python 으로 CDR 라이브러리 스크립트를 실행한다.
  if ! "$PY" -c 'import anarci' >/dev/null 2>&1; then
    die "선택된 환경에 anarci(ANARCI)가 없습니다: $ENV_DIR
  CDR 변이는 ANARCI(번호매김)와 hmmscan(HMMER)이 필요합니다.
  설치: bash setup.sh --reuse-env
  다른 환경: BOLTZ_ENV=/path/to/env ./run.sh --make-cdr-library ..."
  fi
  local args=(
    --cdrs "$CDR_CDRS" --n-mutations "$CDR_N_MUTATIONS"
    --n-variants "$CDR_N_VARIANTS" --mode "$CDR_MODE"
    --scheme "$CDR_SCHEME" --seed "$SEED"
    --paratope-min-contacts "$CDR_PARATOPE_MIN_CONTACTS"
  )
  # 비어 있으면 넘기지 않아 스크립트 기본값(inputs/nanobody.fasta, inputs/target.fasta)을 쓰게 한다.
  [[ -n "$NANOBODY" ]]         && args+=(--nanobody "$NANOBODY")
  [[ -n "$TARGET" ]]           && args+=(--target "$TARGET")
  [[ -n "$REFERENCE" ]]        && args+=(--reference "$REFERENCE")
  [[ -n "$CDR_PREFIX" ]]       && args+=(--prefix "$CDR_PREFIX")
  [[ -n "$CDR_OUTDIR" ]]       && args+=(--outdir "$CDR_OUTDIR")
  [[ -n "$CDR_BATCH_OUT" ]]    && args+=(--batch-out "$CDR_BATCH_OUT")
  [[ -n "$CDR_PARATOPE_FROM" ]] && args+=(--paratope-from "$CDR_PARATOPE_FROM")
  [[ -n "$CDR_EXCLUDE_POSITIONS" ]] && args+=(--exclude-positions "$CDR_EXCLUDE_POSITIONS")
  [[ "$CDR_EXHAUSTIVE" == "1" ]] && args+=(--exhaustive)
  [[ "$CDR_ALLOW_CYS" == "1" ]]  && args+=(--allow-cys)
  log "CDR 변이 라이브러리 생성 (env: $ENV_DIR)"
  "$PY" "$SCRIPTS/make_cdr_library.py" "${args[@]}" || die "CDR 라이브러리 생성 실패"
  if [[ -n "$CDR_BATCH_OUT" ]]; then
    printf '  배치 표로 실행: ./run.sh --batch %s --name cdr_screen1\n' "$CDR_BATCH_OUT"
  fi
}

do_serve() {
  local dir="$SERVE_DIR"
  [[ -f "$dir" ]] && dir="$(dirname -- "$dir")"
  [[ -f "$dir/index.html" ]] || die "리포트 index.html 이 없습니다: $dir/index.html
  --doctor 는 환경 점검만, --serve 는 이미 생성된 리포트 열기만 수행합니다.
  완료된 리포트 확인: ./run.sh --list
  first_run 생성 예시: ./run.sh --name first_run (완료 후 리포트를 여세요)"
  log "웹서버 시작: http://127.0.0.1:$PORT/  (Ctrl+C 종료, 로컬 접속만 허용)"
  ( cd "$dir" && "$PY" -m http.server "$PORT" --bind 127.0.0.1 )
}

do_list() {
  "$PY" - "$OUT_ROOT" "$GEN_ROOT" <<'PYEOF'
import glob, json, os, sys
out_root, gen_root = sys.argv[1], sys.argv[2]
rows = []
def primary(m):
    p = m.get("primary_interface") or {}
    iptm = p.get("iptm", (m.get("boltz_pair_iptm") or {}).get("nanobody_in_antigen_frame"))
    ipsae = p.get("ipsae", ((m.get("ipsae") or {}).get("max") or {}).get("ipsae"))
    return iptm, ipsae

def num(v):
    return v if isinstance(v, (int, float)) else None

for d in sorted(glob.glob(os.path.join(out_root, "*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    report = os.path.join(d, "report", "index.html")
    has_report = "O" if os.path.exists(report) else "-"
    if name.startswith("batch_"):
        # M-01: 표시 코호트는 디렉터리 수가 아니라 canonical manifest 를 따른다.
        # (과거 코호트의 잔존 결과 폴더가 있으면 디렉터리 수가 더 크게 보인다.)
        man_path = os.path.join(gen_root, name, "manifest.json")
        job_names = None
        if os.path.exists(man_path):
            try:
                mj = json.load(open(man_path))
                if not mj.get("errors"):
                    job_names = [j["name"] for j in mj.get("jobs", [])]
            except Exception:
                job_names = None
        if job_names is not None:
            subs = [os.path.join(d, jn, "analysis", "results.json") for jn in job_names]
            subs = [p for p in subs if os.path.exists(p)]
            count_label = f"{len(job_names)} job"
        else:
            subs = sorted(glob.glob(os.path.join(d, "*", "analysis", "results.json")))
            count_label = f"{len(subs)} job"
        best = None
        for p in subs:
            try:
                m = json.load(open(p))["models"][0]
            except Exception:
                continue
            key = num(primary(m)[0]) or 0
            if best is None or key > best[0]:
                best = (key, m)
        if best:
            _iptm, _ipsae = primary(best[1])
            rows.append((name, "배치", count_label, f"{best[0]:.3f}",
                         f"{_ipsae:.3f}" if isinstance(_ipsae, (int, float)) else "-", "-", has_report))
        else:
            rows.append((name, "배치", count_label, "-", "-", "-", has_report))
        continue
    res = os.path.join(d, "analysis", "results.json")
    if not os.path.exists(res):
        rows.append((name, "단일", "미완료", "-", "-", "-", has_report))
        continue
    try:
        r = json.load(open(res))
        m = r["models"][0]
    except Exception:
        rows.append((name, "단일", "읽기 실패", "-", "-", "-", has_report))
        continue
    _dqd = m.get("dockq") or {}
    dq = _dqd.get("headline", _dqd.get("best_dockq"))
    _iptm, _ipsae = primary(m)
    _ptm = (m.get("boltz") or {}).get("ptm")
    _score = _iptm if _iptm is not None else _ptm
    rows.append((name, "단일", f"{len(r['models'])} model",
                 f"{_score:.3f}" if isinstance(_score, (int, float)) else "-",
                 f"{_ipsae:.3f}" if isinstance(_ipsae, (int, float)) else "-",
                 f"{dq:.3f}" if isinstance(dq, (int, float)) else "-", has_report))
if not rows:
    print("아직 실행 결과가 없습니다. 먼저 ./run.sh 를 실행하세요.")
    sys.exit(0)
w = max(len(r[0]) for r in rows) + 2
print(f"{'이름':<{w}}{'종류':<5}{'모델':<11}{'ipTM/pTM':>9}{'ipSAE':>7}{'DockQ':>8}{'리포트':>8}")
print("-" * (w + 46))
for name, kind, n, a, b, c, rep in rows:
    print(f"{name:<{w}}{kind:<5}{n:<11}{a:>9}{b:>7}{c:>8}{rep:>8}")
print()
print("리포트 열기: ./run.sh --serve outputs/<이름>/report")
PYEOF
}

do_doctor() {
  local fail=0
  echo "== Boltz-2 환경 점검 =="
  echo "env          : $ENV_DIR"
  echo "python       : $("$PY" --version 2>&1)"
  if "$BOLTZ" --help >/dev/null 2>&1; then
    echo "boltz        : OK"
  else
    echo "boltz        : FAIL"
    fail=1
  fi
  "$PY" - <<'PYEOF' || fail=1
import importlib, shutil, sys
failed = False
def has(m):
    try:
        importlib.import_module(m); return True
    except Exception:
        return False
print("torch        :", end=" ")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(torch.__version__, "cuda:", cuda_ok,
          torch.cuda.get_device_name(0) if cuda_ok else "")
    failed = failed or not cuda_ok
except Exception as e:
    print("FAIL", e)
    failed = True
required = ("boltz", "matplotlib", "gemmi", "numpy")
optional = ("DockQ", "anarci")
for mod in required:
    ok = has(mod)
    print(f"{mod:13s}:", "OK" if ok else "missing")
    failed = failed or not ok
for mod in optional:
    print(f"{mod:13s}:", "OK" if has(mod) else "missing (optional)")
# 커널은 '패키지 존재'가 아니라 '실제 사용 모듈 import'로 판정한다.
try:
    import cuequivariance_torch  # noqa: F401
    import cuequivariance_ops_torch.attention_pair_bias  # noqa: F401
    print("kernel       : OK (cuequivariance 사용)")
except Exception:
    print("kernel       : unavailable -> --no-kernels 자동 사용")
raise SystemExit(1 if failed else 0)
PYEOF
  echo "hmmscan      : $(command -v hmmscan || echo 'missing (CDR 주석 생략됨)')"
  echo "가중치 캐시   : ${BOLTZ_CACHE:-$HOME/.boltz}"
  ls -la "${BOLTZ_CACHE:-$HOME/.boltz}" 2>/dev/null | grep -E "ckpt" || echo "  (ckpt 없음 - 첫 실행 시 자동 다운로드)"
  for a in molstar.js molstar.css report.js; do
    if [[ -f "$ASSETS/$a" ]]; then
      echo "asset $a: OK ($(du -h "$ASSETS/$a" | cut -f1))"
    else
      echo "asset $a: MISSING"
      fail=1
    fi
  done
  return "$fail"
}

export SCRIPTS

case "$MODE" in
  single|yaml) do_single;;
  batch) do_batch;;
  report) do_report_only;;
  batch-report)
    bdir="$RUN_DIR"
    bname="$(basename "$bdir" | sed 's/^batch_//')"
    bman="$GEN_ROOT/batch_$bname/manifest.json"
    [[ -f "$bman" ]] || die "manifest 를 찾을 수 없습니다: $bman (--batch 로 실행한 배치만 재생성 가능)"
    br_args=(--manifest "$bman" --jobs-dir "$bdir" --outdir "$bdir/report")
    [[ "$NO_EMBED" == "1" ]] && br_args+=(--no-embed)
    "$PY" "$SCRIPTS/batch_report.py" "${br_args[@]}" \
      && ok "배치 리포트 재생성: $bdir/report/index.html";;
  serve) do_serve;;
  doctor) do_doctor;;
  list) do_list;;
  cdr-library) do_cdr_library;;
esac
