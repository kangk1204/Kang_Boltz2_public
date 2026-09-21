# 시나리오 예제 입력 (바로 실행)

README 3-1의 시나리오 A~F를 **그대로 복사해 실행**할 수 있게 준비한 입력 모음입니다.
모든 명령은 **저장소 루트**에서 실행하고, 첫 설치는 끝났다고 가정합니다
(`bash setup.sh --verify`).

> **빠른 확인 팁**: 명령 끝에 `--samples 1 --steps 20 --recycles 1 --no-kernels` 를 붙이면
> 수십 초 안에 동작만 확인할 수 있습니다(품질은 낮음). 정식 결과는 붙이지 말고 실행하세요.
> 배치/CDR은 `SAMPLES=1 STEPS=20 RECYCLES=1` 를 앞에 붙입니다.

---

## A. 나노바디–항원 복합체
- 입력: `A_nanobody/target.fasta`(항원, 1MEL lysozyme), `A_nanobody/nanobody.fasta`(cAb-Lys3 VHH)
- 참조: `../demo_reference_1MEL_AB.cif` (DockQ 계산용)

```bash
./run.sh --target examples/scenarios/A_nanobody/target.fasta \
         --nanobody examples/scenarios/A_nanobody/nanobody.fasta \
         --reference examples/demo_reference_1MEL_AB.cif \
         --name scn_A
```
결과: `outputs/scn_A/report/index.html`

## B. 항체(Fv, VH+VL)–항원 복합체
- 입력: `B_fv/fv.yaml` — A=lysozyme(1VFB C), B=D1.3 VH(1VFB B), C=D1.3 VL(1VFB A)
  (보조 파일: `B_fv/target.fasta`, `B_fv/vh.fasta`, `B_fv/vl.fasta`)
- 참조: `B_fv/reference_D1.3_lysozyme.cif` (PDB **1VFB**에서 모델 체인명 A/B/C 로 맞춘 것 → DockQ)

```bash
./run.sh --yaml examples/scenarios/B_fv/fv.yaml --name scn_B \
  --nanobody-chain B --antigen-chain A --antigen-chains A \
  --reference examples/scenarios/B_fv/reference_D1.3_lysozyme.cif \
  --msa-subsample 512
```
결과: `outputs/scn_B/report/index.html` — VH 기준 분석, VL(C)은 `other`(3D에만 표시), VH–항원 DockQ 계산

`--msa-subsample 512`는 16GB GPU에서 메모리 사용을 줄이도록 모델의 MSA 행 수를 제한합니다.
기본 모델 3개는 유지하지만 전체 MSA 실행과 수치는 달라질 수 있습니다. 실제 설정은
`.run_params.json`에 기록됩니다. WT–변이 비교에서는 이 설정도 동일하게 맞추세요.

> **관찰(정직한 기록)**: 이 D1.3–lysozyme 예제에서 기존 전체 MSA 기본 실행의 VH–항원 ipTM은 낮게(≈0.29) 나왔고
> DockQ도 낮았습니다. 반면 Fv 자체는 잘 접힙니다(VH–VL ipTM≈0.96). 즉 **모델이 이 항체–항원
> 인터페이스를 확신하지 못한 결과**이며, 파이프라인·참조 구조 문제가 아닙니다(참조가 있어 DockQ로
> 정량 확인 가능). 나노바디 예제 A(1MEL)가 안정적으로 잘 맞는 대조 사례입니다.

## C. 항원만 예측 (target-only)
- 입력: `C_target_only/target.fasta`

```bash
./run.sh --target examples/scenarios/C_target_only/target.fasta --target-only --name scn_C
```
결과: `outputs/scn_C/report/index.html`

## D. 배치 (여러 후보)
- 입력: `D_batch/batch.tsv` (job 3개: 야생형·CDR2 변이·Fv VH)

```bash
./run.sh --batch examples/scenarios/D_batch/batch.tsv --name scn_D
```
결과: `outputs/batch_scn_D/report/index.html` (+ `report/summary.tsv`)

## E. CDR 변이 — 나노바디
- 입력: `E_cdr_nanobody/nanobody.fasta` (라이브러리는 자동 생성)

```bash
./run.sh --make-cdr-library \
  --nanobody examples/scenarios/E_cdr_nanobody/nanobody.fasta \
  --target examples/scenarios/A_nanobody/target.fasta \
  --reference examples/demo_reference_1MEL_AB.cif \
  --cdrs CDR3 --n-mutations 1-3 --n-variants 5 --seed 42 \
  --prefix scnE --outdir examples/scenarios/E_cdr_nanobody/library \
  --batch-out examples/scenarios/E_cdr_nanobody/cdr_library.tsv

SAMPLES=1 ./run.sh --batch examples/scenarios/E_cdr_nanobody/cdr_library.tsv --name scn_E
```
결과: `outputs/batch_scn_E/report/index.html`

## F. CDR 변이 — 항체 VH / VL (각각)
- 입력: `F_cdr_antibody/vh.fasta`(D1.3 VH), `F_cdr_antibody/vl.fasta`(D1.3 VL)
- **핵심**: 변이는 한 번에 한 사슬, 접두어·출력경로를 H/L로 분리(안 하면 job 이름 중복).

```bash
# VH (CDR3만)
./run.sh --make-cdr-library \
  --nanobody examples/scenarios/F_cdr_antibody/vh.fasta \
  --target examples/scenarios/B_fv/target.fasta \
  --cdrs CDR3 --n-mutations 1-3 --n-variants 5 --seed 42 \
  --prefix D13_H --outdir examples/scenarios/F_cdr_antibody/lib_H \
  --batch-out examples/scenarios/F_cdr_antibody/cdr_H.tsv

# VL (CDR1,2,3)
./run.sh --make-cdr-library \
  --nanobody examples/scenarios/F_cdr_antibody/vl.fasta \
  --target examples/scenarios/B_fv/target.fasta \
  --cdrs CDR1,CDR2,CDR3 --n-mutations 1-3 --n-variants 5 --seed 42 \
  --prefix D13_L --outdir examples/scenarios/F_cdr_antibody/lib_L \
  --batch-out examples/scenarios/F_cdr_antibody/cdr_L.tsv
```
평가(VH): `SAMPLES=1 ./run.sh --batch examples/scenarios/F_cdr_antibody/cdr_H.tsv --name scn_F_H`
평가(VL/Fv): `cdr_*.tsv` 배치는 단일 binder 전제라, 변이 서열을 Fv YAML의 해당 체인에 넣어 `--yaml`로 실행합니다.

---

## 서열 출처
- A/C: 기존 예제(1MEL, lysozyme + cAb-Lys3 VHH).
- B/F: PDB **1VFB** — hen egg lysozyme(chain C) + anti-lysozyme **D1.3 Fv**(VH chain B, VL chain A).
  (https://www.rcsb.org/structure/1VFB)
