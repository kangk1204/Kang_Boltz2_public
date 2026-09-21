# 전달용 테스트셋

현재 코호트는 `batch_wt_updated.tsv`의 WT 9개와 `batch_mutants_updated.tsv`의 변이 180개다.
`cohort_snapshot_2026-09-19.json`에 입력·분석 JSON의 SHA-256, 대표 모델, 수치와 실행 조건을 기록했다.
`mutant_screening_summary.tsv`는 이 180개의 기존 탐색적 결과표다.
기존 결과는 WT 3 samples, 변이 1 sample이고, 변이 DockQ와 WT 대비 Δ는 없다.
6개 표적군은 cache, 3개는 server MSA 정책으로 실행했다. 하나의 동일 조건 비교 실험이 아니다.

Git에는 입력·요약·snapshot만 포함되며 `outputs/`의 구조·그림·HTML·MSA·가중치는 포함되지 않는다.
로컬 원본 결과가 있는 경우 snapshot은 다음 명령으로 재생성할 수 있다.

```bash
python scripts/export_testset_snapshot.py --out examples/testset/cohort_snapshot_2026-09-19.json
```

## 새 WT–변이 비교 실행

`batch_matched_wt_mutants.tsv`는 동일 표적군마다 WT 1개와 변이 20개를 넣은 **새 실행용 입력**이다.
기존 결과를 이 조건으로 이미 검증했다는 뜻이 아니다. WT와 변이 모두 reference를 비워
순수한 구조 신뢰도 비교에 사용하며, WT 실험 구조를 변이체 실험 구조로 오인하지 않도록 했다.

```bash
./run.sh --batch examples/testset/batch_matched_wt_mutants.tsv \
  --name matched_wt_mutants_s42 --msa cache --samples 3 --seed 42 --steps 200 --recycles 3
```

위 수치는 재현 가능한 시작 설정이지 충분한 예측 반복 수가 통계적으로 검증되었다는 뜻은 아니다.
cache 최초 생성도 외부 서버에 서열을 보낸다. 독립적인 paired MSA를 제공하지 않는 체인별 cache
정책을 WT와 변이에 동일하게 적용한다. seed를 늘릴 때는 WT와 변이를 함께 새 이름으로 실행한다.
모델별 pose·epitope·점수 변동을 확인하고, 반복 예측을 생물학적 반복으로 세지 않는다.
9개 표적군 안의 변이 180개를 독립적인 항원 180종으로 해석하지 않는다.
Δ는 구조 신뢰도 차이이며 Kd·kon·koff·결합 에너지 변화가 아니다.

## 보관 자료

- `testset.tsv`: 과거 WT 10종의 입력 목록. 현재 9종 코호트의 기준 파일이 아니다.
- `archive/2026-09-19/testset_mutants_legacy200.tsv`: 과거 200행 변이 목록. 갱신된 180행 목록으로 대체되었다.
- `8emz/`: 역사적 입력. 최신 manifest 밖이지만 이번 검토에서 결과 무효가 입증된 것은 아니다.
- 로컬 `outputs/batch_testset_mutants/archive/2026-09-19/analysis_summary_legacy200.json`:
  변이 수와 일부 점수가 현재 결과와 다른 구 요약. 원본 예측은 삭제하지 않았다.

논문·발표·협업에는 코호트 이름, snapshot 해시, 계산 조건을 함께 남기고 다른 시점의 표를 섞지 않는다.
