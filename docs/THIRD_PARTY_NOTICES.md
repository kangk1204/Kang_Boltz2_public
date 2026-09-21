# 서드파티 고지 (Third-Party Notices)

검증 환경에서 확인한 라이선스 기준입니다 (2026-09-18).

## 이 저장소가 직접 포함하는 구성요소

| 구성요소 | 위치 | 라이선스 |
|---|---|---|
| Mol* 뷰어 번들 v5.11.0 | `assets/molstar.js`, `assets/molstar.css` | MIT |
| DunbrackLab IPSAE (`ipsae.py` v4) | `scripts/vendor/ipsae_official.py` | MIT (원본 헤더 유지) |
| PDB 예제 구조/서열 | `examples/`, `inputs/` | RCSB PDB 공개 데이터 |

`setup.sh`가 검증하는 Mol* v5.11.0 자산 SHA-256은 다음과 같습니다.

- `assets/molstar.js`: `7fad5561c74bc900930fb57d6ab028d1aafdda82223a901bf932b1098e84f1f3`
- `assets/molstar.css`: `5b68ceb6d3642549b4e9b2c071e58e41b98a5350ae269180587b39da86925d55`

IPS AE 파일에는 로컬 패치 1곳이 있습니다: confidence JSON 에 `pair_chains_iptm` 키가 없을 때
원본이 KeyError 로 종료하던 부분을 0점 처리로 방어한 것(수치 로직 불변, 주석으로 표기).

## 설치 스크립트가 설치만 하고 재배포하지 않는 구성요소

| 패키지 | 라이선스 |
|---|---|
| Boltz (및 Boltz-2 가중치) | MIT |
| PyTorch | Apache-2.0 계열 (BSD-2/3, MIT, BSL-1.0 등 포함) |
| cuequivariance (NVIDIA) | Apache-2.0 |
| DockQ | MIT |
| ANARCI | BSD-3-Clause |
| gemmi | MPL-2.0 |
| matplotlib | Matplotlib License (PSF 기반) |
| numpy | BSD-3-Clause |
| PyYAML | MIT |

## 외부 서비스

- **ColabFold MMseqs2 MSA 서버**(`api.colabfold.com`): 공용 서비스입니다. 논문용 사용 시
  Mirdita et al., *Nature Methods* 2022 인용이 필요합니다. 대량 배치에서는 서버 부담을 고려하고
  `--msa cache` 로 재사용하거나 `--msa empty`(품질↓)를 검토하세요.
- HuggingFace: 모델 가중치 최초 다운로드 시 사용됩니다.

## 인용

이 파이프라인은 아래 도구를 사용합니다. 방법 기술 시 출처를 함께 인용하세요.
Boltz-2(Passaro et al., 2025) · ipSAE(Dunbrack lab, 2025) ·
pDockQ(Bryant, Pozzati & Elofsson, 2022) · pDockQ2(Zhu et al., 2023) ·
LIS(Kim et al., 2024 preprint) ·
DockQ(Mirabello & Wallner, 2024) · Mol*(Sehnal et al., 2021) · ANARCI(Dunbar et al., 2016) ·
ColabFold(Mirdita et al., 2022)

## 이 저장소 자체의 라이선스

아직 지정되지 않았습니다. 외부 배포 전에 연구실 방침에 맞는 라이선스(예: 내부용 독점,
또는 MIT/BSD)를 추가하는 것을 권장합니다.
