여기에 두 파일을 넣으면 됩니다.

  target.fasta     항원 서열 (여러 체인이면 레코드 여러 개 -> 체인 A, C, D ...)
  nanobody.fasta   나노바디(VHH) 서열 1개 -> 체인 B

지금 들어 있는 내용은 예제(라이소자임 + cAb-Lys3)입니다. 실제 타겟 서열로 바꿔서 쓰세요.
정답 복합체 구조(cif/pdb)가 있으면 --reference 로 넘기면 DockQ까지 계산됩니다.

실행:  ./run.sh --name <이름>
결과:  outputs/<이름>/report/index.html
