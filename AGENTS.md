# AGENTS.md: 저장소 운영 규약

사람과 코딩 에이전트가 이 저장소에서 작업할 때 따르는 규약이다. 이 저장소는 공개 저장소다. 올린 내용은 모두 공개된다.
[inner-king/policy](https://github.com/inner-king/policy) 의 규약을 이 저장소에 맞게 옮겼다.

## 1. 무엇을 어디에 두나

이 저장소는 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) 의 fork 이고 DINO-WM 재현에 쓴다. GPU 환경이 다른 사람들이 같이 쓰므로 main 에는 환경과 무관한 내용만 둔다.

| 위치 | 두는 것 | 예 |
|---|---|---|
| `main` 루트 | upstream 코드와 그 위의 공용 수정. 공용 수정은 어느 환경에서나 틀린 동작을 고치는 버그 수정과 공용 기능이다. 커밋하고 수정 블록마다 `[repro]` 주석을 단다 | `models/dino.py` `train.py` |
| `main` 의 `repro/` | 환경 설치·점검, 작업 큐, 벤치마크 작업 목록(프로토콜), 평가·요약, 환경 목록, 실험 문서 | `repro/setup/` `repro/queue/` `repro/jobs/` |
| 환경별 브랜치 `<GitHub ID>/<환경 또는 주제>` | 자기 GPU 환경에 맞춘 구현(메모리·속도를 위한 수정, 러너 기본값), 그 환경에서 한 실행의 기록(`repro/runs/`), 그 환경의 측정·가이드 문서. 메모리·속도를 위한 수정은 결과에 영향이 없어도 여기에 둔다 | `hanbin5/local` |
| `.claude/` (커밋하지 않는다) | 머신 하나의 값(작업 루트, GPU 번호·메모리 조건, 작업별 overlay)과 그 머신의 실행 이력·메모 | `.claude/env.local.sh` |
| 이슈 | 실험 내용: 결과, 해석, 그림, 다음에 할 것 | – |

- 실험 결과(점수, 학습 곡선, 분석)는 커밋하지 않고 이슈로 올린다. 수치 요약 JSON 이나 그림은 이슈에 첨부한다. 예외로 루트 `README.md` 에는 대표 결과 요약 표를 둔다. 표에는 출처 이슈와 평가 영상 폴더 링크를 함께 적는다. 표의 수치는 이슈·로그와 같게 유지한다.
- main 에는 PR 없이 커밋한다. 모든 환경에 해당하는 내용만 넣는다. 브랜치에서 나온 수정이 어느 환경에서나 틀린 동작을 고치는 수정이면 main 으로 옮긴다. 메모리·속도용 수정은 팀이 합의하기 전까지 브랜치에 둔다.
- 환경 브랜치는 `git merge main` 으로 main 의 변경을 받는다. 이슈에 적은 커밋 해시가 사라지지 않게 공개된 브랜치는 rebase·force push 하지 않는다. rebase·force push 를 해야 하면 이슈에 적은 커밋에 태그를 먼저 단다. 남의 브랜치에는 push 하지 않는다.
- upstream 파일을 고칠 때는 수정 블록마다 `[repro]` 주석을 달고, 주제별로 커밋을 나눈다. upstream 에 반영할 만한 수정은 PR 을 검토한다.

## 2. 재현 디렉토리 구조

```
repro/
├── README.md        [main]   무엇을 하는가, 구성, 환경, 실행 방법
├── SETTINGS.md      [main]   모델·데이터·하이퍼파라미터·평가 프로토콜 (수치는 코드·로그에서 가져온다)
├── env.sh setup/ queue/ jobs/ eval/   [main]   공용 코드
├── requirements/    [main]   환경 목록 (pip freeze, conda list)
├── LOCAL.md         [브랜치] 그 환경에 맞춘 수정·러너 기본값·측정
├── make_run_yaml.py [브랜치] 실행 기록 생성
└── runs/{train,eval}/<run-id>/run.yaml   [브랜치] 실행 하나의 설정 기록 (결과 필드는 넣지 않는다)
```

- run-id 형식: `<모델>-<초기값>[-<총량>][-at<체크포인트>][-<변형>]`. 공개 체크포인트는 `<초기값>=released` 로 쓰고 `<총량>` 을 뺀다.
  예: `dinowm_rope-dinov2s14-100ep`, `dinowm_rope-dinov2s14-100ep-at100-mpccem`, `dinowm_pusht-released-at2-mpccem`.
- run.yaml 에는 무엇을 어떻게 실행했는지(체크포인트, 에폭·스텝, 배치, 옵티마이저, 실행 명령, 평가 프로토콜)만 적는다. 결과 수치와 머신별 실행 이력은 넣지 않는다.
- 산출물(로그, `logs.json`, 체크포인트, 그림·영상)은 레포 밖 `$DINO_WORK` 에 생긴다.

## 3. 이슈 작성

- 제목에 요지를 쓴다. 예: `[dinowm] 공개 체크포인트 planning: PointMaze·PushT·Wall 50 에피소드`.
- 본문에 재현 정보를 적는다. 재현 정보는 브랜치와 커밋 해시, run-id, 해당 `repro/SETTINGS.md`·`repro/LOCAL.md` 절이다.
- 결과를 낸 코드가 그 브랜치의 현재 코드와 다르면 그 차이를 적는다.
- 그림·표는 이슈에 첨부한다. 영상은 필요한 구간만 올린다.

## 4. 커밋하면 안 되는 것

| 대상 | 이유 |
|---|---|
| 데이터셋, 공개·학습 체크포인트 (`*.pth` 등) | 용량, 재배포. OSF 에서 받는다. README 에는 받는 곳만 적는다 |
| 다른 upstream 코드 사본 (PyFleX, DINOv2) | 고정 커밋을 받아 쓴다 (setup 스크립트, `models/dino.py`) |
| 로그·영상·planning 산출물(`logs.json`, `plan_targets.pkl`, mp4, png) | 절대경로가 들어간다. 결과는 이슈로 올린다 |
| 사설 IP·내부 호스트명·홈 경로(`/home/<user>`)·계정명 | 공개 저장소 |
| 토큰·API 키·비밀번호, `.env` | – |
| 머신별 설정·실행 이력 (GPU 번호 고정, 메모리 조건, 재시도·OOM 기록) | 머신마다 다르다. `.claude/` 에 둔다 |

커밋 전에 확인한다:

```bash
git diff --cached --stat                  # 큰 파일·산출물이 섞이지 않았나
git diff --cached | grep -nE '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.|192\.168\.|/home/|hf_[A-Za-z0-9]{10,}|gh[po]_|sk-|password'
```

커밋 작성자 이메일은 GitHub noreply 주소(`<id>+<login>@users.noreply.github.com`)를 쓴다.

## 5. 코드·문서 작성

- 경로는 한 곳(`repro/env.sh`)에서 정하고 환경변수로 덮어쓸 수 있게 한다. 머신마다 다른 값(GPU 번호, 설치 경로, 메모리 조건)은 코드에 하드코딩하지 않고 `.claude/env.local.sh` 와 overlay 파일에 둔다. GPU 환경에 맞춘 코드는 환경 브랜치에 둔다.
- 문서는 한국어로, 수사 없이 기술 용어로 쓴다. 루트 `README.md` 만 영어로 쓴다. 수치에는 출처(run.yaml, 코드, 로그)가 있어야 한다. 확인하지 않은 절차는 "실행해 보지 않았다"고 적는다.
- 에이전트는 사람이 요청하지 않은 브랜치 삭제·강제 push·히스토리 수정을 하지 않는다.

## 6. Colab 에서 돌리기

로컬 GPU 가 차 있거나 더 큰 GPU 가 필요할 때는 Google Colab 을 **`colab` CLI** (`google-colab-cli`, 없으면 `uv tool install google-colab-cli`) 로 쓴다. 세션 생성·코드 실행·파일 전송을 터미널에서 다 할 수 있으므로, 사람에게 브라우저로 Colab 을 열어 달라고 하지 않는다.

- **사용법은 CLI 안에 있다**: `colab skill` (에이전트용 요약), `colab readme`, `colab help <명령>`. 처음 쓸 때 `colab skill` 을 먼저 읽는다.
- **GPU**: `colab new -s <세션 이름> --gpu G4`. `G4` 가 NVIDIA RTX PRO 6000 (Blackwell, 96 GB) 이다. 그 밖에 `T4` `L4` `A100` `H100`. 이름을 틀리면 조용히 A100 이 잡히므로 만든 뒤 `colab status -s <이름>` 과 VM 의 `nvidia-smi` 로 확인한다.
- **PyTorch**: Blackwell 은 PyTorch 2.7 이상(CUDA 12.8)이 필요하다. 이 레포의 고정 환경(`environment.yaml`, torch 2.3.0+cu121)은 G4 에서 돌지 않는다. Colab VM 에 깔려 있는 Python·PyTorch 를 쓰고 필요한 패키지만 더 설치한다. `plan.py`·`train.py` 는 mujoco-py 를 import 하므로 VM 에서는 그것을 거치지 않는 스크립트를 쓴다 (예: `repro/colab/predictor_long.py`).
- **실행은 사용자 계정의 tmux 세션 안에서 한다**: `tmux new -d -s <이름> '<명령> 2>&1 | tee <로그 파일>'`. 에이전트의 백그라운드 작업으로 띄우지 않는다. 사람이 `tmux attach -t <이름>` 으로 볼 수 있고 에이전트 세션이 끝나도 계속 돈다.
- **코드 실행**: `colab exec -s <이름> -f script.py` (로컬 파일을 VM 의 커널에서 실행, 작업 디렉토리 `/content`). 커널 상태는 호출 사이에 유지된다. 파일은 `colab upload` / `colab download`. `colab drivemount`·`colab auth`·대화형 `repl`/`console` 은 사람이 터미널에 있어야 하므로 에이전트가 쓰지 않는다.
- **데이터·체크포인트**: 공개 데이터는 VM 에서 직접 받고 (`repro/setup/download_acwm_phys.py` 등), 체크포인트는 필요한 가중치만 올린다. VM 은 세션을 끝내면 사라지므로 결과는 끝내기 전에 내려받는다.
- **비용**: 세션은 compute unit 을 쓴다. 시작 전후에 `colab usage` 로 잔액과 시간당 사용량을 보고 사람에게 알린다. 일이 끝나면 반드시 `colab stop -s <이름>` 하고 `colab sessions` 로 남은 세션이 없는지 확인한다.
