# AGENTS.md — 저장소 운영 규약

사람과 코딩 에이전트가 이 저장소에서 작업할 때 따르는 규약이다. **이 저장소는 공개(public)다** — 올린 것은 전부 공개된다.
([inner-king/policy](https://github.com/inner-king/policy) 의 규약을 이 저장소에 맞게 옮겼다.)

## 1. 무엇을 어디에 두나

이 저장소는 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) 의 fork 이고, DINO-WM 재현 하나를 위한 것이다. 재현에 쓰는 것은 모두 main 에 둔다.

| 위치 | 두는 것 | 예 |
|---|---|---|
| `main` 루트 | upstream 코드와 그 위의 재현 수정 (커밋, 수정 블록마다 `[repro]` 주석) | `models/` `planning/` `train.py` |
| `main` 의 `repro/` | 환경 설치·점검, 작업 큐, 평가, 실행 설정 기록, 환경 목록, 실험 문서 | `repro/setup/` `repro/queue/` `repro/runs/` |
| `.claude/` (커밋하지 않는다) | 머신별 설정(작업 루트, GPU·메모리 조건, 청크 크기, 작업별 overlay)과 그 머신의 실행 이력·메모 | `.claude/env.local.sh` |
| 개인 브랜치 `<GitHub ID>/<주제>` | 아직 공유하지 않는 각자의 변형 실험 (머신별 값은 브랜치가 아니라 `.claude/`) | — |
| 이슈 | 실험 내용: 결과, 해석, 그림, 다음에 할 것 | — |

- 실험 결과(점수, 학습 곡선, 분석)는 커밋하지 않고 이슈로 올린다. 수치 요약 JSON 이나 그림은 이슈에 첨부한다.
- main 에는 PR 없이 직접 커밋한다. 개인 브랜치의 변경은 재현에 쓰게 되면 main 으로 옮긴다.
- upstream 파일을 고칠 때는 수정 블록마다 `[repro]` 주석을 달고, 주제별로 커밋을 나눈다. upstream 에 돌려줄 만한 수정(버그·메모리)은 PR 을 검토한다.
- 남의 브랜치에는 push 하지 않는다.

## 2. 재현 디렉토리 구조

```
repro/
├── README.md        무엇을 하는가, 구성, 환경, 실행 방법
├── SETTINGS.md      모델·데이터·하이퍼파라미터·평가 프로토콜 (수치는 run.yaml·코드·로그에서 가져온다)
├── env.sh setup/ queue/ jobs/ eval/ make_run_yaml.py   코드
├── runs/{train,eval}/<run-id>/run.yaml   실행 하나의 설정 기록 (결과 필드는 넣지 않는다)
└── requirements/    환경 목록 (pip freeze, conda list)
```

- run-id 형식: `<모델>-<초기값>[-<총량>][-at<체크포인트>][-<변형>]`. 공개 체크포인트는 `<초기값>=released` 로 쓰고 `<총량>` 을 뺀다.
  예: `dinowm_rope-dinov2s14-100ep`, `dinowm_rope-dinov2s14-100ep-at100-mpccem`, `dinowm_pusht-released-at2-mpccem`.
- run.yaml 에는 무엇을 어떻게 돌렸는지(체크포인트, 에폭·스텝, 배치, 옵티마이저, 실행 명령, 평가 프로토콜)만 적는다. 결과 수치와 머신별 실행 이력은 넣지 않는다.
- 산출물(로그, `logs.json`, 체크포인트, 그림·영상)은 레포 밖 `$DINO_WORK` 에 생긴다.

## 3. 이슈 작성

- 제목에 요지를 쓴다. 예: `[dinowm] 공개 체크포인트 planning — PointMaze·PushT·Wall 50 에피소드`.
- 본문에 재현 정보를 적는다: 커밋 해시, run-id, 해당 `repro/SETTINGS.md` 절.
- 결과를 낸 코드가 main 의 현재 코드와 다르면 그 차이를 적는다.
- 그림·표는 이슈에 첨부한다. 영상은 필요한 구간만 올린다.

## 4. 커밋하면 안 되는 것

| 대상 | 이유 |
|---|---|
| 데이터셋, 공개·학습 체크포인트 (`*.pth` 등) | 용량, 재배포. OSF 에서 받는다 — README 에 받는 곳만 적는다 |
| 다른 upstream 코드 사본 (PyFleX, DINOv2) | 고정 커밋을 받아 쓴다 (setup 스크립트, `models/dino.py`) |
| 로그·영상·planning 산출물(`logs.json`, `plan_targets.pkl`, mp4, png) | 절대경로가 들어간다. 결과는 이슈로 |
| 사설 IP·내부 호스트명·홈 경로(`/home/<user>`)·계정명 | 공개 저장소 |
| 토큰·API 키·비밀번호, `.env` | — |
| 머신별 설정·실행 이력 (GPU 번호 고정, 메모리 조건, 재시도·OOM 기록) | 머신마다 다르다. `.claude/` 에 둔다 |

커밋 전에 확인한다:

```bash
git diff --cached --stat                  # 큰 파일·산출물이 섞이지 않았나
git diff --cached | grep -nE '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.|192\.168\.|/home/|hf_[A-Za-z0-9]{10,}|gh[po]_|sk-|password'
```

커밋 작성자 이메일은 GitHub noreply 주소(`<id>+<login>@users.noreply.github.com`)를 쓴다.

## 5. 코드·문서 작성

- 경로는 한 곳(`repro/env.sh`)에서 정하고 환경변수로 덮어쓸 수 있게 한다. 머신마다 다른 값(GPU 번호, 설치 경로, 메모리 조건)은 코드에 박지 않고 `.claude/env.local.sh` 와 overlay 파일에 둔다.
- 문서는 한국어로, 수사 없이 기술 용어로 쓴다. 수치에는 출처(run.yaml, 코드, 로그)가 있어야 하고, 확인하지 않은 절차는 "돌려 보지 않았다"고 적는다.
- 에이전트는 사람이 요청하지 않은 브랜치 삭제·강제 push·히스토리 수정을 하지 않는다.
