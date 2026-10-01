#!/bin/bash
# usage: run_plan.sh <job> <gpu> <plan.py hydra args...>
# 산출물: $DINO_RUNS/<job>/ (hydra run dir: logs.json, plan_targets.pkl, 그림·영상), 로그: $DINO_RUNS/<job>.log
# 체크포인트 루트는 CKPT_BASE (기본 $DINO_CKPT = 공개 체크포인트, 직접 학습한 모델은 $DINO_TRAIN).
# 본문을 함수로 감싼다: bash 는 함수 정의를 끝까지 읽은 뒤 실행하므로, 도는 중에 이 파일이 바뀌어도 영향이 없다
main() {
  HERE=$(cd "$(dirname "$0")" && pwd)
  source "$HERE/../env.sh"
  NAME=$1; GPU=$2; shift 2
  # GPU 환경에 맞춘 옵션(메모리·속도)은 환경별 브랜치의 이 파일에서 export 한다 (AGENTS.md §1)
  # [hanbin5/local] 24 GB GPU 용 기본값 (repro/LOCAL.md). 끄려면 0 으로 export 한다
  export DINO_WM_SDPA=${DINO_WM_SDPA:-1}                      # predictor attention 을 fp32 SDPA 로 (상대오차 ≤ 1.3e-6)
  export DINO_WM_SKIP_SOLVED=${DINO_WM_SKIP_SOLVED:-1}        # MPC 에서 이미 성공한 에피소드는 CEM 을 건너뛴다
  export DINO_WM_ROLLOUT_CHUNK=${DINO_WM_ROLLOUT_CHUNK:-150}  # CEM 롤아웃 300 샘플을 150 씩 (결과 같음)
  export DINO_WM_GD_CHUNK=${DINO_WM_GD_CHUNK:-8}              # GD 순전파·역전파를 에피소드 8개씩 (그래디언트 같음)
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  OUT=$DINO_RUNS/$NAME
  if [ -f "$OUT/DONE" ]; then echo "$NAME already done"; return 0; fi
  mkdir -p "$OUT"
  cd "$DINO_WM"
  echo "START $(date -Is) pid=$$ gpu=$GPU branch=$(git -C "$DINO_WM" rev-parse --abbrev-ref HEAD 2>/dev/null) commit=$(git -C "$DINO_WM" rev-parse --short HEAD 2>/dev/null) env=[$(env | grep -E '^DINO_WM_' | sort | tr '\n' ' ')] args=$*" > "$OUT/run_info.txt"
  CUDA_VISIBLE_DEVICES=$GPU python plan.py "$@" ckpt_base_path=${CKPT_BASE:-$DINO_CKPT} hydra.run.dir=$OUT > "$DINO_RUNS/$NAME.log" 2>&1
  RC=$?
  echo "END $(date -Is) rc=$RC" >> "$OUT/run_info.txt"
  [ $RC -eq 0 ] && touch "$OUT/DONE"
  return $RC
}
main "$@"; exit $?
