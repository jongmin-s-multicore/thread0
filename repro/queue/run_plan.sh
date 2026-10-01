#!/bin/bash
# usage: run_plan.sh <job> <gpu> <plan.py hydra args...>
# 산출물: $DINO_RUNS/<job>/ (hydra run dir: logs.json, plan_targets.pkl, 그림·영상), 로그: $DINO_RUNS/<job>.log
# 체크포인트 루트는 CKPT_BASE (기본 $DINO_CKPT = 공개 체크포인트, 직접 학습한 모델은 $DINO_TRAIN).
# 본문을 함수로 감싼다: bash 는 함수 정의를 끝까지 읽은 뒤 실행하므로, 도는 중에 이 파일이 바뀌어도 영향이 없다
main() {
  HERE=$(cd "$(dirname "$0")" && pwd)
  source "$HERE/../env.sh"
  NAME=$1; GPU=$2; shift 2
  # 재현 패치의 옵션 (SETTINGS.md §6). 끄려면 0 으로 export 한다
  export DINO_WM_SDPA=${DINO_WM_SDPA:-1}                      # predictor attention 을 fp32 SDPA 로 (상대오차 1e-6)
  export DINO_WM_SKIP_SOLVED=${DINO_WM_SKIP_SOLVED:-1}        # MPC 에서 이미 성공한 에피소드는 CEM 을 건너뛴다
  # 메모리 조정 (결과는 같다). GPU 메모리에 맞춰 머신별로 준다 — 24 GB 에서는 예: 150, 8
  export DINO_WM_ROLLOUT_CHUNK=${DINO_WM_ROLLOUT_CHUNK:-0}    # CEM 롤아웃을 N 샘플씩 (0 = 한 번에)
  export DINO_WM_GD_CHUNK=${DINO_WM_GD_CHUNK:-0}              # GD 순전파·역전파를 에피소드 N 개씩 (0 = 한 번에)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  OUT=$DINO_RUNS/$NAME
  if [ -f "$OUT/DONE" ]; then echo "$NAME already done"; return 0; fi
  mkdir -p "$OUT"
  cd "$DINO_WM"
  echo "START $(date -Is) pid=$$ gpu=$GPU chunk=$DINO_WM_ROLLOUT_CHUNK sdpa=$DINO_WM_SDPA skip_solved=$DINO_WM_SKIP_SOLVED gdchunk=$DINO_WM_GD_CHUNK args=$*" > "$OUT/run_info.txt"
  CUDA_VISIBLE_DEVICES=$GPU python plan.py "$@" ckpt_base_path=${CKPT_BASE:-$DINO_CKPT} hydra.run.dir=$OUT > "$DINO_RUNS/$NAME.log" 2>&1
  RC=$?
  echo "END $(date -Is) rc=$RC" >> "$OUT/run_info.txt"
  [ $RC -eq 0 ] && touch "$OUT/DONE"
  return $RC
}
main "$@"; exit $?
