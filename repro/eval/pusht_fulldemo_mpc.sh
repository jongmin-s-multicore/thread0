#!/bin/bash
# usage: pusht_fulldemo_mpc.sh <name> <gpu> <targets.pkl> [plan.py hydra args...]
# PushT MPC 를 데모 전체 목표(pusht_fulldemo_targets.py 가 만든 파일)로 돌린다. 설정은 벤치마크 작업 pusht_mpc 와 같고
# (plan_pusht.yaml, max_iter 10, repro/queue/run_plan.sh 의 환경변수) 목표만 goal_source=file 로 바꾼다.
# plan_dump_actions.py 를 거쳐 plan.py 를 돌리므로 계획한 행동(diag_actions.pkl)도 남는다.
# 산출물: $DINO_RUNS/<name>/, 로그: $DINO_RUNS/<name>.log
main() {
  HERE=$(cd "$(dirname "$0")" && pwd)
  source "$HERE/../env.sh"
  NAME=$1; GPU=$2; TARGETS=$(realpath "$3"); shift 3
  export DINO_WM_SDPA=${DINO_WM_SDPA:-1}
  export DINO_WM_SKIP_SOLVED=${DINO_WM_SKIP_SOLVED:-1}
  export DINO_WM_ROLLOUT_CHUNK=${DINO_WM_ROLLOUT_CHUNK:-150}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  OUT=$DINO_RUNS/$NAME
  if [ -f "$OUT/DONE" ]; then echo "$NAME already done"; return 0; fi
  mkdir -p "$OUT"
  N=$(python -c "import pickle,sys; print(len(pickle.load(open(sys.argv[1],'rb'))['state_0']))" "$TARGETS")
  set -- --config-name plan_pusht.yaml model_name=pusht planner.max_iter=10 ckpt_base_path=$DINO_CKPT \
    goal_source=file +goal_file_path=$TARGETS n_evals=$N n_plot_samples=$N "$@"
  echo "START $(date -Is) pid=$$ gpu=$GPU branch=$(git -C "$DINO_WM" rev-parse --abbrev-ref HEAD 2>/dev/null) commit=$(git -C "$DINO_WM" rev-parse --short HEAD 2>/dev/null) env=[$(env | grep -E '^DINO_WM_' | sort | tr '\n' ' ')] args=$*" > "$OUT/run_info.txt"
  CUDA_VISIBLE_DEVICES=$GPU python "$HERE/plan_dump_actions.py" "$OUT" "$@" > "$DINO_RUNS/$NAME.log" 2>&1
  RC=$?
  echo "END $(date -Is) rc=$RC" >> "$OUT/run_info.txt"
  [ $RC -eq 0 ] && touch "$OUT/DONE"
  return $RC
}
main "$@"; exit $?
