#!/bin/bash
# usage: run_cmd.sh <job> <gpu> <command...>   ($DINO_WM 에서 env 를 켠 채 실행. 인자의 $VAR 는 여기서 펼친다)
# 본문을 함수로 감싼다: bash 는 함수 정의를 끝까지 읽은 뒤 실행하므로, 도는 중에 이 파일이 바뀌어도 영향이 없다
main() {
  HERE=$(cd "$(dirname "$0")" && pwd)
  source "$HERE/../env.sh"
  NAME=$1; GPU=$2; shift 2
  OUT=$DINO_RUNS/$NAME
  if [ -f "$OUT/DONE" ]; then echo "$NAME already done"; return 0; fi
  mkdir -p "$OUT"
  cd "$DINO_WM"
  echo "START $(date -Is) pid=$$ gpu=$GPU cmd=$*" > "$OUT/run_info.txt"
  CUDA_VISIBLE_DEVICES=$GPU bash -c "$*" > "$DINO_RUNS/$NAME.log" 2>&1
  RC=$?
  echo "END $(date -Is) rc=$RC" >> "$OUT/run_info.txt"
  [ $RC -eq 0 ] && touch "$OUT/DONE"
  return $RC
}
main "$@"; exit $?
