#!/bin/bash
# usage: kill_job.sh <job>  — 러너, 그 python, python 의 워커(SubprocVectorEnv 등)를 PID 로 끈다.
# (pkill -f 는 이 셸 자신의 명령줄과도 맞을 수 있어서 쓰지 않는다)
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
NAME=$1
for sh in $(ps -eo pid,args | awk -v n="$NAME" '$0 ~ "run_(plan|train|cmd).sh "n" " {print $1}'); do
  for py in $(ps -eo pid,ppid | awk -v p="$sh" '$2==p {print $1}'); do
    kill $py $(ps -eo pid,ppid | awk -v p="$py" '$2==p {print $1}') 2>/dev/null
  done
  kill "$sh" 2>/dev/null
done
