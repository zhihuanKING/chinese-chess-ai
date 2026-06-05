#!/usr/bin/env bash
# 安全杀进程：按模式杀，但绝不杀调用者自身/父进程/本脚本（杜绝 pkill -f 自杀）。
# 用法: bash scripts/safe_kill.sh <pattern> [<pattern>...]
# 例:   bash scripts/safe_kill.sh train_distributed.py eval_loop.py
self=$$; parent=$PPID
selfargs="safe_kill"   # 本脚本命令行特征，用于排除
for pat in "$@"; do
  for pid in $(pgrep -f -- "$pat" 2>/dev/null); do
    # 排除：本脚本、父进程、命令行里含 safe_kill 的（即调用链自身）
    [ "$pid" = "$self" ] && continue
    [ "$pid" = "$parent" ] && continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$selfargs" && continue
    kill -9 "$pid" 2>/dev/null && echo "killed $pid ($pat)"
  done
done
echo "safe_kill done."
