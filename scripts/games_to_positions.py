#!/usr/bin/env python
"""对局记录 jsonl(eval_anchor --dump-games 格式) → 局面记录 jsonl(label_service 格式)。

输入行: {"opening_fen", "moves":[ucci...], "result":±1/0(红方视角), "net_side", ...}
输出行: {"fen", "ply", "gid", "outcome_red"} —— 与 gen_positions.py 同构,可直接
投入 data/dagger/queue_p0 给 label_service 消费。

跳过非法/无法复盘的对局(计数告警);FEN 不在此处去重(label_service 有跨 shard seen)。
"""
from __future__ import annotations
import argparse, glob, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from xqai._xqcore import Position
from xqai.arena import _ucci_to_move


def convert_file(path: str, out_dir: str, gid_prefix: str) -> tuple[int, int]:
    rows, bad = [], 0
    with open(path) as fh:
        for li, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
                pos = Position()
                if g.get("opening_fen"):
                    pos = Position.from_fen(g["opening_fen"])
                gid = f"{gid_prefix}:{li}"
                outcome = int(g["result"])
                rows.append({"fen": pos.fen(), "ply": 0, "gid": gid,
                             "outcome_red": outcome})
                ok = True
                for ply, mv in enumerate(g["moves"], start=1):
                    m = _ucci_to_move(mv)
                    if m not in pos.legal_moves():
                        bad += 1
                        ok = False
                        break
                    pos.push(m)
                    rows.append({"fen": pos.fen(), "ply": ply, "gid": gid,
                                 "outcome_red": outcome})
                if not ok:
                    # 丢弃该局已收集的行(复盘断裂后 FEN 不可信)
                    rows = [r for r in rows if r["gid"] != gid]
            except Exception:
                bad += 1
    if rows:
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, f"pos_{base}.jsonl")
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, out)
    return len(rows), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True, help="对局 jsonl glob")
    ap.add_argument("--out-dir", default="data/dagger/queue_p0")
    ap.add_argument("--done-dir", default="data/dagger/games_done",
                    help="转换完的原文件挪到这里(幂等;空串=原地保留)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.done_dir:
        os.makedirs(a.done_dir, exist_ok=True)
    tot, totbad = 0, 0
    for f in sorted(glob.glob(a.in_glob)):
        n, bad = convert_file(f, a.out_dir, gid_prefix=os.path.basename(f))
        tot += n; totbad += bad
        if a.done_dir:
            os.replace(f, os.path.join(a.done_dir, os.path.basename(f)))
        print(f"[g2p] {os.path.basename(f)}: {n} positions ({bad} bad)", flush=True)
    print(f"[g2p] TOTAL {tot} positions, {totbad} bad games", flush=True)


if __name__ == "__main__":
    main()
