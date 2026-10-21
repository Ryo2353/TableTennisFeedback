r"""
検出がどれだけ連続しているかを調べる(撮り直し不要。今ある rec フォルダで実行できる)

使い方:
    python check_detection_continuity.py <recフォルダ>
    例) python check_detection_continuity.py ..\..\data\captures\20261016_xxxxxx\rec01

分かること:
    - 各カメラで、全コマのうち何%で検出できたか
    - 検出が何コマ続いたか(連続の長さ)と、途切れた長さ
    - 2台とも検出できたコマ(=組にできるコマ)の数
"""
import json
import os
import sys
from collections import Counter


def load_dets(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):           # 念のため、辞書形式にも対応
        data = data.get("detections", list(data.values()))
    return data


def runs(sorted_deltas):
    """連続して検出できたコマの長さ一覧と、途切れた長さ一覧を返す"""
    if not sorted_deltas:
        return [], []
    run_lens, gaps = [], []
    cur = 1
    for a, b in zip(sorted_deltas, sorted_deltas[1:]):
        if b == a + 1:
            cur += 1
        else:
            run_lens.append(cur)
            gaps.append(b - a - 1)
            cur = 1
    run_lens.append(cur)
    return run_lens, gaps


def bucket(values, edges=(1, 2, 3, 5, 10, 20, 50)):
    """値を「1」「2」「3〜4」…のような区間に分けて数える"""
    c = Counter()
    for v in values:
        label = None
        for i, e in enumerate(edges):
            nxt = edges[i + 1] if i + 1 < len(edges) else None
            if nxt is None or v < nxt:
                if v >= e:
                    label = f"{e}" if (nxt is None or nxt == e + 1) else f"{e}〜{nxt - 1}"
                    if nxt is None:
                        label = f"{e}以上"
                    break
        c[label or f"{v}"] += 1
    order = []
    for i, e in enumerate(edges):
        nxt = edges[i + 1] if i + 1 < len(edges) else None
        order.append(f"{e}以上" if nxt is None else (f"{e}" if nxt == e + 1 else f"{e}〜{nxt - 1}"))
    return [(k, c[k]) for k in order if c[k]]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rec = sys.argv[1]
    info_path = os.path.join(rec, "session_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        print(f"縮小率(scale): {info.get('scale', '記録なし')}")

    deltas = {}
    for cam in ("camera_a", "camera_b"):
        dets = load_dets(os.path.join(rec, cam, "detections.json"))
        ds = sorted({d["delta"] for d in dets if d.get("delta") is not None})
        deltas[cam] = ds
        total = info.get(cam, {}).get("captured")
        print(f"\n=== {cam} ===")
        if total:
            print(f"全コマ {total} のうち検出 {len(ds)} コマ ({100 * len(ds) / total:.1f}%)")
        else:
            print(f"検出 {len(ds)} コマ")
        r, g = runs(ds)
        if r:
            print(f"連続して検出できた長さ: 最長 {max(r)} コマ、平均 {sum(r) / len(r):.1f} コマ、{len(r)} 回に分かれている")
            print("  連続の長さの分布:", ", ".join(f"{k}コマ:{n}回" for k, n in bucket(r)))
        if g:
            print("  途切れた長さの分布:", ", ".join(f"{k}コマ:{n}回" for k, n in bucket(g)))

    both = sorted(set(deltas["camera_a"]) & set(deltas["camera_b"]))
    only_a = len(set(deltas["camera_a"]) - set(both))
    only_b = len(set(deltas["camera_b"]) - set(both))
    print("\n=== 2台の組 ===")
    print(f"2台とも検出: {len(both)} コマ / Aだけ: {only_a} / Bだけ: {only_b}")
    r, _ = runs(both)
    if r:
        print(f"2台とも検出が続いた長さ: 最長 {max(r)} コマ、{len(r)} 回に分かれている")
        print("  分布:", ", ".join(f"{k}コマ:{n}回" for k, n in bucket(r)))
    print("\n目安: 1球が台の上を飛ぶ時間(約0.3〜0.5秒)なら、227fpsで約70〜110コマ続けて写るはず")


if __name__ == "__main__":
    main()