# -*- coding: utf-8 -*-
r"""
撮りながら検出(capture_live_detect.py)の detections.json に残っている「候補」を全部、時間に沿って描く。
2球が同時に写っている時間に、候補が2本の線(2つの軌道)に分かれていれば、
2球とも検出できていた(=今のデータで複数ボールの追跡を作れる)ことが分かる。

使い方(プロジェクトのいちばん上のフォルダで):
    python src\tools\plot_candidates.py <録画フォルダ> [--t0 5.5 --t1 7.0]
例:
    python src\tools\plot_candidates.py data\captures\20261023\20261023_182206\rec02

出力(録画フォルダの中):
    candidates.png   上からカメラAの u, v、カメラBの u, v。横軸は録画開始からの時間(秒)
                     灰色の点 = 候補すべて / 色つきの点 = 各コマで一番大きい候補(今の検出位置)
画面には、候補が2つ以上あったコマの数を出す。
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import font_manager

# 日本語が出せるフォントのうち、このPCにあるものだけを使う(ないフォントを指定すると警告が大量に出るため)
_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic") if f in _have] + ["sans-serif"]

FPS = 227.0


def load(rec, cam):
    with open(os.path.join(rec, cam, "detections.json"), encoding="utf-8") as f:
        return [d for d in json.load(f) if d.get("bid") is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec_dir")
    ap.add_argument("--t0", type=float, default=None, help="描く範囲の始め(秒)")
    ap.add_argument("--t1", type=float, default=None, help="描く範囲の終わり(秒)")
    args = ap.parse_args()

    data = {cam: load(args.rec_dir, cam) for cam in ("camera_a", "camera_b")}
    base = min(min(d["bid"] for d in ds) for ds in data.values() if ds)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for k, (cam, ds) in enumerate(data.items()):
        name = "A" if cam == "camera_a" else "B"
        multi = sum(1 for d in ds if len(d.get("candidates", [])) >= 2)
        print(f"カメラ{name}: 検出 {len(ds)} コマ、そのうち候補が2つ以上 {multi} コマ")
        for j, key in enumerate(("u", "v")):
            ax = axes[2 * k + j]
            ts, vs, bt, bv = [], [], [], []
            for d in ds:
                t = (d["bid"] - base) / FPS
                if (args.t0 is not None and t < args.t0) or (args.t1 is not None and t > args.t1):
                    continue
                for c in d.get("candidates", []):
                    ts.append(t)
                    vs.append(c[j])
                bt.append(t)
                bv.append(d[key])
            ax.scatter(ts, vs, s=10, c="#9CA3AF", label="候補すべて")
            ax.scatter(bt, bv, s=4, c="#2563C9" if name == "A" else "#B5581A", label="一番大きい候補(今の検出位置)")
            ax.set_ylabel(f"カメラ{name} {key} (画素)")
            ax.grid(alpha=0.3)
            if k == 0 and j == 0:
                ax.legend(loc="upper right")
    axes[-1].set_xlabel("録画開始からの時間 (秒)")
    fig.tight_layout()
    out = os.path.join(args.rec_dir, "candidates.png")
    fig.savefig(out, dpi=110)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()