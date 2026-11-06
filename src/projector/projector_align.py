# -*- coding: utf-8 -*-
"""
プロジェクターの位置合わせ(設置するたびに1回行う)。

プロジェクターに4つの印を全画面で映し、キー操作で動かして、
卓球台(プリント)の計測範囲の4隅にぴったり合わせる。
合わせ終わったら保存すると、「卓球台の座標(mm) → プロジェクターの画素」の
変換(射影変換 H)が projector_transform.npz に保存される。
project_heatmap.py はこれを使って、ヒートマップを台の正しい位置に映す。

使い方:
    python projector_align.py --out results\\YYYYMMDD\\projector_transform.npz

    台の大きさは src/table_config.py の設定を使う(--table で指定もできる)。

    プロジェクターの位置・解像度は自動で調べる(screeninfo が入っていれば)。
    うまくいかないときは指定する:
    python projector_align.py --out ... --screen-x 1920 --screen-y 0 --width 1920 --height 1080
      --screen-x/--screen-y : プロジェクター画面の左上の位置(Windowsの「ディスプレイ設定」で
                              PCの画面の右に置いてあるなら、PCの画面の幅がscreen-x)

キー操作(映っている全画面ウィンドウを選んだ状態で):
    1 / 2 / 3 / 4      動かす印を選ぶ
                       1=原点角(給球機側エンドラインの左) 2=給球機側エンドラインの右
                       3=右のもう一方の角             4=左のもう一方の角
    矢印キー           選んだ印を大きく動かす(20画素ずつ。--coarse-step で変更可)→ 大まかに合わせる
    w / a / s / d      選んだ印を 上 / 左 / 下 / 右 に1画素動かす → 細かく合わせる
    g                  台の枠・センターラインのガイド表示の切り替え
                       (印を合わせる範囲の外の線は、画面に入る部分だけ映る)
    Enter              保存して終了
    Esc                保存せずに終了

※ 4隅の座標(mm)は visualize_heatmap.py と同じ卓球台座標系:
   X は幅方向、Y は長さ方向。研究室のプリントなら X: 0〜1165、Y: 0〜840

プロジェクターが計測範囲の全体を覆えないとき(--area):
    印を合わせる範囲を、計測範囲の一部にできる。印は「プリントの線の交点」など、
    台の上で場所がはっきり分かる点にすること。
      --area full    計測範囲全体(標準)
      --area left    センターラインより左半分
      --area right   センターラインより右半分
      --area x0,y0,x1,y1   任意の長方形(mm)。例: --area 0,0,582.5,840
    変換は台の座標全体に対して作られるので、映る範囲が狭いだけで位置は正しい。
    プロジェクターの画面に入らない部分のヒートマップは映らない。
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from table_config import get_table


def set_table(name):
    """台の大きさを決める。TABLE_W, TABLE_L, 4隅(1=原点角 から反時計回り)、--area の名前を作る"""
    global TABLE_W, TABLE_L, TABLE_CORNERS_MM, AREA_PRESETS
    TABLE_W, TABLE_L = get_table(name)[:2]
    TABLE_CORNERS_MM = np.array([[0, 0], [TABLE_W, 0], [TABLE_W, TABLE_L], [0, TABLE_L]], dtype=np.float32)
    AREA_PRESETS = {
        "full": (0.0, 0.0, TABLE_W, TABLE_L),
        "left": (0.0, 0.0, TABLE_W / 2, TABLE_L),
        "right": (TABLE_W / 2, 0.0, TABLE_W, TABLE_L),
    }


def area_corners(area):
    """--area の指定から、印を合わせる4点(mm)を返す。1=X,Y が小さい角から反時計回り"""
    if area in AREA_PRESETS:
        x0, y0, x1, y1 = AREA_PRESETS[area]
    else:
        x0, y0, x1, y1 = (float(v) for v in area.split(","))
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
WINDOW = "projector"


def detect_projector(args):
    """プロジェクター画面の位置と大きさを返す (x, y, w, h)"""
    if args.width and args.height:
        return args.screen_x, args.screen_y, args.width, args.height
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        print("見つかった画面:")
        for i, m in enumerate(mons):
            print(f"  [{i}] x={m.x} y={m.y} {m.width}x{m.height} primary={getattr(m, 'is_primary', '?')}")
        others = [m for m in mons if not getattr(m, "is_primary", False)]
        m = others[0] if others else mons[-1]
        print(f"→ プロジェクターとして使う画面: x={m.x} y={m.y} {m.width}x{m.height}")
        return m.x, m.y, m.width, m.height
    except ImportError:
        print("screeninfo が入っていないため、--screen-x などの指定が必要です。")
        print("  (pip install screeninfo で自動検出できるようになります)")
        raise SystemExit(1)


def open_fullscreen(x, y):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW, x, y)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw(w, h, pts, sel, guide, corners_mm):
    img = np.zeros((h, w, 3), np.uint8)
    if guide:
        # ガイド(緑の線)は、4つの印の位置から計算した「台の枠とセンターライン」。
        # 印を1つ動かすと、ガイドの線全体の形が変わる(印そのものは選んだ1つしか動かない)。
        H = cv2.getPerspectiveTransform(corners_mm, pts.astype(np.float32))
        outline = cv2.perspectiveTransform(TABLE_CORNERS_MM.reshape(-1, 1, 2), H).astype(np.int32)
        cv2.polylines(img, [outline], True, (0, 160, 0), 1)
        center = np.array([[[TABLE_W / 2, 0]], [[TABLE_W / 2, TABLE_L]]], np.float32)
        c = cv2.perspectiveTransform(center, H).astype(np.int32)
        cv2.line(img, tuple(c[0, 0]), tuple(c[1, 0]), (0, 160, 0), 1)
    # 4つの印を結ぶ四角(合わせる範囲)
    cv2.polylines(img, [pts.astype(np.int32)], True, (120, 120, 120), 1)
    for i, (px, py) in enumerate(pts.astype(int)):
        if i == sel:
            # 選んでいる印: 明るい黄色で大きく(赤はプロジェクターだと暗く見えにくいので使わない)
            color, size, thick = (0, 255, 255), 90, 3
            cv2.circle(img, (px, py), 28, color, 2)
        else:
            color, size, thick = (255, 255, 255), 60, 2
        cv2.drawMarker(img, (px, py), color, cv2.MARKER_CROSS, size, thick)
        cv2.putText(img, str(i + 1), (px + 16, py - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    mx, my = corners_mm[sel]
    cv2.putText(img, f"selected: {sel + 1}  -> table X={mx:g} Y={my:g} mm", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(img, "1-4: select   arrows: coarse   wasd: fine   g: guide   Enter: save   Esc: quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="保存先 (例: results\\20261016\\projector_transform.npz)")
    parser.add_argument("--screen-x", type=int, default=0)
    parser.add_argument("--screen-y", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--coarse-step", type=int, default=20, help="矢印キーで動かす画素数")
    parser.add_argument("--area", default="full", help="印を合わせる範囲: full / left / right / x0,y0,x1,y1(mm)")
    parser.add_argument("--table", default=None, help="台の名前(src/table_config.py)。省略すると今の設定")
    args = parser.parse_args()
    set_table(args.table)
    corners_mm = area_corners(args.area)

    x, y, w, h = detect_projector(args)

    # 前回の結果があれば、その位置から始める(毎回の調整が少なくて済む)
    pts = None
    if os.path.exists(args.out):
        d = np.load(args.out)
        if tuple(d["size"]) == (w, h) and np.allclose(d["table_corners_mm"], corners_mm):
            pts = d["proj_corners"].astype(np.float64)
            print(f"前回の位置から始めます: {args.out}")
    if pts is None:
        m = 0.2
        pts = np.array([[w * m, h * (1 - m)], [w * (1 - m), h * (1 - m)],
                        [w * (1 - m), h * m], [w * m, h * m]], dtype=np.float64)

    open_fullscreen(x, y)
    sel = 0
    guide = True
    c = args.coarse_step
    # 細かく: w/a/s/d(1画素)
    fine = {ord("w"): (0, -1), ord("a"): (-1, 0), ord("s"): (0, 1), ord("d"): (1, 0)}
    # 大まかに: 矢印キー。waitKeyEx が返す値は OS によって違うので両方登録する
    #   Windows: ← 2424832, ↑ 2490368, → 2555904, ↓ 2621440
    #   Linux  : ← 65361,   ↑ 65362,   → 65363,   ↓ 65364
    coarse = {2424832: (-c, 0), 2490368: (0, -c), 2555904: (c, 0), 2621440: (0, c),
              65361: (-c, 0), 65362: (0, -c), 65363: (c, 0), 65364: (0, c)}
    print("印1〜4を、台の上の次の位置(mm)に合わせてください(原点角=給球機側エンドラインの左):")
    for i, (mx, my) in enumerate(corners_mm):
        print(f"  {i + 1}: X={mx:g}, Y={my:g}")

    while True:
        cv2.imshow(WINDOW, draw(w, h, pts, sel, guide, corners_mm))
        raw = cv2.waitKeyEx(30)   # 矢印キーを受け取るため waitKeyEx を使う
        if raw == -1:
            continue
        if raw in coarse:
            move = coarse[raw]
            key = None
        else:
            key = raw & 0xFF
            move = fine.get(key)
        if move is not None:
            dx, dy = move
            pts[sel, 0] = np.clip(pts[sel, 0] + dx, 0, w - 1)
            pts[sel, 1] = np.clip(pts[sel, 1] + dy, 0, h - 1)
            continue
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            sel = key - ord("1")
        elif key == ord("g"):
            guide = not guide
        elif key in (13, 10):
            H = cv2.getPerspectiveTransform(corners_mm, pts.astype(np.float32))
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            np.savez(args.out, H=H, proj_corners=pts, table_corners_mm=corners_mm,
                     size=np.array([w, h]), screen_pos=np.array([x, y]))
            print(f"保存しました: {args.out}")
            break
        elif key == 27:
            print("保存せずに終了しました")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()