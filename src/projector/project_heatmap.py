# -*- coding: utf-8 -*-
"""
着地点のヒートマップを、プロジェクターで卓球台の上に映す。

    着地点(CSV) → 卓球台座標(mm)で密度を計算 → 黒背景の画像にする
    → projector_transform.npz の変換で、プロジェクターの画面に合わせてゆがめる → 全画面表示

黒は光らないので、台の上には「着地が多かった場所」だけが光って見える。
色は1つの色相(オレンジ)の明るさで多い/少ないを表す(多いほど明るい)。

使い方:
    python project_heatmap.py <着地点CSV> --transform results\\YYYYMMDD\\projector_transform.npz

    <着地点CSV> は x_mm, y_mm の列があるもの(run_all.py の all_landings.csv と同じ形)

オプション:
    --bandwidth 60     1点をどのくらいの広さにぼかすか(mm)。点が少ないときは大きめが見やすい
    --points           着地点そのものを白い小さな丸で重ねて映す
    --outline          台の計測範囲の枠を薄く映す(位置合わせの確認用)
    --save out.png     映す画像をファイルにも保存する(家での確認用)
    --no-show          画面に映さない(--save と組み合わせて、家で画像だけ作るとき)
    --test-pattern     ヒートマップの代わりに、位置確認用の目盛り(10cmごとの格子と数字)を映す。
                       着地点CSVは要らない。物差しで「X=50cmの線が、原点角から50cmの所に
                       映っているか」を確かめる。
                       例: python project_heatmap.py --test-pattern --transform results\\YYYYMMDD\\projector_transform.npz

キー操作: Esc または q で終了
"""

import argparse
import csv
import os

import cv2
import numpy as np

TABLE_W = 1165   # mm(X: 幅方向)
TABLE_L = 840    # mm(Y: 長さ方向)
HUE_BGR = np.array([0, 140, 255], np.float32)      # オレンジ(1つの色相)
TOP_BGR = np.array([150, 225, 255], np.float32)    # いちばん多い所は少し明るく(同じ系統の色)
CUTOFF = 0.05     # 最大の5%未満は真っ黒にする(うっすら全体が光るのを防ぐ)
WINDOW = "heatmap"


def load_points(path):
    xs, ys = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x_mm"]))
            ys.append(float(row["y_mm"]))
    return np.array(xs), np.array(ys)


def density_map(xs, ys, bandwidth):
    """卓球台座標 1画素=1mm の密度画像(0〜1)を作る(ガウスカーネルの重ね合わせ)"""
    gx = np.arange(TABLE_W, dtype=np.float32)
    gy = np.arange(TABLE_L, dtype=np.float32)
    d = np.zeros((TABLE_L, TABLE_W), np.float32)
    for x, y in zip(xs, ys):
        kx = np.exp(-0.5 * ((gx - x) / bandwidth) ** 2)
        ky = np.exp(-0.5 * ((gy - y) / bandwidth) ** 2)
        d += np.outer(ky, kx)
    if d.max() > 0:
        d /= d.max()
    return d


def render_table_image(xs, ys, bandwidth, points, outline):
    """卓球台座標(1画素=1mm)の、黒背景のヒートマップ画像(BGR)"""
    img = np.zeros((TABLE_L, TABLE_W, 3), np.uint8)
    if len(xs) > 0:
        d = density_map(xs, ys, bandwidth)
        d[d < CUTOFF] = 0
        t = d[..., None]
        # 0→黒、中くらい→オレンジ、最大→明るいオレンジ(明るさで多さを表す)
        low = HUE_BGR * np.clip(t / 0.7, 0, 1)
        high = HUE_BGR + (TOP_BGR - HUE_BGR) * np.clip((t - 0.7) / 0.3, 0, 1)
        col = np.where(t <= 0.7, low, high)
        img = np.clip(col, 0, 255).astype(np.uint8)
    if outline:
        cv2.rectangle(img, (0, 0), (TABLE_W - 1, TABLE_L - 1), (90, 90, 90), 3)
        cv2.line(img, (TABLE_W // 2, 0), (TABLE_W // 2, TABLE_L - 1), (90, 90, 90), 2)
    if points:
        for x, y in zip(xs, ys):
            cv2.circle(img, (int(round(x)), int(round(y))), 8, (255, 255, 255), 2)
    return img


def render_test_pattern():
    """位置確認用: 10cmごとの格子(50cmごとに太く)、計測範囲の枠、センターライン、原点角の赤丸。
    数字は、ゆがめると逆さ・鏡文字になるので、ここでは描かずに位置(mm)だけ返す。"""
    img = np.zeros((TABLE_L, TABLE_W, 3), np.uint8)
    labels = []
    for x in range(0, TABLE_W + 1, 100):
        major = x % 500 == 0
        xx = min(x, TABLE_W - 1)
        cv2.line(img, (xx, 0), (xx, TABLE_L - 1), (255, 255, 255) if major else (150, 150, 150), 4 if major else 2)
        if x > 0:
            labels.append((f"{x // 10}", x, 0))
    for y in range(0, TABLE_L + 1, 100):
        major = y % 500 == 0
        yy = min(y, TABLE_L - 1)
        cv2.line(img, (0, yy), (TABLE_W - 1, yy), (255, 255, 255) if major else (150, 150, 150), 4 if major else 2)
        if y > 0:
            labels.append((f"{y // 10}", 0, y))
    cv2.rectangle(img, (0, 0), (TABLE_W - 1, TABLE_L - 1), (0, 255, 0), 8)
    cv2.line(img, (TABLE_W // 2, 0), (TABLE_W // 2, TABLE_L - 1), (0, 255, 0), 4)
    cv2.circle(img, (0, 0), 40, (0, 0, 255), -1)
    return img, labels


def draw_labels(proj_img, labels, H):
    """目盛りの数字を、ゆがめた後の画面にまっすぐ描く(台の座標 → 画面の位置だけ変換する)"""
    h, w = proj_img.shape[:2]
    for text, x, y in labels:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
        px, py = int(p[0]), int(p[1])
        if 0 <= px < w and 0 <= py < h:
            cv2.putText(proj_img, text, (px + 6, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None, help="着地点CSV(x_mm, y_mm の列)")
    parser.add_argument("--transform", required=True, help="projector_align.py で作った projector_transform.npz")
    parser.add_argument("--bandwidth", type=float, default=60.0)
    parser.add_argument("--points", action="store_true")
    parser.add_argument("--outline", action="store_true")
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--test-pattern", action="store_true")
    args = parser.parse_args()

    if not args.test_pattern:
        if args.csv is None:
            parser.error("着地点CSVを指定してください(位置確認だけなら --test-pattern)")
        xs, ys = load_points(args.csv)
        print(f"着地点: {len(xs)}点")
        inside = (xs >= 0) & (xs <= TABLE_W) & (ys >= 0) & (ys <= TABLE_L)
        if (~inside).any():
            print(f"  (計測範囲の外の点 {int((~inside).sum())}個は映りません)")

    tf = np.load(args.transform)
    H = tf["H"]
    w, h = (int(v) for v in tf["size"])
    sx, sy = (int(v) for v in tf["screen_pos"])

    labels = None
    if args.test_pattern:
        table_img, labels = render_test_pattern()
        print("位置確認用の目盛りを映します(数字は原点角からのcm。赤丸が原点角)")
    else:
        table_img = render_table_image(xs, ys, args.bandwidth, args.points, args.outline)
    proj_img = cv2.warpPerspective(table_img, H, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if labels:
        draw_labels(proj_img, labels, H)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        cv2.imwrite(args.save, proj_img)
        base, ext = os.path.splitext(args.save)
        cv2.imwrite(base + "_table" + ext, table_img)
        print(f"保存しました: {args.save}(台の座標のままの画像: {base}_table{ext})")

    if not args.no_show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW, sx, sy)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(WINDOW, proj_img)
        print("投影中(Esc または q で終了)")
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (27, ord("q")):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()