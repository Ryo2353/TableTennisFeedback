"""
画像上の点をクリックしてピクセル座標(u, v)を記録するツール。

既知位置(ボールを置いた7か所など)の画像から、triangulate.py に渡す
(u, v)座標を手作業で拾うために使う。ノートPC側で実行すること
(このサンドボックス環境にはディスプレイがないため実行できない)。

使い方:
    python pick_points.py <画像パス> [出力先フォルダ] [--max-dim N]

出力先フォルダを省略した場合は、従来通り画像と同じフォルダに保存する。
撮影データ(画像)と結果(json/確認画像)を分けたい場合は、2番目の引数で
出力先フォルダを指定する(例: python pick_points.py data/captures/.../known_01.png results/20260918/precision_check)。

画面に収まりきらない場合は --max-dim で表示サイズ(長辺のピクセル数)を
小さく指定する(デフォルト1000)。例: python pick_points.py known_01.png --max-dim 700

操作方法:
    左クリック   : その位置に点を追加(クリック順に0,1,2...と番号が振られる)
    'z' キー     : 直前の点を1つ取り消す
    's' キー     : 保存して終了
    'q' / ESC    : 保存せず終了

保存されるファイル(画像と同じフォルダに):
    <画像名>_points.json : [[u1,v1], [u2,v2], ...] のリスト(クリック順)
    <画像名>_picked.png  : 点と番号を描き込んだ確認用画像

重要: カメラA画像とカメラB画像で、同じ物理的な点を「同じ順番」でクリックすること。
例えば既知位置1→2→3...の順でボールを置いたなら、カメラA・カメラB両方の画像で
その順番通りにクリックする。順番がずれると、triangulate.py が別々の点を
対応づけてしまい、結果が完全に狂う。

ウィンドウが画面より大きい場合は自動的に縮小表示するが、保存される座標は
常に元画像の解像度(フル解像度)基準になる。
"""

import sys
import os
import json
import cv2

DEFAULT_MAX_DISPLAY_DIM = 1000


def main():
    args = sys.argv[1:]

    max_display_dim = DEFAULT_MAX_DISPLAY_DIM
    if "--max-dim" in args:
        i = args.index("--max-dim")
        max_display_dim = int(args[i + 1])
        args = args[:i] + args[i + 2:]

    if len(args) not in (1, 2):
        print(__doc__)
        sys.exit(1)

    img_path = args[0]
    out_folder = args[1] if len(args) == 2 else None
    img = cv2.imread(img_path)
    if img is None:
        print(f"[エラー] 画像を読み込めません: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_display_dim:
        scale = max_display_dim / max(h, w)
    disp_w, disp_h = int(w * scale), int(h * scale)

    points = []  # フル解像度基準の(u, v)を保持

    window_name = f"pick_points - {os.path.basename(img_path)} (左クリック:追加 z:取消 s:保存 q:終了)"

    def redraw():
        disp = cv2.resize(img, (disp_w, disp_h)) if scale != 1.0 else img.copy()
        for i, (u, v) in enumerate(points):
            du, dv = int(u * scale), int(v * scale)
            cv2.circle(disp, (du, dv), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i), (du + 8, dv - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 表示座標 -> 元画像のフル解像度座標に変換して保存
            orig_u, orig_v = x / scale, y / scale
            points.append((orig_u, orig_v))
            print(f"  点{len(points) - 1}: (u={orig_u:.1f}, v={orig_v:.1f})")
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    print(f"\n画像: {img_path}  (元解像度 {w}x{h}, 表示倍率 {scale:.3f})")
    print("左クリックで点を追加、'z'で取消、's'で保存、'q'/ESCで終了せず抜けます。\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('z'):
            if points:
                removed = points.pop()
                print(f"  取消: (u={removed[0]:.1f}, v={removed[1]:.1f})")
                redraw()
        elif key == ord('s'):
            break
        elif key == ord('q') or key == 27:
            print("保存せずに終了します。")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    base = os.path.splitext(os.path.basename(img_path))[0]
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
        stem = os.path.join(out_folder, base)
    else:
        stem = os.path.splitext(img_path)[0]
    json_path = f"{stem}_points.json"
    png_path = f"{stem}_picked.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([[u, v] for u, v in points], f, ensure_ascii=False, indent=2)

    # 確認用画像(フル解像度に点を描き込んで保存)
    confirm_img = img.copy()
    for i, (u, v) in enumerate(points):
        cv2.circle(confirm_img, (int(u), int(v)), 8, (0, 0, 255), -1)
        cv2.putText(confirm_img, str(i), (int(u) + 12, int(v) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(png_path, confirm_img)

    print(f"\n{len(points)}点を保存しました:")
    print(f"  座標: {json_path}")
    print(f"  確認用画像: {png_path}")


if __name__ == "__main__":
    main()