# -*- coding: utf-8 -*-
"""
台の計測範囲(mm)の設定。ここ1か所を変えれば、台の位置合わせ・追跡の図・着地点・投影がそろって変わる。

新しい台に切り替えるとき:
    下の CURRENT を "half" に変える(1行だけ)。

切り替えた後で、前の台の録画を処理し直すとき:
    コマンドに --table lab_print を付ける。

座標の決まりは台が変わっても同じ:
    原点はマシン側エンドラインの左の角。X は幅方向、Y は長さ方向(ネットに近づくほど大きい)。

far_label は compute_table_pose.py の4隅の説明に使う、原点から遠い側の辺の名前。
"""

TABLES = {
    # 研究室のプリント(10/31 までの録画はすべてこれ)
    "lab_print": {"w": 1165.0, "l": 840.0, "far_label": "プリント切れ目"},
    # 新しい台のマシン側の半面(幅 152.5cm、エンドライン〜ネット 137cm)
    "half": {"w": 1525.0, "l": 1370.0, "far_label": "ネットの真下"},
}

CURRENT = "lab_print"   # ← 新しい台にするときは "half" に変える


def get_table(name=None):
    """(幅 mm, 長さ mm, 遠い側の辺の名前) を返す。name が None なら CURRENT の台"""
    name = name or CURRENT
    if name not in TABLES:
        raise SystemExit(f"[エラー] 台の名前 {name} はありません。使える名前: {', '.join(TABLES)}")
    t = TABLES[name]
    return t["w"], t["l"], t["far_label"]
