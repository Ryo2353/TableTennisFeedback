# table-tennis-feedback

卒研用リポジトリ。三フレーム差分でボール検出するコード。

## ファイル
- `Detectball_frame.py` … 三フレーム差分法(メイン)
- `Detectball.py` … MOG2背景差分(比較用)

## 実行方法
1. `frames_◯◯/` に連番jpg画像を置く
2. コード内の `IMAGE_FOLDER = r"frames_◯◯"` を書き換える
3. `python Detectball_frame.py`
4. 結果は `output_detect_frame/`(検出結果画像)、
   `output_detect_frame/debug/`(二値化後の差分画像)に出力される

## 注意点
- 画像は.jpg限定(コード内でglobパターンが`*.jpg`固定)
- ファイル名は `◯◯_連番.jpg` の形式である必要がある(正規表現でソートしているため)