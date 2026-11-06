# table-tennis-feedback

卒業研究「2台のカメラを用いた多球練習の着地点分布可視化システム」のプログラムとデータ。
個人用。

2台の高速カメラ(Teli BU160MCG、1440×1080、約227fps)でボールを検出し、3D位置 → 着地点 → ヒートマップを求め、1セット終了後にプロジェクターで卓球台の上に映す。

## フォルダ構成

```
table-tennis-feedback/
  README.md           このファイル
  NOTES.md            開発メモ(7月ごろ)
  src/
    run_set.py        撮影の後を1回の操作でまとめる(追跡 → 着地点 → ヒートマップ)
    table_config.py   台の計測範囲の設定(新しい台にするときはここの CURRENT を変える)
    capture/          撮影
      capture_live_detect.py   撮りながら検出し、ボールの位置だけ記録する(本番用)
      capture_dual_diag.py     全コマを画像で保存する(確認・調査用。1回500枚まで)
      capture_dual.py          全コマを画像で保存する(元の版)
    projector/        プロジェクター
      projector_align.py       投影の4隅合わせ
      project_heatmap.py       ヒートマップを台に合わせて投影する
    calibration/      キャリブレーション
      calibrate_single_charuco.py   A. カメラ1台ずつのクセ(K, dist)
      stereo_calibrate_charuco.py   B. 2台の位置関係(stereo_result.npz)
      check_camera_moved.py         撮影中にカメラが動いていないかの確認
      ほか確認用
    ball_detection/   検出(オフライン版)
    triangulation/    3D位置・着地点・診断
      compute_table_pose.py      C. 台の位置(4隅クリック → table_transform.npz)
      triangulate_trajectory.py  検出結果 → 3D位置・着地点
      diagnose_triangulation.py  再投影誤差・コマのずれの診断
      show_pair_crops.py / show_epipolar_crops.py  2台の切り出し画像の確認
      plot_trajectories.py       軌道の図
    tools/            確認・集計用(検出の連続性、点滅テスト、速度測定、bmp→png変換など)
    old/              今は使っていないもの
  data/               撮影データ(git には入れない)
    captures/
      20260918/       9月の撮影(hitball1〜10、ChArUco、位置確認用の画像)
      20261016/       10/16の撮影(charuco、table_start/end、live_*、known_positions)
  results/            計算結果(npz・csv・json は git に入れる。画像は入れない)
    20260918/
    20261016/
```

## 置き場所のルール

1. **撮影日ごとにフォルダを作る。** その日の撮影データは `data/captures/YYYYMMDD/`、その日のデータから計算した結果は `results/YYYYMMDD/` に入れる。
2. **日付フォルダには、その日に作ったものだけを入れる。** 別の日のファイルをコピーしない。
   (Windowsはコピーしても更新日が変わらないので、混ざると見分けがつかない。10/17〜10/21に9月のキャリブレーションを10/16の解析に使ってしまった原因)
3. **キャリブレーション結果は、必ずその撮影日の `results/YYYYMMDD/` に出力する。** プロジェクトのいちばん上に置かない。
4. **カメラを置き直したら、キャリブレーション(B. 2台の位置関係、C. 台の位置)はやり直す。** 別の日の stereo_result.npz を使わない。
5. **消す前に git に記録しておく。** 撮影データ(data/)は git に入らないので、消す・移すときは別の場所に控えがあるか確認する。

## 本番の流れ(撮影 → 着地点 → 投影を1回の操作で)

```
python src/capture/capture_live_detect.py --min-circ 0.1 --calib results/YYYYMMDD --projector results/YYYYMMDD/projector_transform.npz
```

- 1セットは `r` で開始してから止めるまでの1回。止めると自動で追跡 → 着地点 → 投影まで進む。
- 投影の画面で Esc または q を押すと閉じて、カメラの画面に戻る。処理と投影の間は `r` も `q` も効かない。
- `--projector` を省くと投影せず、画像の保存だけをする。`--calib` がなければ今までと同じ動き。
- 録画は安全のため 180 秒(`MAX_RECORD_SEC`)で自動的に止まる。

## 撮影の後の処理(1回の操作)

自動の処理が止まったときや、検出結果から処理し直すときは、次のコマンドで動かす。

```
python src/run_set.py <録画フォルダ> [<録画フォルダ> ...] --calib results/YYYYMMDD --projector results/YYYYMMDD/projector_transform.npz
```

- 録画を複数渡すと、着地点を1つにまとめて1枚のヒートマップにする。
- `--projector` を省くと投影せず、台の座標のままの画像だけを保存する。`--no-show` は画面に映さない。
- 結果は毎回 `results/YYYYMMDD/set_日付_時刻/` に新しく作るので、前の結果を上書きしない。

## キャリブレーション結果の見分け方

stereo_result.npz の `T` の長さが、2台のカメラの距離(メートル)。

| ファイル | 2台の距離 |
|---|---|
| results/20260918/stereo_result_from_0918.npz | 約751mm |
| results/20261016/stereo_result.npz | 約452mm |
