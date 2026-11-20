# Pilot 01: Wan2.2 TI2V-5B, 12 jobs (2026-11-20)

段階1の試作結果。生成物そのもの（`work/pilot/`）は Git 管理外。

## 条件

| 項目 | 値 |
|---|---|
| 元データ | Pixel Art Atlas `bf78f8c8…`, `data/train-000.parquet` |
| 題材 × 動作 | 6 × 2 = 12（slime, candle, toy robot, ghost, cat, cup of coffee） |
| 教師モデル | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff73…`, `WanImageToVideoPipeline` |
| 推論 | 1280×704, 49 frames, 24 fps, 40 steps, guidance 5.0, CPU offload, VAE tiling |
| 参照画像 | 16×16 を 28 倍（448px）、単色キー背景の中央 |
| 後処理 | 等間隔 6 コマ、背景色はコマごとに測定、元絵パレットへスナップ |
| 環境 | Colab L4 (22.5 GB), Python 3.13, torch 2.14.1+cu130, diffusers 0.42.1 |

## 実測

- 生成 1 本: 445 秒（40 steps × 約 8.6 s + VAE デコード）
- モデルロード: 初回 137 秒（ダウンロード込み）、キャッシュ後 62 秒
- 消費: 約 5.6 ユニット（375.31 → 369.75）。失敗した試行・待機時間を含む
- VAE タイリングなしでは L4 で `vae.decode` が OOM

## 結果

採用 0 / 12。目視と機械判定は概ね一致（下表）。

| 題材 | 動作 | 起きたこと | 機械判定のフラグ |
|---|---|---|---|
| slime | bob | 動かず、体の質感がちらつく | loop_jump, identity_drift |
| slime | hop | 頭から元絵にないピンクの物体が生える | edge_cut, identity_drift |
| candle | flicker | 炎の揺れが 16×16 では消える | all_identical（除外） |
| candle | hop | 下へ沈んで画面外へ | edge_cut, background_unstable, identity_drift |
| toy robot | hop | 別の小さい全身キャラに変身、背景色も変化 | loop_jump, edge_cut, background_unstable, identity_drift |
| toy robot | wave | **腕を上げる動作が出た**（1コマだけ、腕が枠の上にはみ出す） | edge_cut |
| ghost | blink | 形が崩れる | loop_jump, identity_drift |
| ghost | bob | 途中 2 コマで別の形に崩れて戻る | identity_drift |
| cat | blink | 16×16 ではほぼ変化なし | loop_jump |
| cat | hop | 写実的な猫に変わり、跳んで小さくなる | edge_cut, identity_drift |
| cup of coffee | bob | ほぼ動かない | loop_jump |
| cup of coffee | steam | **湯気が出た**（カップの上、枠の上端で切れる） | edge_cut |

## わかったこと

12 本・単一設定（上の条件、シード各 1）での観察であり、一般的な結論ではない。

1. **全体移動（hop / bob）は今回の 6 本すべてで成立しなかった。** 動かないか、別キャラへの変身・縮小・画面外への移動になった。
2. **局所的な変化（wave / steam）は 2 本で部分的に出た。** ただし枠の外へはみ出した。
3. **炎の揺れ・まばたき（3 本）は 16×16 に戻すとほぼ消えた。** ピクセル操作ルートを用意する根拠になる。
4. **1 コマ目だけ参照画像そのもので、2 コマ目以降は質感と背景色が変わる。** 背景色をコマごとに測るよう後処理を修正した。
5. **初期の機械判定は「途中で変身して戻る」を見逃した。** 平行移動を許した元絵との差分（identity_drift）を全コマで見る検査に置き換えた。試作では、元絵を保ったコマが 0.23 以下、描き直されたコマが 0.35 以上で、閾値 0.3 で分離できた。

## 次の候補

- 全体移動（hop / bob / 上下移動）はピクセル操作で作る（`generation_method = "pixel_ops"`）
- 局所的な描き直し（wave / steam）は動画モデルに任せ、コマ選択の窓（`selections.jsonl`）で動作区間だけを切り出す
- 教師モデルを I2V-A14B（A100 / H100）に替えて同じ 12 本で比較する
- プロンプトの言い換え・推論設定（guidance、フレーム数）の比較
