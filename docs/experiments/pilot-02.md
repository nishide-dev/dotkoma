# Pilot 02: Wan2.2 I2V-A14B + 4-step Lightning LoRA, 12 jobs (2026-11-20)

Pilot 01（`pilot-01.md`, TI2V-5B）と同じ 12 組（題材・動作・プロンプト・シード導出）を、教師モデルと推論設定だけ替えて生成した。生成物（`work/pilot-a14b/`）は Git 管理外。

## 条件

| 項目 | Pilot 01 (5B) | Pilot 02 (A14B) |
|---|---|---|
| モデル | `Wan2.2-TI2V-5B-Diffusers` @ `b8fff73…` | `Wan2.2-I2V-A14B-Diffusers` @ `596658f…` |
| LoRA | なし | `lightx2v/Wan2.2-Distill-Loras` @ `57adea5…` の I2V 4-step rank64（high/low noise を各エキスパートへ融合） |
| ステップ / guidance | 40 / 5.0 | 4 / 1.0（CFG なし） |
| 重み | bf16 | 融合後に fp8 で保持（計算は bf16） |
| 解像度 / フレーム | 1280×704, 49 frames @ 24 fps | 832×480, 49 frames @ 16 fps |
| 参照画像 | 28 倍（448 px） | 24 倍（384 px） |
| GPU | Colab L4 (22.5 GB), CPU offload | Colab A100 40 GB, CPU offload（`scripts/colab_pilot.py` の既定。`--no-offload` なし） |
| 環境 | torch 2.14.1+cu130, diffusers 0.42.1 | 同左 + peft 0.21.2（torchao 0.10.0 は削除） |

## 実測

| 項目 | Pilot 01 (5B) | Pilot 02 (A14B) |
|---|---|---|
| 生成 1 本 | 445 s | 61 s（初回のみ 106 s） |
| モデルロード | 62 s（キャッシュ後） | 417–428 s（うち LoRA 融合 68–71 s） |
| 消費ユニット（12 本、失敗・待機込み） | 約 5.6 | 約 5.0 |

- A14B は bf16 で読み込む時点で CPU RAM を約 84 GB 使う。前回の実行が残ったカーネルで再ロードすると、83 GB の VM で OOM になりカーネルが落ちた。実行ごとにカーネルを再起動して回避した。
- 1 本あたりの生成は 5B の約 1/7。ロードは 1 回の実行につき 1 度なので、本数が多いほど差が開く。

## 結果

機械判定は candidate 2 / flagged 10。目視で使えそうなのは 3 本（Pilot 01 は 0 本）。

| 題材 | 動作 | 起きたこと | 機械判定のフラグ |
|---|---|---|---|
| slime | bob | 上へ動いたまま戻らず、下にしずくが垂れる | loop_jump, edge_cut |
| slime | hop | 上へ動いた後、下側が溶けたような形になる | edge_cut, identity_drift |
| candle | flicker | **炎だけがゆらめく。16×16 でも見え、ループもなめらか** | なし（candidate） |
| candle | hop | 途中で別の物体に崩れて戻る | identity_drift |
| toy robot | hop | 跳ばずに腕と脚を動かし、形が崩れる | edge_cut, identity_drift |
| toy robot | wave | **本体を保ったまま腕を繰り返し振る**（腕が少し枠外へ） | edge_cut |
| ghost | blink | 目の形が変わる（閉じる動きではない） | loop_jump, identity_drift |
| ghost | bob | 形が変わり、影が出る | edge_cut, identity_drift |
| cat | blink | 目が開く。ただし白目が 16×16 で消える | なし（candidate） |
| cat | hop | 歩く猫に変身する | edge_cut, identity_drift |
| cup of coffee | bob | 浮かずに影が出て形が変わる | edge_cut, identity_drift |
| cup of coffee | steam | **カップの上に湯気が立ちのぼる**（枠の上端で切れる） | edge_cut |

## わかったこと

12 本・単一設定（シード各 1）での観察であり、一般的な結論ではない。

1. **局所的な動き（炎の揺れ・手振り・湯気）は A14B で出る。** 5B では 0 本だった。
2. **全体の移動（hop / bob）は A14B でも 6 本すべて成立しなかった。** 戻らない・変身・影の追加になる。ピクセル操作ルートで作る方針を支持する。
3. **元絵の見た目の保持は大きく改善した。** 5B で全コマに出ていた質感のちらつきは出ていない。
4. **元絵パレットへのスナップが、新しく現れた色を消す。** cat blink では開いた目の白が元絵のパレットにないため、16×16 で消えた。湯気の色も最も近い既存色に置き換わる。
5. **edge_cut は「少しはみ出す」局所動作にも付く。** 手振り・湯気は見た目では使えるが、要確認として残る。

## 次の候補

- 全体移動（hop / bob / sway）はピクセル操作で作る（`generation_method = "pixel_ops"`）
- 局所動作は A14B + Lightning で量産する。1 本約 61 秒
- パレット処理を見直す。元絵パレットに、複数コマで安定して現れる新しい色を加えるなど
- edge_cut の扱いを見直す。はみ出し量の閾値、または局所動作では要確認のまま目視に回す
