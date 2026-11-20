# dotkoma

Pixel Art Atlas の 16×16 透過スプライトを出発点に、動画モデル（Wan2.2）で短いループ候補を合成し、
16×16 に戻しても成立するものだけを残すデータセット構築パイプライン。

```
prepare  (手元CPU)   Atlas抽出 → 参照PNG + jobs.jsonl
generate (Colab GPU) 参照PNG → raw/<job_id>.mp4 + results.jsonl
finalize (手元CPU)   一周期抽出 → 透過・16×16化 → 品質チェック → frames/*.png + sample.json + preview.gif
```

工程の境界はファイルだけ。すべて `work/<run>/` 以下に出力され、Git には入らない。

## セットアップ

```sh
uv sync          # 手元環境。PyTorch / CUDA は入らない
uv run pytest    # GPU なしで prepare → generate(fake) → finalize を検証
```

## 段階0: CPU だけで一周させる

`configs/smoke.toml` は手描きの fixture スプライトと `fake` バックエンド（既知の上下・左右移動を合成）を使う。

```sh
uv run dotkoma prepare  configs/smoke.toml
uv run dotkoma generate configs/smoke.toml
uv run dotkoma finalize configs/smoke.toml
open work/smoke/final/*/preview.gif
```

## 段階1: 12本の試作（Colab）

```sh
uv run dotkoma prepare configs/pilot.toml        # Atlas の1シャード(29MB)を取得して12ジョブを作る

uv tool install --python 3.12 google-colab-cli   # Colab CLI は dotkoma とは別環境
colab new -s dotkoma-pilot --gpu L4
echo '!nvidia-smi' | colab exec -s dotkoma-pilot
# src/ scripts/ work/pilot/{jobs.jsonl,inputs/} を colab upload で送り、VM 上で:
#   python scripts/colab_pilot.py work/pilot --sync-dir /content/drive/MyDrive/dotkoma/pilot
# work/pilot/{raw/,results.jsonl,colab_env.json} を colab download で回収
colab stop -s dotkoma-pilot

uv run dotkoma finalize configs/pilot.toml
```

> `colab upload/download` の正確な引数と、TI2V-5B を `WanImageToVideoPipeline` で呼べるかは初回の Colab 実行で確認する（未検証）。

## ファイル契約（`work/<run>/`）

| ファイル | 書き手 | 内容 |
|---|---|---|
| `jobs.jsonl` | prepare | `job_id`、元データ（repo・revision・row_id・subject・caption）、`split`、`requested_action`、プロンプト、シード、モデル・推論設定、参照レイアウト、キー色 |
| `inputs/<row_id>.png` / `_ref_<key>.png` | prepare | 元の 16×16 RGBA と、単色背景に整数倍拡大した参照画像 |
| `results.jsonl` | generate | 追記専用。ジョブごとの `ok`/`failed`、実モデルIDとリビジョン、ランタイム、所要時間、失敗理由 |
| `raw/<job_id>.mp4` | generate | 一時ファイルに書いてから rename。存在 + 最後の記録が `ok` で完了扱い |
| `selections.jsonl` | 人 | 目視後のコマ選択の上書き `{"job_id", "start", "end"}` または `{"job_id", "indices"}` |
| `final/<job_id>/` | finalize | `frames/NNN.png`（正本）、`sample.json`（表示時間・出典・検査値）、`preview.gif` |
| `finalize.jsonl` | finalize | 全ジョブの機械判定（`candidate` / `flagged` / `reject` とフラグ） |
| `review.jsonl` | 人 | 最終目視の採否（機械判定とは別に残す） |

- `job_id` は生成に効く入力すべてのハッシュ。同じ入力なら同じID、条件を変えれば別ID。
- `generate` を再実行すると未完了のジョブだけ処理する。失敗ジョブは `--retry-failed` で再試行。
- 動作ごとに言い換えプロンプトを `prompts = [...]` で複数持てる。スプライト側の `prompt_variants`（`"all"` または `[0, 2]` など、既定は `[0]`）で実行する版を選ぶ。同じ動作の版はシードを共有するので、`prompt_variant` 別の採用率で言い回しの効果だけを比べられる。
- `finalize` は元動画を読み直すだけなので、後処理設定やコマ選択を変えて何度でもやり直せる。

## 品質チェック（`sample.json` の `checks`）

| フラグ | 種類 | 意味 |
|---|---|---|
| `wrong_size` `frame_count` `empty_frame` `all_identical` | 除外 | 形式不正・全透過コマ・動きなし |
| `loop_jump` | 要確認 | 最後→最初の差分が隣接コマの典型差分より大きい |
| `edge_cut` | 要確認 | 16×16 枠の外側に前景がはみ出している |
| `background_unstable` | 要確認 | 画面の縁で背景色が保たれていない |
| `source_mismatch` | 要確認 | 1コマ目が元絵から大きく変わっている |

## リポジトリ構成

```
configs/        smoke.toml（段階0）, pilot.toml（段階1）
src/dotkoma/    prepare.py generate.py finalize.py + config/records/imaging/video/source
scripts/        colab_pilot.py（Colab VM 上で generate を回す）
tests/          fixtures/（手描きスプライト）と CPU だけで動くテスト
docs/           dataset-card.md
```

## ライセンス

コードは MIT（`LICENSE`）。生成データの利用条件は別に扱い、`docs/dataset-card.md` に記載する。
