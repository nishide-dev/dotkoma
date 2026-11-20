# パイプライン内部仕様

開発者向けのファイル契約と品質チェックの詳細。利用者向けの説明は `README.md`。

## ファイル契約（`work/<run>/`）

| ファイル | 書き手 | 内容 |
|---|---|---|
| `jobs.jsonl` | prepare | `job_id`、元データ（repo・revision・row_id・subject・caption）、`split`、`requested_action`、`generation_method`、プロンプト、シード、モデル・推論設定、参照レイアウト、キー色（`pixel_ops` ジョブはプロンプト・モデル・参照の代わりに `pixel_ops`） |
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
- `generation_method = "pixel_ops"` の動作（`pixel_ops = { kind = "hop" | "bob" | "sway" | "custom", ... }`）は動画モデルを使わない。prepare は `pixel_ops`（展開済みのコマごとのずらし量 `offsets` と表示時間 `durations_ms`）を持つジョブを書き、generate はこれを飛ばし、finalize が元絵を整数ピクセルずらしてコマを作る。`results.jsonl` と `raw/` は作られない。`job_id` には展開後のずらし量と表示時間が入る（プリセット名ではない）。
- `pixel_ops` の動作は、動作ごとに別々に元絵を選ぶ。候補はその動きで 16×16 からはみ出さない元絵だけで、1 体もなければ警告を出してその（題材, 動作）のジョブを作らない。`row_id` 指定の元絵が収まらない場合も同様に飛ばす。動画の動作は従来どおりの順で選ぶ（既存の `job_id` は変わらない）。Atlas の元絵は枠いっぱいに描かれたものが多い。
- `pixel_ops` ジョブの `job_id` には元絵の画素のハッシュ（`sprite_sha256`）も入る。ローカルの元絵を書き換えれば別ジョブになる。
- `[generation]` は `pixel_ops` の動作だけの設定では省略できる（backend `none`）。
- `finalize` は元動画を読み直すだけなので、後処理設定やコマ選択を変えて何度でもやり直せる。

## 品質チェック（`sample.json` の `checks`）

| フラグ | 種類 | 意味 |
|---|---|---|
| `wrong_size` `frame_count` `empty_frame` `all_identical` | 除外 | 形式不正・全透過コマ・動きなし |
| `loop_jump` | 要確認 | 最後→最初の差分が隣接コマの典型差分より大きい |
| `edge_cut` | 要確認 | 16×16 枠の外側に前景がはみ出している |
| `background_unstable` | 要確認 | 画面の縁で背景色が保たれていない |
| `identity_drift` | 要確認 | いずれかのコマが、元絵を ±4px 平行移動したどれとも 30% 超のピクセルで食い違う（描き直し・変身） |

