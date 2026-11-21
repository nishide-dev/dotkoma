# パイプライン内部仕様

開発者向けのファイル契約と品質チェックの詳細。利用者向けの説明は `README.md`。

## ファイル契約（`work/<run>/`）

| ファイル | 書き手 | 内容 |
|---|---|---|
| `jobs.jsonl` | prepare | `job_id`、元データ（repo・revision・row_id・subject・caption）、`split`、`requested_action`、`generation_method`、プロンプト、シード、モデル・推論設定、参照レイアウト、キー色（`pixel_ops` ジョブはプロンプト・モデル・参照の代わりに `pixel_ops`） |
| `inputs/<row_id>.png` / `_ref_<key>.png` | prepare | 元の 16×16 RGBA と、単色背景に整数倍拡大した参照画像 |
| `plan_requests.jsonl` | prepare | `method = "vlm"` の動作について、計画を頼む（元絵, 動作）の一覧。`request_id`、元データ、動作の説明、VLM への指示文、計画モデルの設定 |
| `plans.jsonl` | plan | 追記専用。`request_id` ごとの VLM の答え（`feasible`、`moving_part`、`reason`、`motion_text`、生の出力）。最後の `ok` の記録が有効 |
| `results.jsonl` | generate | 追記専用。ジョブごとの `ok`/`failed`、実モデルIDとリビジョン、ランタイム、所要時間、失敗理由 |
| `raw/<job_id>.mp4` | generate | 一時ファイルに書いてから rename。存在 + 最後の記録が `ok` で完了扱い |
| `selections.jsonl` | 人 | 目視後のコマ選択の上書き `{"job_id", "start", "end"}` または `{"job_id", "indices"}` |
| `final/<job_id>/` | finalize | `frames/NNN.png`（正本）、`sample.json`（表示時間・出典・検査値）、`preview.gif` |
| `finalize.jsonl` | finalize | 全ジョブの機械判定（`candidate` / `flagged` / `reject` とフラグ） |
| `judge.jsonl` | judge | 追記専用。サンプルごとの VLM の判定（`motion_matches`、`drawing_kept`、`verdict` = `accept` / `reject`、`reason`、生の出力）。`request_id` ごとに最後の `ok` が有効 |
| `review.jsonl` | 人 | 最終目視の採否（機械判定とは別に残す）。1 行 `{"job_id", "postprocess_id", "verdict": "accept" / "reject" / "unsure", "motion_matches", "drawing_kept", "reason", "reviewer", "method"}`。`postprocess_id` が今の後処理と違う行は古いコマへの判定として扱う |

- `job_id` は生成に効く入力すべてのハッシュ。同じ入力なら同じID、条件を変えれば別ID。
- `generate` を再実行すると未完了のジョブだけ処理する。失敗ジョブは `--retry-failed` で再試行。
- 動作ごとに言い換えプロンプトを `prompts = [...]` で複数持てる。スプライト側の `prompt_variants`（`"all"` または `[0, 2]` など、既定は `[0]`）で実行する版を選ぶ。同じ動作の版はシードを共有するので、`prompt_variant` 別の採用率で言い回しの効果だけを比べられる。
- `method = "vlm"` の動作は、VLM（`[plan]`、既定 Qwen3.5-9B）が元絵を見て、その動作ができるか・どこが動くか・動画モデル向けの一文（`motion_text`）を答える。手順は `prepare`（`plan_requests.jsonl` を書く）→ `plan`（Colab GPU、`scripts/colab_pilot.py --stage plan`）→ もう一度 `prepare`（`feasible` な計画から、動作の `prompt_frame` の `{motion_text}` を埋めてジョブを作る）。`feasible = false` の組はジョブを作らない。`request_id` は元絵（画素のハッシュ `sprite_sha256` を含む）・動作・VLM に渡す実際の指示文（題材名・説明文を含む）・計画モデルの設定のハッシュ。計画の文面が変わればプロンプトが変わり、`job_id` も変わる。VLM のジョブの `job_id` には元絵の画素のハッシュも入る（定型文のジョブの `job_id` は従来どおり）。`plans.jsonl` は request_id ごとに最後の `ok` が有効で、後の失敗記録で消えない。`plan` は `plan_requests.jsonl` がなければエラーにし、読み込んだ元絵が依頼の `sprite_sha256` と違えばその依頼を失敗として記録する。`--sync-dir` では開始時と答えるたびに `plans.jsonl` を一時ファイル経由で写す（generate の同期も同じ方式）。VLM のジョブは `sprite_sha256` を記録し、generate は推論の前に元絵と参照画像が、finalize は後処理の前に元絵がそれに一致するかを確かめ、違えば失敗として記録する。ジョブと `sample.json` には `plan`（`request_id`、`moving_part`）が残る
- `generation_method = "pixel_ops"` の動作（`pixel_ops = { kind = "hop" | "bob" | "sway" | "custom", ... }`）は動画モデルを使わない。prepare は `pixel_ops`（展開済みのコマごとのずらし量 `offsets` と表示時間 `durations_ms`）を持つジョブを書き、generate はこれを飛ばし、finalize が元絵を整数ピクセルずらしてコマを作る。`results.jsonl` と `raw/` は作られない。`job_id` には展開後のずらし量と表示時間が入る（プリセット名ではない）。
- `pixel_ops` の動作は、動作ごとに別々に元絵を選ぶ。候補はその動きで 16×16 からはみ出さない元絵だけで、1 体もなければ警告を出してその（題材, 動作）のジョブを作らない。`row_id` 指定の元絵が収まらない場合も同様に飛ばす。動画の動作は従来どおりの順で選ぶ（既存の `job_id` は変わらない）。Atlas の元絵は枠いっぱいに描かれたものが多い。
- `pixel_ops` ジョブの `job_id` には元絵の画素のハッシュ（`sprite_sha256`）も入り、ジョブにも記録する。ローカルの元絵を書き換えれば別ジョブになる。finalize は `input_png` の画素がこのハッシュと一致しなければそのジョブを `finalize_failed` にする（prepare の中断などで入力とジョブが食い違った場合）。
- `[generation]` は `pixel_ops` の動作だけの設定では省略できる（backend `none`）。
- 色はループ全体で 1 つのパレットに寄せる（`finalize.palette`）。既定の `source+stable` は、元絵の色に「元絵のどの色からも `new_color_distance`（RGB 距離 80）より離れ、`new_color_min_frame_share`（半分）以上のコマに `new_color_min_pixels`（2 px）以上ずつ現れる色」を加える（開いた目の白など）。一部のコマにしか出ない色（変形途中・一瞬のノイズ）は元絵の色に寄せる。加えた色が引き寄せるのは、まとめるときと同じ条件を満たす画素（元絵のどの色からも `new_color_distance` より離れ、加えた色から `new_color_distance / 2` 以内）だけで、それ以外は最も近い元絵の色に寄せる。加えた色は `checks.added_colors` に残す。`preview.gif` は 255 色を超えると、多く使われる 255 色で近似する（`frames/` の PNG は近似しない）。`source` は元絵の色だけ、`none` は寄せない。
- 生成時のモデルの置き方（`generation.offload`、Colab では `scripts/colab_pilot.py --offload`）は、`model`（部品ごとに CPU オフロード）・`none`（全部 GPU）・`resident`（プロンプトを先にまとめてエンコードし、テキストエンコーダを外して 2 エキスパートと VAE を GPU に常駐）。A100 40 GB では `resident` が最も速い（`docs/experiments/speed-01.md`）。`resident` は出力がわずかに変わるので `job_id` に入る（`model.placement`）。設定で指定し、generate はジョブの指定に従う。`model` / `none` は実行時の選択で `job_id` に入らない（`results.jsonl` の `runtime.offload` に残る）。`results.jsonl` の `timings` に 1 本ごとの時間の内訳が残る
- `finalize` は元動画を読み直すだけなので、後処理設定やコマ選択を変えて何度でもやり直せる。

## VLM の判定（judge）

`dotkoma judge configs/judge.toml work/<run> ...`（Colab GPU では `scripts/colab_pilot.py --stage judge --judge-config configs/judge.toml`）。finalize の後に、機械判定が `judge.verdicts`（既定 `candidate` と `flagged`）のサンプルを VLM（既定 Qwen3.5-9B、計画と同じモデル）に見せる。

- 見せる画像は 1 枚: 元絵（S）と各コマ（1〜N、長いループは `max_frames` に間引く）を同じ倍率（`scale`、既定 12）で横に並べたもの。その下に差分の段（`diff_row`、既定 `silhouette`）: 各コマの下に、元絵の形を灰色で描き、元絵と違う画素をマゼンタで示す（`faded` は薄くした元絵の上に示す旧版、`none` は差分の段なし）。背景は白・黒・青のうち、どのコマの色からも最も遠いもの（白い湯気が白背景で消えないように）。人が目視するときも同じ画像（`judge.sheet`）を使う。判定の版の比較は `docs/experiments/judge-01.md`
- 問うことは 2 つ。`motion_matches`: 指示した動作（`[judge.criteria]` か既定の `DEFAULT_CRITERIA` の説明）がはっきり起きているか。別の変化（湯気の代わりにきらめき、泡の代わりに液が満ちる、全体の色が変わる）やほぼ静止は false。`drawing_kept`: 動く部分以外が元絵のままか。両方 true なら `accept`
- 既定の指示文は `diff_row` ごとに違う（`judge.DEFAULT_INSTRUCTIONS`。画像にない段を説明しないように）
- `request_id` はコマと元絵の画素・使ったコマ番号・`postprocess_id`・実際の指示文・判定設定（シートの描き方が変わったら `SHEET_VERSION` を上げる）のハッシュ。後処理や判定設定を変えると新しい判定になる。推論の直前にコマと元絵を読み直し、依頼を作ったときと違えば（判定中に finalize し直したなど）失敗として記録する
- 開始前に止まる設定エラー: `finalize.jsonl` がない、判定の説明（`[judge.criteria]` / `DEFAULT_CRITERIA`）がない動作がある。読めない・形式の壊れたサンプル（`sample.json` や PNG の欠け、必要な項目の欠け）は失敗として記録し、他のサンプルは判定を続ける。失敗は `--retry-failed` で再試行する
- `--sync-dir` では、開始時に同期先の `judge.jsonl` にしかない記録を手元に取り込んでから書く（新しい VM で `judge.jsonl` をアップロードし忘れても、同期先の記録を短いログで上書きしない）。合わせたログは `started_at` の順に並べ直す（同じ時刻なら手元の記録を後にする）ので、古い同期先の記録が新しい判定より後ろに来ない
- 機械判定（`finalize.jsonl`）・VLM の判定（`judge.jsonl`）・目視（`review.jsonl`）は別のファイルに残す。一致率は `scripts/judge_agreement.py configs/judge.toml work/<run> ...` で見る（機械判定の行は対象のラベル付きサンプル全部、judge の行は判定のあるサンプルで数え、件数も出す）

## 品質チェック（`sample.json` の `checks`）

`flags` が判定（`auto_verdict`）を決める。`notes` は判定を変えない目視向けの注記。

| フラグ | 種類 | 意味 |
|---|---|---|
| `wrong_size` `frame_count` `empty_frame` `all_identical` | 除外 | 形式不正・全透過コマ・動きなし（`finalize.min_frames` は 2 以上） |
| `loop_jump` | 要確認 | 最後→最初の差分が隣接コマの典型差分より大きい |
| `edge_cut` | 要確認 | 16×16 枠のすぐ外側（1 マス幅の帯）の `edge_cut_fraction`（2%）超が前景。局所的な動き（`motion = "local"`）で `edge_cut_local_fraction`（未指定なら 6% と `edge_cut_fraction` の大きい方）以下なら、フラグではなく注記 `edge_overflow`（`checks.notes`）にする |
| `background_unstable` | 要確認 | 画面の縁で背景色が保たれていない |
| `low_motion` | 要確認 | 1 周（最後→最初を含む）の隣接差分の中央値が `low_motion_pixels`（3 px）未満。1〜2 回だけ変わって後は止まっているもの |
| `detail_lost` | 要確認 | 元絵の不透明画素のうち、2 コマ目以降のどのコマにもない（各コマで最も一致する ±4px の平行移動で比べる）ものが `detail_lost_pixels`（5 px）超。小さな要素が消えたまま戻らないもの。透明になった画素だけを数え、別の色への塗り替えは数えない（閉じた目や傾いた炎のように局所動作は色を変え続けるため。塗り替えは `identity_drift` が見る） |
| `identity_drift` | 要確認 | いずれかのコマが、元絵を ±4px 平行移動したどれとも 30% 超のピクセルで食い違う（描き直し・変身） |

