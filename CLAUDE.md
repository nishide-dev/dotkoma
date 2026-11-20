# CLAUDE.md

## プロジェクト概要

Dotkoma は、16×16 のドット絵ループアニメーションの学習用データセットを作るプロジェクト。

- Pixel Art Atlas 1.4M（`Angais/pixel-art-atlas-1.4m`）の 16×16 透過スプライトを元絵にする
- 動画モデル（Wan2.2）で画像→動画の候補を合成する
- 16×16 に戻しても動きと見た目が成立するものだけを採用する
- 学習用の正本は順序付き RGBA フレーム列 + 表示時間（ms）。GIF は閲覧用
- 次段階（初期版の対象外）は、Tessera 風の色トークンを時間方向に広げた小型のドット絵動画モデル

パイプラインは 3 工程で、境界はファイルだけ（API サーバーは使わない）。

| 工程 | 実行場所 | 出力（`work/<run>/`） |
|---|---|---|
| `prepare` | 手元 CPU | `inputs/*.png`, `jobs.jsonl` |
| `generate` | Colab GPU | `raw/<job_id>.mp4`, `results.jsonl`（追記専用） |
| `finalize` | 手元 CPU | `final/<job_id>/{frames/,sample.json,preview.gif}`, `finalize.jsonl` |

ファイル契約・品質チェックの詳細は `README.md`。設計の背景はローカルの `dotkoma-project-report.pdf`（Git 管理外）。

## 開発コマンド

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run dotkoma {prepare,generate,finalize} configs/smoke.toml   # GPU なしで一周
```

## 設計上の約束（変更するときは理由を明示する）

- 手元環境（uv, Python 3.11）に PyTorch / CUDA / diffusers を入れない。GPU 依存は `scripts/colab_pilot.py` と `generate.py` の遅延 import に閉じ込める
- `job_id` は生成に効く入力すべてのハッシュ。生成条件を変えたら必ず別 ID になるようにする
- 完了は「出力と記録が揃った」時点。ファイルは一時ファイル → rename で書き、部分ファイルを完了扱いしない
- 後処理（finalize）は元動画を使い回す。後処理の変更で再生成を要求しない
- 全コマで同じグリッド・同じパレット規則を使う（コマ単位の中央揃え・パレット生成はしない）
- データの分割は subject 単位。Atlas 既定の分割は使わない
- `requested_action`（指示）と採用ラベルを分ける。機械判定（`finalize.jsonl`）と目視採否（`review.jsonl`）も分ける
- Git にはコード・設定・小さなテスト素材だけを入れる。モデル重み、元動画、実行記録（`work/`）、認証情報、PDF は入れない
- コードは MIT。データの利用条件はコードと分けて `docs/dataset-card.md` に書く

## Git / GitHub のルール

- リポジトリ: `nishide-dev/dotkoma`
- コミットメッセージは Conventional Commits 形式、英語で書く: `type(scope): summary`
  - type: `feat` / `fix` / `docs` / `test` / `refactor` / `chore` / `perf` / `ci`
  - scope は対象（`prepare`, `generate`, `finalize`, `config`, `colab`, `docs` など）。全体に及ぶ場合は省略可
  - 例: `feat(prepare): add prompt variants per action`
- Issue / PR のタイトルは英語（コミットと同じ `type(scope): summary` 形式を推奨）、本文は日本語
- PR 本文には「何を・なぜ」「確認方法（実行したコマンドと結果）」を書く
- `main` へ直接 push しない。作業はブランチを切って PR にする（ブランチ名例: `feat/prompt-variants`）
- コミット前に `uv run pytest` と `uv run ruff check .` を通す
