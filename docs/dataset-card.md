# Dotkoma dataset card (draft)

公開前に埋める。数値はすべて実測で記入する。

## 元データ
- Pixel Art Atlas 1.4M — https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m
- revision: `bf78f8c8cbafa6d07718111a94d41e2dd79ddaa7`
- 各サンプルは `source.row_id` / `source.subject` で元スプライトに辿れる。

## 生成
- 教師モデル: （model_id とリビジョンを results.jsonl から転記）
- 推論設定・シード: jobs.jsonl / results.jsonl を同梱
- generation_method: `video_model`（将来 `pixel_ops` を追加予定）

## 後処理
- 単色キー背景の除去、参照レイアウトの固定グリッドで 16×16 に縮小、元絵パレットへのスナップ（全コマ共通）
- コマ選択と表示時間（ms）は sample.json に記録

## 採否と分割
- 機械判定（finalize.jsonl）と目視採否（review.jsonl）を分けて記録
- 分割は subject 単位のハッシュ（Atlas 既定の分割は使わない）
- 採用率: 全体 / 題材タイプ × 動作 （未測定）

## 利用条件
- 元スプライト・キャプション: Angais が CC0 1.0 で提供（Angais が放棄できる範囲）。Krea 2 Community License と利用方針への留意。
- Wan2.2 の重みは Apache 2.0、生成物への権利主張なし（モデルカード参照）。
- 本データセットで追加した部分（動画・フレーム・メタデータ）の条件: 未決定
- 既存キャラクター・ロゴの混入は除外する。

## 既知の偏り
（試作後に記入）
