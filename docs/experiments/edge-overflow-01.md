# Edge overflow 01: 局所動作の小さなはみ出しを注記にする（2026-11-21）

Pilot 02（`pilot-02.md`）では、見た目では使える局所動作（toy robot wave の腕、cup of coffee steam の湯気）に `edge_cut` が付き、flagged になっていた。`motion = "local"` の動作で、はみ出しの割合が `edge_cut_local_fraction`（6%）以下なら、フラグではなく注記 `edge_overflow` にした。Pilot 02 の 12 本を後処理だけやり直した（再生成なし。パレットは `source+stable`）。

## Pilot 02 の `edge_cut_fraction`（16×16 枠の外側 1 マス幅の帯のうち前景の割合）

| 題材 | 動作 | motion | 値 | 目視 |
|---|---|---|---|---|
| candle | flicker | local | 0.001 | 使える |
| toy robot | wave | local | 0.051 | 使える（腕が少し枠外へ） |
| cup of coffee | steam | local | 0.032 | 使える（湯気が上端で切れる） |
| slime | bob | global | 0.046 | 使えない |
| toy robot | hop | global | 0.077 | 使えない |
| cup of coffee | bob | global | 0.096 | 使えない |
| slime | hop | global | 0.108 | 使えない |
| ghost | bob | global | 0.127 | 使えない |
| cat | hop | global | 0.324 | 使えない |

値だけでは「少しはみ出す局所動作」と「破綻した全体移動」が重なる（0.046 と 0.051）。motion の区別を併用した。

## 結果

機械判定の candidate は candle flicker / toy robot wave / cup of coffee steam の 3 本になり、目視で使えると判断した 3 本と一致した（以前は candidate 2 本のうち cat blink が白目の消失による見落とし、wave と steam は flagged）。wave と steam には `notes: ["edge_overflow"]` が残る。16×16 の出力では、はみ出した部分（腕の先・湯気の上端）は切れている。

12 本・単一シードでの観察で、6% という値もこの 12 本から決めた。本数を増やしたら見直す。
