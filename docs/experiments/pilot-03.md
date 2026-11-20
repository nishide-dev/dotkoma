# Pilot 03: 局所動作 8 種を A14B + Lightning で 44 本（2026-11-21）

Pilot 02（`pilot-02.md`）で局所動作（炎・手振り・湯気）が A14B で出ることがわかったので、局所動作を 8 種に広げて試した。設定は `configs/pilot-03.toml`（モデル・推論設定は Pilot 02 と同じ）、生成物（`work/pilot-03/`）は Git 管理外。後処理は PR #5・#6 の規則（パレット `source+stable`、局所動作の小さなはみ出しは注記）。

## 条件

- 動作 8 種 × 題材 2〜3（計 22 題材、各 1 スプライト）× プロンプト 2 通り = 44 本。同じ動作の 2 通りはシードを共有する
- 動作: flicker（炎）、steam（湯気）、smoke（煙）、bubble（泡）、sparkle（きらめき）、flutter（旗のはためき）、blink（まばたき）、wag（尻尾を振る）。すべて `motion = "local"`
- Colab A100 40 GB。H100 は割り当て不可、A100 は 1 回目に Service Unavailable で、再試行で確保

## 実測

| 項目 | 値 |
|---|---|
| 生成 1 本 | 中央値 60.5 s（初回のみ 103 s） |
| モデルロード | 771 s（新しい VM で重みのダウンロードから。Pilot 02 は 417–428 s） |
| 生成全体 | 45 分（44 本、失敗 0） |
| 消費ユニット | 約 10.3（セッション確保からダウンロード完了まで） |

生成後の `colab exec`（出力をまとめる tar）が応答しなくなったため、`colab download` で 1 本ずつ取得した。

## 結果

機械判定: candidate 27 / flagged 16 / reject 1。

以下の目視は 6 コマを並べた静止画での判断で、アニメーションとしては確認していない。

| 動作 | 題材 | 機械判定（v0 / v1） | 目視 |
|---|---|---|---|
| flicker | candle, tea light, torch | 6 本とも candidate | 6 本とも炎だけが揺れる |
| steam | cup of herbal tea, soup bowl | 4 本とも candidate | 4 本とも湯気が立つ |
| steam | mug of coffee | candidate / candidate | **使えない**。1 コマ目以降、カップの上のクッキーが消える |
| smoke | chimney | candidate / candidate | 煙が立つ |
| smoke | fire pit | flagged / candidate | v0 は煙が出るが背景が乱れる。v1 は炎が揺れるだけで煙は出ない（動作ラベルと合わない） |
| bubble | cooking cauldron | candidate / candidate | 鍋の中が泡立つ |
| bubble | potion decanter | candidate / candidate | 光沢が少し動く程度。泡には見えない |
| bubble | round potion flask | flagged / flagged | 白い瓶に液体が満ちていく（変化であって泡ではない） |
| sparkle | mana crystal | candidate / candidate | 表面の光がゆらぐ |
| sparkle | crown | candidate / reject | v0 もほぼ静止（隣接差分 0〜7 px）。v1 は完全に静止 |
| sparkle | star-cut gem | flagged / flagged | 星形に変形する |
| flutter | castle flag, waving flag | 4 本とも flagged（identity_drift） | 布の形が変わる。はためきとして使えるかはアニメーションで要確認 |
| flutter | flag on pole | flagged / flagged | 布に元絵にない模様が現れる |
| blink | owl | candidate / flagged | v0 は目を閉じて開くように見える |
| blink | penguin | flagged / flagged | v0 は目を閉じて開くが、最後のコマが元に戻りきらない（loop_jump）。v1 はくちばしが変形 |
| blink | frog | flagged / candidate | 16×16 では変化が判別しにくい |
| wag | fox | candidate / candidate | v0 は尻尾が動く。v1 は顔の向きも変わる |
| wag | dog | candidate / flagged | 尻尾の動きは小さい |

## わかったこと

44 本・シード各 1 での観察。

1. **炎・湯気・煙・泡（鍋）・光のゆらぎは安定して出る。** flicker 6/6、steam（2 題材）4/4、chimney 2/2、cauldron 2/2、mana crystal 2/2 が目視で使えそう。
2. **機械判定の candidate 27 本のうち、目視で使えないものが少なくとも 5 本ある。**
   - ほぼ静止（crown v0）。`all_identical` は完全に同一のときしか付かない
   - 小さな要素が消える（mug of coffee のクッキー）。消えるのが 1 コマ目の直後なので、ずれを許す identity_drift の閾値に届かない
   - 動作ラベルと違う動き（fire pit v1、fox v1）
3. **flutter は identity_drift で全部 flagged になる。** 布の変形そのものが動作なので、元絵との食い違いを測る指標と相性が悪い。
4. 題材の選び方の影響が大きい。泡は中身の見える鍋では出るが、閉じた瓶では出ない。

## 次の候補

- 機械判定に「動きが小さすぎる」（隣接差分の最大が数 px 以下）と「元絵の要素が消えたまま」（元絵の不透明画素が 2 コマ目以降ずっと欠ける）を加える
- flutter のように形そのものが変わる動作は、identity_drift の閾値を動作ごとに変えられるようにする
- 使える組み合わせ（flicker / steam / smoke / bubble / sparkle の一部）で題材を増やして量産する
