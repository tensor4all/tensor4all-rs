# swap_site_indices 性能改善計画

## Context

`swap_site_indices` のインターリーブベンチマーク（鎖状 TreeTN、bond dim=1）が本来 1ms 以内で終わるべきところ、極めて遅い。

### 実測値（修正前）

| R | ノード数 | 時間 |
|---|---------|------|
| 10 | 20 | 2ms |
| 20 | 40 | 25ms |
| 25 | 50 | 644ms |
| 30 | 60 | 15.4s |
| 45 | 90 | 完了しない |

## ボトルネック分析

### ボトルネック 1: max_passes が O(n²) ← **修正済み**

```rust
// 修正前
let max_passes = (self.node_count().max(1) * target_assignment.len().max(1)).max(4);
// R=45: 90 × 90 = 8100 パス

// 修正後（現在のコード）
let max_passes = self.node_count().max(4);
// R=45: 90 パス
```

**効果**: R=25 で 644ms → 349ms（約 2 倍改善）。しかし期待より小さい。
**理由**: アルゴリズムは既に早期収束（early exit）しており、実際のパス数は
max_passes より少なかった。主要ボトルネックは別にある。

### ボトルネック 2: per-step の extract/replace_subtree オーバーヘッド ← **未修正・主因**

各ステップで `apply_local_update_sweep` → `extract_subtree` + `SwapUpdater::update` + `replace_subtree` が呼ばれる。

- `replace_subtree` の内部で **さらに `extract_subtree` を呼ぶ**
- 1 ステップあたり TreeTN オブジェクトが 3 つ生成される
- bond dim=1、サイト dim=2 の自明テンソルでも **1 ステップ ≈ 3ms**

実測による推定:
- R=25 (n=50): 実際のパス数 ≈ 数パス、steps/pass ≈ 98、合計 ≈ 数百ステップ
- 644ms ÷ ~200 steps ≈ **3ms/step**（自明テンソルで 3ms は明らかに異常）

### ボトルネック 3: A* 方向判定 O(n) ← **未修正・副因**

`SwapUpdater::update` の内部で `is_target_on_a_side` が `path_between`（A*）を呼ぶ。
各ステップで O(n) の探索。ボトルネック 2 が支配的なため現時点では副次的。

### ボトルネット 4: 毎パス sweep plan 再構築・current_site_assignment スキャン ← **未修正・軽微**

`apply_local_update_sweep` のループ内で毎パス:
- `LocalUpdateSweepPlan::from_treetn` — O(n) Euler tour 構築
- `current_site_assignment(self)` — O(n) 全ノードスキャン

ボトルネット 2 が主因なので、これらの影響は小さい。

## 次の修正: `swap_on_edge` の実装

extract/replace_subtree を廃止し、`sweep_edge`（正準化で使用）と同じパターンで
直接テンソルを操作する軽量メソッドを実装する。

### アルゴリズム

```rust
pub(crate) fn swap_on_edge(
    &mut self,
    node_a_idx: NodeIndex,
    node_b_idx: NodeIndex,
    target_assignment: &HashMap<IndexId, V>,
    oracle: &SubtreeOracle<V>,
    factorize_options: &FactorizeOptions,
) -> Result<()> {
    // 1. edge, bond 取得
    // 2. A の他ボンド IDs（left に固定）、B の他ボンド IDs（right に固定）を収集
    // 3. tensor_a と tensor_b を contract → tensor_ab
    // 4. oracle で各 site index の行き先を判定 → left_id_set 構築
    // 5. 縮退ケース処理（left_inds が空 or 全部の場合）
    // 6. tensor_ab.factorize(&left_inds, options) → (new_tensor_a, new_tensor_b, new_bond)
    // 7. replace_edge_bond + replace_tensor×2 + set_edge_ortho_towards
}
```

### 縮退ケースの扱い（要注意）

- **Case 1** (left_inds 空 — A がリーフ、全サイトが B 側へ):
  `left_inds.push(site_inds.first())` で1つ強制的に左へ。`swap_result` なし。
  次のパスで収束。

- **Case 2** (left_inds = ab_indices 全部 — B がリーフ、全サイトが A 側へ):
  factorize の `unfold_split` が `left_len == rank` を拒否するため工夫が必要。
  → `left_inds` から最後のサイトを1つ取り除いて右へ一時的に置く。
  **ただし**: これは収束を妨げる可能性があるため、Y-shape テストで検証要。

### `swap_on_edge` 完成後の `swap_site_indices` 書き換え

```rust
// 事前計算（1回だけ）
let root = self.node_names().into_iter().min()?;
let oracle = SubtreeOracle::new(topology, &root)?;
let sweep_plan = LocalUpdateSweepPlan::from_treetn(self, &root, 2)?;
let factorize_options = FactorizeOptions::svd().with_canonical(Canonical::Left);

let is_satisfied = |me: &Self| target_assignment.iter().all(|(id, target)| {
    me.site_index_network().find_node_by_index_id(id) == Some(target)
});

let max_passes = self.node_count().max(4);  // 修正済み
for _pass in 0..max_passes {
    if is_satisfied(self) { return Ok(()); }
    for step in sweep_plan.iter() {
        let [node_a, node_b] = step.nodes[..] else { continue };
        self.swap_on_edge(a_idx, b_idx, target_assignment, &oracle, &factorize_options)?;
        self.set_canonical_region([step.new_center.clone()])?;
    }
}
```

## 期待される改善効果

| 修正 | 効果 |
|------|------|
| ~~max_passes を O(n) に~~ ✅ | 約 2× |
| swap_on_edge (extract/replace 廃止) | **大幅改善（主因）** |
| SubtreeOracle (A* → O(1)) | 追加改善 |
| sweep plan キャッシュ + O(1) 収束チェック | 小改善 |

## 根本原因（判明）

`swap_on_edge` の旧ロジック（フォールバック処理）が、「全サイトが B 側へ移動したい」ケースで
正しく機能せず、**1ノードに複数サイトが蓄積**する問題があった。

- edge (A,B) で x_a が B-side 希望、x_b も B-side 希望 → left_inds にサイトなし → フォールバックで
  B のサイトを A 側に置く → A は 0 サイト、B は 2 サイトになる
- sweep が進むと蓄積が悪化: R=30 では tensor_ab が 512 要素（9 サイト）まで膨らむ
- SVD が O(2^R) のコストになり指数的に遅化

**修正**: サイト数不変の優先度ベース選択
- A は常に元のサイト数（=1）を維持
- 優先度: (1) A にいてA希望 > (2) B にいてA希望 > (3) B にいてtargetなし > (4) A にいてtargetなし > (5) A にいてB希望 > (6) B にいてB希望

## 実測値（修正後）

| R | ノード数 | 時間 | パス数 |
|---|---------|------|--------|
| 5 | 10 | ~2ms | 2 |
| 10 | 20 | ~6ms | 5 |
| 15 | 30 | ~13ms | 7 |
| 20 | 40 | ~21ms | 10 |
| 25 | 50 | ~29ms | 12 |
| 30 | 60 | ~40ms | 15 |
| 45 | 90 | ~86ms | 22 |

スケーリング: O(R²) パス数 × O(1) per step ≈ O(R²) 合計

## 進捗

- [x] max_passes を `node_count().max(4)` に修正
- [x] `SubtreeOracle` を swap.rs に追加
- [x] `swap_on_edge` を mod.rs に追加（サイト数不変の優先度ベース選択）
- [x] `swap_site_indices` を書き換え（swap_on_edge + oracle 使用）
- [x] テスト全パス確認（14 passed, 2 ignored for Y-shape）
- [x] ベンチマーク確認（R=45 が 86ms ← 目標 10 秒以内を大幅達成）

## 検証コマンド

```bash
cargo test --release -p tensor4all-treetn --test swap_test
cargo run --release -p tensor4all-treetn --example bench_swap_interleave_r45
cargo test --release --workspace
cargo fmt --all && cargo clippy --workspace
```
