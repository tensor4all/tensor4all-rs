# swap_site_indices 性能改善計画

## Context

`swap_site_indices` のインターリーブベンチマーク（R=45, 90ノード鎖, bond dim=1）が本来1秒以内で終わるべきところ、膨大な時間を要している。

根本原因は以下の3点:

1. **per-step コスト**: 各ステップで `extract_subtree`/`replace_subtree`（replace内でさらにextract）+ A*探索（O(n)）
2. **per-sweep コスト**: sweep plan再構築、`current_site_assignment`全スキャン
3. **アルゴリズム**: バブルソート的（~45パス必要）だが、各パスが軽量になれば十分高速

## 方針: 軽量な `swap_on_edge` + 事前計算によるO(1)方向判定

`sweep_edge`（正準化で使用、`mod.rs:504`）と同パターンの軽量swap操作を実装する。
`extract_subtree`/`replace_subtree` の重いフレームワークを経由せず、
`replace_tensor` + `replace_edge_bond` で直接テンソルを操作する。

## 変更ファイル

1. **`crates/tensor4all-treetn/src/treetn/swap.rs`** — ヘルパー追加
2. **`crates/tensor4all-treetn/src/treetn/mod.rs`** — `swap_on_edge` 追加 + `swap_site_indices` 書き換え

## Step 1: 方向判定の事前計算 (`swap.rs`)

### 問題

`is_target_on_a_side` (swap.rs:205) が毎ステップ `path_between` (A*, O(n)) を呼ぶ。
90ノード鎖では ~16,000回の A* 呼び出しが発生。

### 解決: `SubtreeOracle` 構造体

DFS の entry/exit timestamp を使い、任意の edge に対して「target がどちら側か」を O(1) で判定。

```rust
/// Pre-computed subtree membership for O(1) direction queries.
/// For any edge (A, B), determines whether a target node T is on A's side or B's side.
pub(crate) struct SubtreeOracle<V> {
    parent: HashMap<V, V>,         // node -> parent (root has no entry)
    entry: HashMap<V, usize>,      // DFS entry timestamp
    exit: HashMap<V, usize>,       // DFS exit timestamp
}

impl SubtreeOracle<V> {
    /// Build from tree topology. O(n) DFS.
    pub fn new(topology: &NodeNameNetwork<V>, root: &V) -> Result<Self>;

    /// O(1): Is `node` a descendant of `ancestor` (inclusive)?
    fn is_descendant(&self, node: &V, ancestor: &V) -> bool {
        entry[ancestor] <= entry[node] && exit[node] <= exit[ancestor]
    }

    /// O(1): Is `target` on B's side of edge (A, B)?
    /// Uses parent info to determine which is child in the rooted tree.
    pub fn is_on_b_side(&self, node_a: &V, node_b: &V, target: &V) -> bool;
}
```

**正しさの保証**: DFS timestamp による subtree membership はツリートポロジーの静的性質。
swap 中にトポロジーは変わらないため、全パスで共通に使える。
`SwapPlan.direction_map()` と異なり、中間状態での index 変位にも正しく対応。

### その他の変更

- `normalize_edge` (swap.rs:70) を `pub(crate)` に変更

## Step 2: `swap_on_edge` メソッド (`mod.rs`, ~line 597 の後)

### 問題

現在の per-step 処理:
- `extract_subtree` → 新 TreeTN 生成 + tensor clone + 接続性DFS
- `SwapUpdater::update` → contract + A* × 2 + SVD + outer_product
- `replace_subtree` → **内部で再度 `extract_subtree`** + topology 比較 + tensor 置換

→ 1ステップあたり TreeTN が3つ生成される

### 解決: `sweep_edge` パターンの直接操作

```rust
/// Lightweight swap operation on a single edge.
/// Contracts tensors at node_a and node_b, distributes site indices
/// based on target assignment, and factorizes back.
///
/// Analogous to `sweep_edge` for canonicalization, but handles
/// site index redistribution between two nodes.
pub(crate) fn swap_on_edge(
    &mut self,
    node_a_idx: NodeIndex,
    node_b_idx: NodeIndex,
    target_assignment: &HashMap<<T::Index as IndexLike>::Id, V>,
    oracle: &SubtreeOracle<V>,
    factorize_options: &FactorizeOptions,
) -> Result<()>
```

**アルゴリズム:**

1. edge, bond_ab 取得
2. tensor_a, tensor_b の site indices 収集（bond_ab 以外の external indices）
3. `left_id_set` 構築（A 側に残すべき index IDs）:
   - A の他ノードへの bond indices → 常に left（A の構造的接続）
   - 各 site index I について:
     - `target_assignment` に I がない → 現在の側に留まる
     - `oracle.is_on_b_side(A, B, target)` → false なら left（A側）、true なら除外（B側へ）
4. tensor_a と tensor_b を `T::contract` で結合
5. `left_inds` = contracted tensor の indices のうち `left_id_set` に含まれるもの
6. 縮退ケース処理（既存 swap.rs:361-369 と同じロジック）
7. `tensor_ab.factorize(&left_inds, factorize_options)` (SVD)
8. **直接更新**（`sweep_edge` と同パターン）:
   ```
   self.replace_edge_bond(edge, new_bond)
   self.replace_tensor(node_a_idx, new_tensor_a)
   self.replace_tensor(node_b_idx, new_tensor_b)
   self.set_edge_ortho_towards(edge, Some(node_b_name))
   ```

### `sweep_edge` との違い

| | `sweep_edge` | `swap_on_edge` |
|---|---|---|
| 目的 | src テンソルを factorize → dst に吸収 | 2テンソルを contract → site indices 再配置 → factorize |
| テンソル操作 | factorize(src) + contract(dst, right) | contract(A, B) + factorize(AB) |
| left_inds | src の bond_to_dst 以外すべて | oracle による方向判定で動的決定 |

## Step 3: `swap_site_indices` 書き換え (`mod.rs`, lines 1291-1363)

### Before (現在の実装)

```rust
// _plan は検証のみに使用、破棄される
let _plan = SwapPlan::new(&current, target_assignment, topology)?;

while pass < max_passes {
    let current = current_site_assignment(self);        // 毎回 O(n) 全スキャン
    if is_target_satisfied(&current) { return Ok(()) }
    let plan = LocalUpdateSweepPlan::from_treetn(...);  // 毎回 Euler tour 再構築
    apply_local_update_sweep(self, &plan, &mut updater); // 重い extract/replace
    pass += 1;
}
```

### After (新実装)

```rust
// 1. 検証 (既存)
let current = current_site_assignment(self);
let _plan = SwapPlan::new(&current, target_assignment, topology)?;

// 2. 正準化 (既存)
if !self.is_canonicalized() { self.canonicalize_mut(...)?; }

// 3. 事前計算 (1回だけ)
let root = self.canonical_region().iter().next().cloned()?;
let oracle = SubtreeOracle::new(self.site_index_network().topology(), &root)?;
let sweep_plan = LocalUpdateSweepPlan::from_treetn(self, &root, 2)?;  // キャッシュ
let factorize_options = build_factorize_options(options);

// 4. 収束判定 (find_node_by_index_id で O(1) per index)
let is_satisfied = |me: &Self| -> bool {
    target_assignment.iter().all(|(id, target)| {
        me.site_index_network().find_node_by_index_id(id)
            == Some(target)
    })
};

// 5. メインループ
let max_passes = (self.node_count().max(1) * target_assignment.len().max(1)).max(4);
for _pass in 0..max_passes {
    if is_satisfied(self) { return Ok(()) }
    for step in sweep_plan.iter() {
        let [node_a, node_b] = &step.nodes[..] else { continue };
        let a_idx = self.node_index(node_a)?;
        let b_idx = self.node_index(node_b)?;
        self.swap_on_edge(a_idx, b_idx, target_assignment, &oracle, &factorize_options)?;
        self.set_canonical_region([step.new_center.clone()])?;
    }
}
// 6. 最終チェック
if is_satisfied(self) { return Ok(()) }
Err(anyhow!("did not converge"))
```

### 削除されるもの

- `SwapUpdater` の使用（構造体自体は残す。`LocalUpdater` trait の実装として他で使われる可能性）
- `apply_local_update_sweep` の呼び出し（swap からのみ。truncation等は引き続き使用）
- `is_target_on_a_side` の呼び出し（`SubtreeOracle` に置換）

## 性能比較

| | Before | After |
|---|---|---|
| 方向判定 | A* (O(n)) × ~16K回 | O(1) × ~16K回 |
| テンソル操作/step | extract×2 + replace + outer_product | contract + factorize + replace_tensor×2 |
| TreeTN生成/step | 3個 | 0個 |
| sweep plan | 毎パス再構築 | 1回構築、キャッシュ |
| current scan | 毎パス O(n) | find_node_by_index_id O(1) per index |
| **全体 (n=90)** | **~45 × 178 × O(90)** | **~45 × 178 × O(1)** |

## 検証

```bash
# 既存テストが全パス（swap関連9テスト）
cargo nextest run --release -p tensor4all-treetn --test swap_test

# ベンチマーク: 10秒以内（目標: 1秒以内）
cargo run --release -p tensor4all-treetn --example bench_swap_interleave_r45

# 全体テスト + lint
cargo nextest run --release --workspace
cargo fmt --all
cargo clippy --workspace
```
