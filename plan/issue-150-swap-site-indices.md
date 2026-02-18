# Issue #150: サイトインデックススワップユーティリティ実装プラン

## 概要

TreeTN のサイトインデックスを隣接ノード間でスワップし、目標の配置に再配列する機能を実装する。

## 既存インフラの把握

実装に活用する既存の主要コンポーネント:

| コンポーネント | ファイル | 役割 |
|---|---|---|
| `LocalUpdater` trait | `crates/tensor4all-treetn/src/treetn/localupdate.rs` | 2-site スイープの抽象化 |
| `TruncateUpdater` | `crates/tensor4all-treetn/src/treetn/localupdate.rs` | `LocalUpdater` の参照実装 |
| `apply_local_update_sweep` | `crates/tensor4all-treetn/src/treetn/localupdate.rs` | スイープの実行エンジン |
| `LocalUpdateSweepPlan` | `crates/tensor4all-treetn/src/treetn/localupdate.rs` | オイラーツアーベースのスイープ計画 |
| `SiteIndexNetwork` | `crates/tensor4all-treetn/src/site_index_network.rs` | サイトインデックス→ノードのマッピング管理 |
| `path_between` | `crates/tensor4all-treetn/src/node_name_network.rs` | 2ノード間の最短パス（木なので一意） |
| `canonicalize` / `canonicalize_mut` | `crates/tensor4all-treetn/src/treetn/canonicalize.rs` | 正準化 |
| `extract_subtree` / `replace_subtree` | `crates/tensor4all-treetn/src/treetn/localupdate.rs` | サブツリーの抽出と置換 |
| `fuse_to` / `split_to` | `crates/tensor4all-treetn/src/treetn/transform.rs` | 構造変換（参考実装パターン） |

## 設計方針

### アプローチ: `LocalUpdater` ベースのスイープ

Issue の提案と既存インフラを踏まえ、以下の方針をとる:

1. **動的判定**により、2-site update ごとに「このエッジでどの site index を左右に移すべきか」を決める（複数ホップ移動も自然に扱う）
2. スイープ計画は既存の `LocalUpdateSweepPlan` (nsite=2) をそのまま使用し、各エッジが2回訪問される点は `processed_edges` 等で抑制する
3. **`SwapUpdater`** を `LocalUpdater` trait として実装し、`apply_local_update_sweep` で実行
4. スワップ不要なエッジではテンソルをそのまま返す（no-op）
5. （任意・デバッグ用）事前に `SwapPlan` を計算して「総移動距離の見積り」「どのエッジで動くはずか」の可視化を可能にする

### 代替案の検討

| アプローチ | メリット | デメリット |
|---|---|---|
| **A: LocalUpdater ベース (採用)** | 既存インフラ活用、正準化管理が自動 | スワップ不要なエッジも通過 |
| B: fuse_to/split_to の組み合わせ | 概念的にシンプル | fuse→split で新しいボンド生成、制御が粗い |
| C: 手動でエッジを一つずつ処理 | 完全な制御 | 正準化管理を自前で実装、コード重複多い |

**アプローチ A を採用する理由**: `TruncateUpdater` と同じパターンで実装でき、正準化の追跡やサブツリー抽出/置換が自動化される。スワップ不要なステップは `update()` 内でサブツリーをそのまま返すことで効率的にスキップできる。

## データ構造

### `SwapPlan<V, I>`

```rust
/// スワップ計画: 初期配置から目標配置へのスワップ手順
pub struct SwapPlan<V, I> {
    /// エッジごとのスワップ操作 (key: 正規化されたエッジ (min, max))
    swaps: HashMap<(V, V), Vec<SwapStep<V, I>>>,
}

/// 1回のスワップ操作
pub struct SwapStep<V, I> {
    /// スワップが発生するエッジ (node1, node2)
    pub edge: (V, V),
    /// node1 → node2 に移動するサイトインデックス (None = 移動なし)
    pub index_to_node2: Option<I>,
    /// node2 → node1 に移動するサイトインデックス (None = 移動なし)
    pub index_to_node1: Option<I>,
}
```

### `SwapUpdater<V, I>`

```rust
/// LocalUpdater 実装: スワップ計画に従いサイトインデックスを再配置
pub struct SwapUpdater<V, I> {
    /// 実行するスワップ計画
    swap_plan: SwapPlan<V, I>,
    /// 打ち切りオプション
    max_rank: Option<usize>,
    rtol: Option<f64>,
}
```

## アルゴリズム詳細

### Step 1: スワップ計画の計算

**入力:**
- `current_assignment: HashMap<I::Id, V>` — 各サイトインデックスの現在のノード
- `target_assignment: HashMap<I::Id, V>` — 各サイトインデックスの目標ノード
- `topology: &NodeNameNetwork<V>` — ツリーのトポロジー

**アルゴリズム:**
1. 移動が必要なインデックスを特定（current != target）
2. 各インデックスについて `path_between(current_node, target_node)` で経路を取得
3. 経路上の各エッジにスワップステップを登録
4. バリデーション:
   - `target_assignment` は **partial 指定を許す**（未指定の index は現状維持）
   - `target_assignment` に含まれる index id がすべて現在のネットワークに存在するか
   - 各目標ノードが木上に存在するか
   - 到達不可能な配置がないか（木の場合は常に到達可能）
   - （あれば）同じ index id が複数ノードに割り当てられていないか（入力の重複/不整合）

**計画の最適化:**
- 同一エッジ上の複数スワップはまとめて実行可能
- 独立なエッジ（共有ノードなし）のスワップは順序に依存しない

### Step 2: スワップの実行 (`SwapUpdater::update`)

`TruncateUpdater` のパターンに従い:

```
2-site update (ノード A, B):
1. edge (A, B) に対するスワップ計画を参照
2. スワップ計画がない → サブツリーをそのまま返す（no-op）
3. スワップ計画がある場合:
   a. テンソル A と B を縮約 → テンソル AB
   b. AB のインデックスを再配分:
      - A に残すインデックス = (A の現在のサイトインデックス - node1→node2) + (node2→node1)
                               + (A の境界ボンドインデックス)
   c. AB を factorize (SVD + truncation) → 新テンソル A', B'
   d. サブツリーのテンソルとボンドを更新
   e. `replace_tensor`/`replace_subtree` により site space は **テンソルの外部インデックスから自動再計算**される（手動更新を基本不要にする）
```

**重要な注意点:**
- `apply_local_update_sweep` はオイラーツアーに基づき各エッジを2回訪問する
- スワップは「最初の訪問時」にのみ実行し、「2回目の訪問時」はno-opとする
- これにより、1回のスイープで全スワップが完了する

### Step 3: サイトインデックスの追跡

スワップ実行中、各ノードのサイトインデックスは動的に変化する。`SwapUpdater` 内部で状態を追跡する:

```rust
/// 現在のインデックス割り当て（スワップ実行中に更新）
current_assignment: HashMap<I::Id, V>,
/// 完了済みスワップの追跡
completed_swaps: HashSet<(V, V)>,  // エッジ単位
```

### Step 4: 複数ホップの処理

サイトインデックスが複数エッジをまたいで移動する場合（例: A→B→C）:
- パス上の各エッジにスワップを登録
- オイラーツアーの順序により、B→C のスワップは A→B のスワップ後に実行される
  - ただし、オイラーツアーの順序が必ずしもパス順とは限らない
  - **対策**: `SwapUpdater` が内部状態で「現在どのノードにどのインデックスがあるか」を追跡し、各ステップで動的にスワップを判断する

**動的スワップ判定アルゴリズム:**
```
update(subtree [A, B], new_center = B):
  1. 現在 A にあるインデックスのうち、目標ノードが B 側のサブツリーにあるもの → A→B に移動
  2. 現在 B にあるインデックスのうち、目標ノードが A 側のサブツリーにあるもの → B→A に移動
  3. 上記に基づき factorize の left_inds を決定
  4. current_assignment を更新
```

この「動的判定」方式により、スワップ計画を事前に詳細に計算する必要がなくなり、実装がシンプルになる。

## 公開API

```rust
// --- メインのパブリック API ---

/// サイトインデックスのスワップオプション
pub struct SwapOptions {
    /// SVD 打ち切りの最大ランク
    pub max_rank: Option<usize>,
    /// SVD 打ち切りの相対許容誤差
    pub rtol: Option<f64>,
}

impl<T, V> TreeTN<T, V> {
    /// サイトインデックスを目標配置に再配列する
    ///
    /// # Arguments
    /// * `target_assignment` - サイトインデックスID → 目標ノード名 のマッピング
    /// * `options` - スワップオプション (truncation パラメータ)
    ///
    /// # Returns
    /// スワップ後の TreeTN (in-place 変更)
    pub fn swap_site_indices(
        &mut self,
        target_assignment: &HashMap<I::Id, V>,
        options: &SwapOptions,
    ) -> Result<()>;
}

// --- 低レベル API (上級ユーザー向け、テスト向け) ---

impl<V, I> SwapPlan<V, I> {
    /// 初期配置と目標配置からスワップ計画を生成
    pub fn new(
        current_assignment: &HashMap<I::Id, V>,
        target_assignment: &HashMap<I::Id, V>,
        topology: &NodeNameNetwork<V>,
    ) -> Result<Self>;

    /// 特定のエッジにスワップが必要か
    pub fn has_swaps_at(&self, edge: &(V, V)) -> bool;

    /// スワップが必要なエッジの集合
    pub fn edges_with_swaps(&self) -> HashSet<(V, V)>;
}
```

## ファイル構成

```
crates/tensor4all-treetn/src/
├── treetn/
│   ├── mod.rs              # swap モジュールの追加
│   └── swap.rs             # NEW: SwapPlan, SwapUpdater, swap_site_indices
└── ...

crates/tensor4all-treetn/tests/
└── swap_test.rs            # NEW: インテグレーションテスト
```

## 実装ステップ

### Phase 1: コアデータ構造とバリデーション
1. `treetn/swap.rs` を作成、`SwapPlan`, `SwapStep`, `SwapOptions` の定義
2. `SwapPlan::new()` の実装（パス計算、バリデーション）
3. バリデーションロジック:
   - `target_assignment` は partial 指定可（未指定は現状維持）
   - `target_assignment` に含まれる index id がすべて現在のネットワークに存在するか
   - target のノードがすべてトポロジー上に存在するか
   - 各インデックスのパスが計算可能か

### Phase 2: SwapUpdater の実装
4. `SwapUpdater` 構造体の定義
5. `LocalUpdater<T, V> for SwapUpdater` の実装:
   - `update()`: 動的スワップ判定 + contract + factorize
   - テンソルのインデックス再配分ロジック
6. `TreeTN::swap_site_indices()` の実装:
   - 正準化（まだ正準化されていない場合）
   - `LocalUpdateSweepPlan` の生成
   - `apply_local_update_sweep` の呼び出し
   - `SiteIndexNetwork` は `replace_tensor`/`replace_subtree` がテンソルから更新する（手動更新しない方針）

### Phase 3: テスト
7. ユニットテスト（`swap.rs` 内 `#[cfg(test)]`）:
   - `SwapPlan` の計算が正しいか
   - 2ノードチェーンでの単純スワップ
   - 無効な target_assignment のエラーハンドリング
8. インテグレーションテスト（`tests/swap_test.rs`）:
   - 2〜4サイトのリニアチェーンスワップ
   - **1鎖TTでの 2R ビット交互配置変換**（例: `x0..x{R-1}, y0..y{R-1}` を `x0,y0,x1,y1,...` に再配置できるか）
     - 期待: swap 前後で `contract_to_tensor()` が一致（テンソル演算側の自動 permutation を前提に比較する）
     - 期待: swap 後の index_id → node の割り当てが target 通り
   - Y字型ツリーでのスワップ（パスをまたぐ移動）
   - **正確性テスト**: スワップ前後で `contract_to_tensor()` が一致（外部インデックスの並び替えを考慮）
   - 無効/到達不能な配置でのエラーテスト
   - f64 と Complex64 の両方でテスト

### Phase 4: ドキュメントと仕上げ
9. 公開 API のドキュメントコメント
10. `mod.rs` への re-export 追加

## Issue の Questions への回答方針

> Should we support partial swaps (only move some indices, leave others)?

→ **Yes**. `target_assignment` に全インデックスを含める必要はなく、含まれていないインデックスは現在の位置を維持する、と定義する。これにより自然に partial swap がサポートされる。

> How to handle the case where target assignment is unreachable?

→ 木構造では任意の2ノード間にパスが存在するため、「到達不能」は発生しない。ただし以下はエラーとする:
- target のノード名がトポロジーに存在しない
- 同じインデックスが複数のノードに割り当てられている

> Should swap plan computation warn/error if total distance is large?

→ 最初はエラーにしない。必要に応じて warning を追加する。ユーザーは `SwapPlan` の内容を検査して移動距離を確認できる。

## 技術的な注意事項

1. **SiteIndexNetwork の更新方針**: `TreeTN::replace_tensor` は「接続(bond)以外の external indices を physical indices とみなす」ことで `SiteIndexNetwork::set_site_space()` を内部的に呼び、`site_spaces`/`index_to_node` を更新する。したがって swap 実装は **テンソルの外部インデックス集合を正しく作ること**に集中し、`SiteIndexNetwork` を手でいじらない方針にする。

2. **正準化の前提条件**: `apply_local_update_sweep` は正準化された TreeTN を要求する。`swap_site_indices` の冒頭で必要に応じて正準化を行う。

3. **ボンド次元（ランク）について**: サイトインデックス配置の変更は一般に bipartition を変えるため、**厳密に同じテンソルを保つにはランクが増大しうる**。`SwapOptions { max_rank, rtol }` を指定すると、ランクを抑制する代わりに **近似（誤差導入）**になることを明記する。デフォルト（None/None）は「必要ならランク増大を許す（厳密優先）」。

4. **`same_topology` チェック**: `replace_subtree` は topology が同じであることを要求する。スワップではノード間のエッジ構造は変わらないため、この条件は満たされる。ただし、サイトインデックスの割り当てが変わるため、`same_appearance` は満たされない可能性がある。
   - → `replace_subtree` は `same_topology` を要求する（`same_appearance` は要求しない）。テンソル置換時には `replace_tensor` が site space をテンソルから更新するため、「サイトインデックスの割当が変わる」こと自体は問題にならない。

5. **`SwapUpdater` の `update()` での left_inds 決定**: スワップ後に A に残るべきインデックスを正しく計算するために、A 側のサブツリーに目標ノードがあるかどうかの判定が必要。これは `path_between` または BFS で実装できる。
