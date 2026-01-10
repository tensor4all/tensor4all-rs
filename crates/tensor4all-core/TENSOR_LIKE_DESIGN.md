# TensorLike Trait 統合設計

## 現状の問題

### 2つのトレイトが存在
1. **TensorLike** - `Id, Symm, Tags` を関連型として持つ
2. **TensorDynLenLike<I>** - `I: IndexLike` でパラメタ化

### 問題点
- `TensorLike` が `Id, Symm` を露出している
- 2つのトレイトで責務が重複
- treetn側で `I::Id: From<DynId>` のような bounds が必要になる
- TreeTN に TensorLike を実装すると `tensordot` の意味が不明確
- `dyn_treetn.rs` の trait object 処理が複雑

## 提案: TensorLike は密テンソル専用

**重要な設計決定**: TreeTN は TensorLike を実装しない。

### 理由

1. **tensordot の意味が不明確**: TreeTN 同士の `tensordot` とは何か？
2. **コストの隠蔽**: `to_tensor()` が指数的コストを隠す
3. **責務の分離**: 密テンソルと TN は本質的に異なるデータ構造

### TensorLike トレイト（密テンソル専用）

```rust
/// 密テンソルの抽象化トレイト
///
/// **注意**: このトレイトは TensorDynLen 専用。TreeTN は実装しない。
pub trait TensorLike: Clone + Debug + Send + Sync {
    /// インデックス型（関連型）
    type Index: IndexLike;

    /// インデックスを返す
    fn indices(&self) -> Vec<Self::Index>;

    /// インデックスの数
    fn num_indices(&self) -> usize {
        self.indices().len()
    }

    /// 明示的な縮約
    fn tensordot(&self, other: &Self, pairs: &[(Self::Index, Self::Index)]) -> Result<Self>;

    /// 分解（SVD/QR/LU）
    fn factorize(&self, left_inds: &[Self::Index], options: &FactorizeOptions)
        -> Result<(Self, Self, Self::Index, Canonical)>;

    /// Downcasting 用
    fn as_any(&self) -> &dyn Any;
}
```

### TreeTN は独自の API を持つ

```rust
impl<T: TensorLike, V> TreeTN<T, V> {
    /// 物理インデックスを返す（ボンドは含まない）
    pub fn site_indices(&self) -> Vec<T::Index> { ... }

    /// 全縮約して密テンソルに変換
    /// **注意**: 指数的コスト
    pub fn contract_to_tensor(&self) -> Result<TensorDynLen> { ... }

    /// ノード間の縮約（グラフ操作）
    pub fn contract_nodes(&mut self, v1: V, v2: V) -> Result<V> { ... }
}
```

## 具体型の設計

### TensorDynLen（具体型、型パラメータなし）

```rust
/// 密テンソル（具体型）
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
}

impl TensorLike for TensorDynLen {
    type Index = DynIndex;

    fn indices(&self) -> Vec<DynIndex> { self.indices.clone() }
    fn tensordot(&self, other: &Self, pairs: &[(DynIndex, DynIndex)]) -> Result<Self> { ... }
    fn factorize(&self, left_inds: &[DynIndex], options: &FactorizeOptions)
        -> Result<(Self, Self, DynIndex, Canonical)> { ... }
    fn as_any(&self) -> &dyn Any { self }
}
```

### TreeTN（T: TensorLike に対してジェネリック、TensorLike は実装しない）

```rust
/// ツリーテンソルネットワーク
///
/// **注意**: TensorLike は実装しない。独自の API を持つ。
pub struct TreeTN<T, V = NodeIndex>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// グラフ: edge は T::Index::Id（軽量、conjugate 非依存）
    graph: NamedGraph<V, T, <T::Index as IndexLike>::Id>,
    canonical_center: HashSet<V>,
    canonical_form: Option<CanonicalForm>,
    site_index_network: SiteIndexNetwork<V, <T::Index as IndexLike>::Id>,
    ortho_towards: HashMap<<T::Index as IndexLike>::Id, V>,
}

// TensorLike は実装しない！
// 代わりに独自メソッドを提供
impl<T: TensorLike, V: ...> TreeTN<T, V> {
    /// 物理インデックス
    pub fn site_indices(&self) -> Vec<T::Index> { ... }

    /// 全縮約（指数的コスト）
    pub fn contract_to_tensor(&self) -> Result<TensorDynLen> { ... }
}

/// デフォルトの TreeTN 型
pub type DefaultTreeTN<V = NodeIndex> = TreeTN<TensorDynLen, V>;
```

### dyn_treetn.rs の削除

`dyn_treetn.rs` は削除する。理由:
- `Box<dyn TensorLike>` をノードとして持つ設計が複雑
- TreeTN が TensorLike を実装しないので `dyn TensorLike` の入れ子が不要
- 具体型 `TreeTN<TensorDynLen>` で十分

## 型パラメータの比較

| | 旧設計 | 新設計 |
|---|---|---|
| TensorDynLen | `<Id, Symm>` | 具体型（パラメータなし） |
| TreeTN | `<I, V>` where `I: IndexLike` | `<T, V>` where `T: TensorLike` |
| TensorLike | `<I: IndexLike>` 型パラメータ | `type Index` 関連型 |
| IndexLike | `type Id, Symm, Tags` | `type Id` のみ（軽量） |
| TreeTN edge | I（Index 全体） | `<T::Index as IndexLike>::Id`（軽量 ID） |

## IndexLike の設計

```rust
/// インデックスの抽象化トレイト
///
/// # 設計方針
///
/// - `Id` を関連型として持つ（軽量な識別子）
/// - ID が同じ Index は同じと定義（`Eq` は ID で比較）
/// - conjugate 状態（方向）は等価性判定では無視される
/// - conjugate の整合性は縮約実行時にチェックされ、不一致なら Runtime エラー
pub trait IndexLike: Clone + Debug + Send + Sync + 'static {
    /// 軽量な識別子型（conjugate 情報を持たない）
    type Id: Clone + Eq + Hash + Debug + Send + Sync;

    /// ID を取得
    fn id(&self) -> &Self::Id;

    /// 次元
    fn dim(&self) -> usize;

    /// 同じ ID か比較（デフォルト実装）
    fn same_id(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// 他の Index の ID と比較（デフォルト実装）
    fn has_id(&self, id: &Self::Id) -> bool {
        self.id() == id
    }
}

// PartialEq/Eq は ID で判定（conjugate 無視）
// 実装例:
// impl PartialEq for DynIndex {
//     fn eq(&self, other: &Self) -> bool {
//         self.id() == other.id()
//     }
// }
```

### ID を関連型にする理由

1. **軽量性**: ID は軽量な型として設計可能（例: `u64`）
2. **TreeTN の edge**: edge には conjugate 非依存な ID を使う
3. **比較の一貫性**: `Eq` は ID 比較と同義
4. **柔軟性**: ID の具体的な型は実装に任せる

### TreeTN での活用

```rust
pub struct TreeTN<T, V = NodeIndex>
where
    T: TensorLike,
{
    /// edge は ID（軽量、conjugate 非依存）
    /// 両側のテンソルを ID で結ぶ綺麗な設計
    graph: NamedGraph<V, T, <T::Index as IndexLike>::Id>,
    // ...
}
```

**メリット**:
- Edge は軽量な ID のみ保持
- Conjugate 情報は各ノードのテンソル側 Index に保持
- ID が両側のテンソルを結ぶ「接着剤」として機能

### Eq の設計方針

`Eq` は「同じ論理的な脚かどうか」を判定する（conjugate 状態は無視）。

具体的な実装での例（ITensor スタイル）:
```rust
// ITensor では内部的に ID で一致性を判定する
let i = Index::new(id, symm);
let i_conj = i.clone().conjugate();  // 方向が反転

// 同じ脚なので等価（ID が同じ）
assert!(i == i_conj);
assert!(i.id() == i_conj.id());

// 縮約時に conjugate の整合性をチェック
// 不一致なら Runtime エラー
tensor_a.contract(&tensor_b)?;  // conjugate 不一致ならエラー
```

**理由**:
- `Eq` と `can_contract` を分けるとコードが複雑になる
- common_inds などの操作で自然に動作する
- 実際の縮約時のみ conjugate 整合性をチェックすれば十分

**重要**:
- `Symm`, `Tags` は `IndexLike` から露出しない（実装の詳細）
- `new_bond()` は `IndexLike` に含めない（具体型 `DynIndex` のメソッドとして提供）

## ボンドインデックスの生成

`new_bond()` は `IndexLike` トレイトではなく、具体型のメソッド:

```rust
impl DynIndex {
    /// 新しいボンドインデックス生成（DynIndex専用）
    pub fn new_bond(dim: usize) -> Result<Self> { ... }
}
```

**理由**:
- `TensorDynLen` は `DynIndex::new_bond()` を直接使用
- `TreeTN` は既存のボンドを使うため `new_bond` 不要
- `IndexLike` を最小限に保つ

## メリット

1. **型パラメータの簡素化**: TensorDynLen が具体型に
2. **自然な設計**: テンソル型がインデックス型を知っている（T::Index）
3. **bounds の削減**: `From<DynId>` が不要に
4. **トレイトの統合**: 2つ → 1つ
5. **責務の明確化**: TensorLike は密テンソル専用、TreeTN は独自 API
6. **軽量な edge**: TreeTN の edge は ID のみ（conjugate 非依存）
7. **綺麗な接続モデル**: ID が両側のテンソルを結ぶ「接着剤」
8. **複雑さの削減**: dyn_treetn.rs を削除

## 移行手順

### Phase 1: core の IndexLike/TensorLike リファクタリング

1. [x] IndexLike を `type Id` のみ持つ設計に変更（Symm, Tags は削除）
2. [ ] IndexLike に `id()`, `same_id()`, `has_id()` メソッド追加
3. [ ] TensorDynLen を具体型に変更（型パラメータ削除）
4. [ ] TensorLike を関連型 `type Index` を使う設計に変更
5. [ ] TensorDynLenLike を削除（TensorLike に統合）
6. [ ] factorize, svd, qr を DynIndex 専用に変更

### Phase 2: treetn の更新

7. [ ] TreeTN から `impl TensorLike` を削除
8. [ ] `dyn_treetn.rs` を削除
9. [ ] `tensor_like.rs` を削除（または最小化）
10. [ ] TreeTN を `<T: TensorLike, V>` に変更
11. [ ] TreeTN の edge を `<T::Index as IndexLike>::Id` に変更
12. [ ] treetn の `From<DynId>` bounds を削除
13. [ ] テストを `contract_to_tensor()` を使うように修正

## ファイル構成

```
tensor4all-core/src/
├── traits.rs                      # IndexLike + TensorLike トレイト定義
├── index.rs                       # Index<Id, Symm, Tags> 構造体 + DynIndex
├── tensordynlen.rs                # TensorDynLen 構造体 + impl TensorLike
└── ...

tensor4all-treetn/src/
├── treetn/
│   ├── mod.rs
│   ├── contraction.rs             # contract_to_tensor() はここ
│   ├── tensor_like.rs             # 削除！
│   └── ...
├── dyn_treetn.rs                  # 削除！
└── ...
```

**traits.rs の内容:**
```rust
//! Core abstractions for tensor and index types.

pub trait IndexLike: Clone + Debug + Send + Sync + 'static {
    /// 軽量な識別子型（conjugate 情報を持たない）
    type Id: Clone + Eq + Hash + Debug + Send + Sync;

    fn id(&self) -> &Self::Id;
    fn dim(&self) -> usize;

    fn same_id(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    fn has_id(&self, id: &Self::Id) -> bool {
        self.id() == id
    }
}

/// 密テンソルの抽象化トレイト（TensorDynLen 専用）
pub trait TensorLike: Clone + Debug + Send + Sync {
    type Index: IndexLike;

    fn indices(&self) -> Vec<Self::Index>;
    fn num_indices(&self) -> usize { self.indices().len() }
    fn tensordot(&self, other: &Self, pairs: &[(Self::Index, Self::Index)]) -> Result<Self>;
    fn factorize(&self, left_inds: &[Self::Index], options: &FactorizeOptions)
        -> Result<(Self, Self, Self::Index, Canonical)>;
    fn as_any(&self) -> &dyn Any;
}

pub type DynIndex = Index<DynId, NoSymmSpace, TagSet>;
```

## 未解決の問題

1. **object safety**: `TensorLike` は trait object として使えるか？
   - 関連型があるので `dyn TensorLike<Index = DynIndex>` のように具体化が必要
   - ただし TreeTN が TensorLike を実装しないので、trait object の必要性は低下
