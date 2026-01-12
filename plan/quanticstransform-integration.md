# QuanticsTransform Integration Plan

## 背景

tensor4all-quanticstransform は Quantics.jl の Rust 実装であるが、以下が未完成:

1. **変換の数値的正確性テスト**: Fourier, flip, shift 等の変換が正しく動作することを検証するテストがない
2. **TreeTN への適用機能**: `LinearOperator` を実際の `TreeTN` に適用する機能
3. **タグベースのインデックス選択**: Quantics.jl のようなタグベースのワークフロー

## 完了した作業 (2025-01-12)

### TensorIndex トレイト (tensor4all-core)

```rust
pub trait TensorIndex: Sized + Clone + Debug + Send + Sync {
    type Index: IndexLike;
    fn external_indices(&self) -> Vec<Self::Index>;
    fn replaceind(&self, old: &Self::Index, new: &Self::Index) -> Result<Self>;
    fn replaceinds(&self, old: &[Self::Index], new: &[Self::Index]) -> Result<Self>;
}
```

- TensorLike から分離し、TreeTN が実装可能に
- TensorLike は TensorIndex を supertrait として継承

### LinkIndexNetwork (tensor4all-treetn)

- ボンドインデックス → エッジの O(1) 逆引き
- `insert()`, `find_edge()`, `replace_index()` メソッド

### SiteIndexNetwork 拡張

- サイトインデックス → ノードの O(1) 逆引き追加
- `find_node_by_index()`, `replace_site_index()` メソッド

### TreeTN の TensorIndex 実装

- `external_indices()`: 全サイトインデックスを返す
- `replaceind()`: サイト/リンクインデックスを自動判別して置換

### apply_linear_operator

```rust
pub fn apply_linear_operator<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
    options: ApplyOptions,
) -> Result<TreeTN<T, V>>
```

- 部分オペレータ対応 (compose_exclusive_linear_operators でギャップを埋める)
- 複数の縮約アルゴリズム (ZipUp, Fit, Naive)
- ApplyOptions でトランケーションパラメータ制御

## 次のステップ

### Phase 1: quanticstransform の数値テスト

**目標**: 各変換が数学的に正しいことを検証

#### 1.1 Fourier 変換テスト

```rust
#[test]
fn test_fourier_transform_correctness() {
    // sin(2πx) の Fourier 変換が δ(k-1) + δ(k+1) になることを確認
    // 1. 入力関数を TreeTN (MPS) として構築
    // 2. Fourier operator を適用
    // 3. 結果を contract_to_tensor して検証
}
```

#### 1.2 Flip 変換テスト

```rust
#[test]
fn test_flip_transform_correctness() {
    // f(x) → f(1-x) の検証
    // 多項式関数で確認
}
```

#### 1.3 Shift 変換テスト

```rust
#[test]
fn test_shift_transform_correctness() {
    // f(x) → f(x + a) の検証
    // 周期境界条件の確認
}
```

### Phase 2: Integration tests with TreeTN

**目標**: apply_linear_operator が正しく動作することを end-to-end で検証

```rust
#[test]
fn test_apply_fourier_to_mps() {
    let mps = create_test_mps();  // sin(2πx) を表現
    let fourier_op = build_fourier_operator(n_bits, grid);
    let result = apply_linear_operator(&fourier_op, &mps, ApplyOptions::default())?;
    // 結果を検証
}
```

### Phase 3: タグシステム (optional)

**現状**: IndexMapping による明示的なマッピングで対応

**検討事項**:
- Quantics.jl のタグシステム ("x", "k" 等) は便利だが、Rust では型安全性とのトレードオフ
- 現在の IndexMapping アプローチで十分な場合が多い
- 必要に応じて、タグ付きインデックスラッパーを追加可能

```rust
// 将来の拡張案
pub struct TaggedIndex<I: IndexLike> {
    inner: I,
    tags: HashSet<String>,
}
```

## ファイル構成

```
crates/tensor4all-quanticstransform/
├── src/
│   ├── lib.rs
│   ├── fourier.rs      # Fourier transform operator
│   ├── flip.rs         # Flip operator
│   ├── shift.rs        # Shift operator
│   └── grid.rs         # Quantics grid utilities
└── tests/
    ├── fourier_test.rs      # [追加] 数値テスト
    ├── flip_test.rs         # [追加] 数値テスト
    ├── shift_test.rs        # [追加] 数値テスト
    └── integration_test.rs  # [追加] TreeTN 適用テスト
```

## 依存関係

```
tensor4all-quanticstransform
├── tensor4all-core (TensorLike, TensorIndex, IndexLike)
└── tensor4all-treetn (TreeTN, LinearOperator, apply_linear_operator)
```

## 優先度

1. **High**: Phase 1 (数値テスト) - 変換の正確性を保証
2. **Medium**: Phase 2 (Integration) - 実用的なワークフロー検証
3. **Low**: Phase 3 (タグ) - 便利だが必須ではない
