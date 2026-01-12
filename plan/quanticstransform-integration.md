# QuanticsTransform Integration Plan

## 背景

tensor4all-quanticstransform は Quantics.jl の Rust 実装であるが、以下が未完成:

1. **変換の数値的正確性テスト**: Fourier, flip, shift 等の変換が正しく動作することを検証するテストがない
2. **TreeTN への適用機能**: `LinearOperator` を実際の `TreeTN` に適用する機能
3. **タグベースのインデックス選択**: Quantics.jl のようなタグベースのワークフロー

## 完了した作業

### Phase 1 & 2: 数値テスト完了 (2025-01-12)

**Big-endian convention への統一** (Julia Quantics.jl と同じ):
- Site 0 = MSB (Most Significant Bit)
- Site R-1 = LSB (Least Significant Bit)
- x = Σ_n x_n * 2^(R-1-n)

以下の変換の数値的正確性を検証:

1. **Flip operator** (Periodic/Open BC)
   - flip(x) = 2^R - x を全 x ∈ [0, 2^R) で検証
   - Open BC: flip(0) = 2^R は overflow → zero vector (Rust拡張機能)
   - Big-endian convention

2. **Shift operator** (Periodic/Open BC)
   - shift(x, offset) = x + offset を全 x と複数の offset で検証
   - Open BC: overflow/underflow は zero vector
   - Big-endian convention

3. **Fourier operator**
   - Unitarity: ||F|x⟩||² = 1 for all basis states
   - Inverse operator creation verified

4. **Phase rotation operator**
   - exp(i*θ*x) multiplication verified for all x
   - Identity tests: θ=0, θ=2π
   - Big-endian convention

5. **Cumsum operator**
   - Strict upper triangular matrix
   - Big-endian bit comparison (MSB first)
   - Full numerical verification for all x

6. **Affine operator**
   - Identity, shift, negation, 2D rotation の operator creation 検証

7. **Binaryop operator**
   - Identity, sum, difference の operator creation 検証

全 24 integration tests + 55 unit tests passing.

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

## ステータス

- ✅ Phase 1: 数値テスト (完了)
- ✅ Phase 2: Integration tests (完了)
- ⏳ Phase 3: タグシステム (optional, 未着手)
