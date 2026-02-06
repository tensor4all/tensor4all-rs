# ブロック行列MPSのGMRESテスト実装プラン

## 概要

`BlockTensor<TensorTrain>`（各ブロックがMPS）を使用したGMRESのテストを実装する。

## 背景

- `BlockTensor<T: TensorLike>`は実装済み（`tensor4all-core/src/block_tensor.rs`）
- `TensorTrain`（MPS/MPO）は`TensorLike`を実装済み
- よって`BlockTensor<TensorTrain>`はそのまま既存の`gmres_with_truncation()`/`restart_gmres_with_truncation()`で使用可能

## テストファイル

**ファイル**: `crates/tensor4all-itensorlike/examples/test_gmres_block_mps.rs`

## テストケース

### テスト1: ブロック対角恒等演算子

**概要**: 最も単純なケース。N=2のMPSを2つ持つ2x1ブロックベクトルに対し、ブロック対角恒等演算子でGMRESを解く。

```
x = [x1, x2]^T  (2x1 BlockTensor<TensorTrain>)
    x1: N=2サイト、全要素1のMPS
    x2: N=2サイト、全要素1のMPS

b = x  (恒等演算なので同じ)

A = [[I, 0], [0, I]]  (ブロック対角恒等演算子)
    I: N=2の恒等MPO
    0: ゼロ演算（出力がゼロMPS）

Ax = [I*x1, I*x2]^T = [x1, x2]^T = b
```

**期待結果**: 収束し、真の残差 `||b - A*x||/||b||` が `rtol` 未満になる（反復回数は「少数回」を期待するが固定しない）

### テスト2: ブロック対角演算子（非恒等）

**概要**: 各ブロックに非自明なMPOを適用する演算子。

注意: `x_true = ones` だと、Pauli-X（各サイト反転）は「全要素が等しい」状態に対して不変になりやすく、恒等演算子テストと実質的に同じになってしまう可能性がある。
このテストでは、少なくとも以下のいずれかで「不変性」を避ける：

- `x_true` を onehot（例: |00..0>）にして、Pauli-Xで |11..1> に写ることを利用する
- あるいは `x_true` を乱数MPS（seed固定）にして、`b = A*x_true` が確実に変化するようにする
- あるいは Pauli-X ではなく、既存例にある対角MPO `diag(2,3)` を使う（最小実装が簡単）

```
A = [[X, 0], [0, X]]  (ブロック対角Pauli-X)
    X: Pauli-X MPO (|0>→|1>, |1>→|0>)

x_true = [ones, ones]^T
b = A * x_true = [X*ones, X*ones]^T

x0 = 0.5 * b  (初期推定)
```

**期待結果**: GMRESが収束し、residualが十分小さくなる

**推奨チェック**:

- `BlockTensor` 解 `x_sol` について、各ブロックで `||A*x_sol - b||/||b|| < rtol` を確認
- 可能なら `||x_sol - x_true||` も併せて確認（ただしMPSの表現ゆらぎがあるので閾値は緩めに）

### テスト3: ブロック上三角演算子

**概要**: 非対角ブロックを持つ演算子。`y1 = I*x1 + B*x2` の「加算」が成立するように、`B` は **ブロック2のインデックス空間 → ブロック1のインデックス空間** への写像として定義する。

（例）ブロック間でサイトインデックスを独立に生成する場合：

- `x1` は `sites[0][*]` を外部インデックスに持つ
- `x2` は `sites[1][*]` を外部インデックスに持つ
- `B` は入力に `sites[1][*]`、出力に `sites[0][*]`（または `mpo_outputs[0][*]` を経由して `replaceinds` で `sites[0][*]` に戻す）

こうすると `B*x2` と `x1` が同じ外部インデックスを持ち、`axpby` による加算が可能になる。

```
A = [[I, B], [0, I]]  (上三角ブロック)
    B: 恒等MPO（またはスケール係数付き）

x = [x1, x2]^T
Ax = [I*x1 + B*x2, I*x2]^T = [x1 + x2, x2]^T

b = [[3, 3, 3, 3]^T as MPS, [1, 1, 1, 1]^T as MPS]
x2 = ones, x1 = b1 - x2 = [2, 2, 2, 2]^T as MPS
```

**期待結果**: GMRESが上三角構造を正しく解く

### テスト4: Restart GMRESとの組み合わせ

**概要**: `restart_gmres_with_truncation`をBlockTensor<TensorTrain>で使用。

```
A = ブロック対角 [[D, 0], [0, D]]
    D: 対角MPO diag(2, 3)

truncation付きでrestart GMRESを実行
```

**期待結果**: truncationありでも収束

## 実装詳細

### ファイル構成

```rust
//! Test: GMRES solver with BlockTensor<TensorTrain> (block MPS)
//!
//! Tests GMRES with block vectors where each block is an MPS.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_block_mps --release

use tensor4all_core::block_tensor::BlockTensor;
use tensor4all_core::krylov::{gmres_with_truncation, restart_gmres_with_truncation, GmresOptions, RestartGmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};
```

### 共有インデックス構造

```rust
/// 各ブロックMPS用の共有インデックス
/// 注意:
/// - ブロック対角（ブロック同士が独立）では、ブロック間でインデックスが独立でも良い
/// - 非対角ブロック（例: 上三角の B）を入れる場合、B は「ブロック間のインデックス写像」になっている必要がある
struct BlockSharedIndices {
    /// ブロック数
    num_blocks: usize,
    /// 各ブロックのサイトインデックス
    sites: Vec<Vec<DynIndex>>,
    /// 各ブロックのボンドインデックス
    bonds: Vec<Vec<DynIndex>>,
    /// 各ブロックのMPO出力インデックス
    mpo_outputs: Vec<Vec<DynIndex>>,
}

impl BlockSharedIndices {
    fn new(num_blocks: usize, n: usize, phys_dim: usize) -> Self {
        // 各ブロックに対して独立したインデックスを生成
        // ...
    }
}
```

### ヘルパー関数

```rust
/// ブロックごとにonesのMPSを作成
fn create_block_ones_mps(indices: &BlockSharedIndices) -> anyhow::Result<BlockTensor<TensorTrain>>

/// ブロックごとにゼロのMPSを作成
fn create_block_zero_mps(indices: &BlockSharedIndices) -> anyhow::Result<BlockTensor<TensorTrain>>

/// ブロック対角恒等演算子を適用
/// A = [[I, 0], [0, I]]
fn apply_block_diagonal_identity(
    x: &BlockTensor<TensorTrain>,
    indices: &BlockSharedIndices,
) -> anyhow::Result<BlockTensor<TensorTrain>>

/// ブロック対角MPO演算子を適用
/// A = [[mpo, 0], [0, mpo]]
fn apply_block_diagonal_mpo(
    x: &BlockTensor<TensorTrain>,
    mpo: &TensorTrain,  // 各ブロックに適用するMPO
    indices: &BlockSharedIndices,
) -> anyhow::Result<BlockTensor<TensorTrain>>

/// ブロック上三角演算子を適用
/// A = [[I, B], [0, I]]
fn apply_block_upper_triangular(
    x: &BlockTensor<TensorTrain>,
    b_operator: &TensorTrain,  // (0,1)ブロックのMPO
    indices: &BlockSharedIndices,
) -> anyhow::Result<BlockTensor<TensorTrain>>

/// `restart_gmres_with_truncation` 用の truncate 関数
/// （ブロックごとに `TensorTrain::truncate` を適用する）
fn truncate_block(x: &mut BlockTensor<TensorTrain>, opts: &TruncateOptions) -> anyhow::Result<()>
```

### main関数の構成

```rust
fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("  Block MPS GMRES Tests");
    println!("========================================\n");

    // Test 1: Block diagonal identity
    test_block_diagonal_identity()?;

    // Test 2: Block diagonal Pauli-X
    test_block_diagonal_pauli_x()?;

    // Test 3: Block upper triangular
    test_block_upper_triangular()?;

    // Test 4: Restart GMRES with block MPS
    test_restart_gmres_block_mps()?;

    println!("\n========================================");
    println!("  All tests completed!");
    println!("========================================");

    Ok(())
}
```

## 参考ファイル

- `crates/tensor4all-itensorlike/examples/test_gmres_mps.rs` - 既存のMPS GMRESテスト
- `crates/tensor4all-itensorlike/examples/test_restart_gmres_mps.rs` - 既存のrestart GMRES MPSテスト
- `crates/tensor4all-core/src/block_tensor.rs` - BlockTensor実装

## 実装の優先順位

1. **必須**: テスト1（ブロック対角恒等演算子）- 最小動作確認
2. **必須**: テスト2（ブロック対角Pauli-X）- 非自明な演算子のテスト
3. **推奨**: テスト3（ブロック上三角演算子）- 非対角ブロックのテスト
4. **推奨**: テスト4（Restart GMRES）- truncation付きの動作確認

## 注意点

1. **インデックス設計**:
    - ブロック対角だけなら「ブロック間で独立」でも「共有」でも成立しうるが、混乱を避けるため独立を推奨。
    - ただし非対角ブロックを入れるなら、演算子側（MPO）で入力/出力インデックスを設計して「加算できる空間」に揃える。

2. **MPO適用時のインデックス置換**: `apply_mpo`後に `replaceinds(old, new)` で MPO 出力インデックスをサイトインデックスに戻す（既存例に合わせる）。

3. **truncation**: `TensorTrain`の`truncate`メソッドは各ブロックに個別に適用される。`BlockTensor`のtruncation用ヘルパーが必要かもしれない。

4. **エラーハンドリング**: `BlockTensor`のメソッドは`Result`を返すので、`?`演算子で適切にエラー伝播する。

## 期待される出力例

```
========================================
  Block MPS GMRES Tests
========================================

=== Test 1: Block Diagonal Identity ===
Block structure: 2x1, each block: N=2 sites
Initial residual: 1.00e+00
Converged: true
Iterations: 1
Final residual: 1.23e-15

=== Test 2: Block Diagonal Pauli-X ===
Block structure: 2x1, each block: N=2 sites
Initial residual: 5.00e-01
Converged: true
Iterations: 2
Final residual: 8.45e-09

...

========================================
  All tests completed!
========================================
```
