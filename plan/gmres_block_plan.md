# ブロック行列GMRES実装プラン

## 概要

ブロック行列 `x`（行列の行列、または行列のベクトル）について線型方程式 `Ax = b` を解くソルバーを実装する。

## 背景と動機

既存のGMRES実装（`tensor4all-core/src/krylov.rs`）は`TensorLike`トレイトを実装する任意のテンソルに対して動作する。ブロック行列を`TensorLike`として扱えば、既存のGMRESをそのまま活用できる。

## 設計方針

### 方針決定（A案）: 既存GMRESを変えない

- 既存の `gmres` は型境界として `T: TensorLike` を要求するため、`BlockTensor<T>` も `TensorLike` を実装する。
- ただし `BlockTensor` は「ブロック構造のベクトル空間」としての利用を主目的とし、GMRESが呼ばない `TensorLike` の必須メソッド（`contract` 等）は **panic (`unimplemented!()`) ではなく `Result` で明示的に未対応エラーを返す**。
- `conj` と `TensorIndex` 系（`replaceind/replaceinds`）はブロックごとに委譲して実装し、将来的に各ブロックが MPS/MPO の場合も同じコードパスで動くようにする。

### アプローチ: BlockTensor型の導入

```rust
/// ブロック構造を持つテンソルのコレクション
/// 各ブロックは同じ型Tを持ち、TensorLikeを実装する
pub struct BlockTensor<T: TensorLike> {
    /// ブロックの1次元配列（2次元ブロック行列は行優先で平坦化）
    blocks: Vec<T>,
    /// ブロック構造 (rows, cols) - 例: 2x1なら (2, 1)
    shape: (usize, usize),
}
```

### TensorLikeトレイトの実装

GMRESに必要なベクトル空間操作：

| メソッド | BlockTensor実装 |
|---------|----------------|
| `norm_squared()` | 全ブロックの`norm_squared()`の和 |
| `scale(s)` | 各ブロックを`s`倍 |
| `axpby(a, other, b)` | 各ブロックで`a*self[i] + b*other[i]` |
| `inner_product(other)` | 全ブロック対の内積の和 |

**注意**: `TensorLike` はベクトル空間操作以外に `factorize/contract/outer_product/permuteinds/diagonal/scalar_one/ones/onehot` などの必須メソッドを含む。
本プランでは GMRES に不要なものは「未対応」とし、`anyhow::bail!("BlockTensor does not support ...")` 等で `Result` を返す（panicしない）。

### BlockLinearOperator

```rust
/// ブロック行列に対する線形演算子
/// Ax を計算する関数として定義
type BlockLinearOperator<T> = Box<dyn Fn(&BlockTensor<T>) -> Result<BlockTensor<T>>>;
```

ユーザーは`BlockLinearOperator`を定義し、既存の`gmres`関数に渡す。

## 実装計画

### Phase 1: BlockTensor型の実装

**ファイル**: `crates/tensor4all-core/src/block_tensor.rs`

1. `BlockTensor<T>` 構造体の定義
2. 基本操作の実装
    - `try_new(blocks: Vec<T>, shape: (usize, usize)) -> Result<Self>` - コンストラクタ（`rows * cols == blocks.len()` を検証）
    - `new(blocks: Vec<T>, shape: (usize, usize)) -> Self` - 便利関数（内部で `try_new(...).expect(...)` しても良いが、ライブラリ用途では `try_new` を推奨）
   - `get(row: usize, col: usize)` - ブロックの取得
   - `get_mut(row: usize, col: usize)` - ブロックの可変参照
   - `shape()` - ブロック構造の取得
   - `num_blocks()` - ブロック数の取得

### Phase 2: TensorLikeトレイトの実装

`BlockTensor<T>`に対して`TensorLike`トレイトを実装：

```rust
impl<T: TensorLike> TensorLike for BlockTensor<T> {
    // ベクトル空間操作（GMRES必須）
    fn norm_squared(&self) -> f64 {
        self.blocks.iter().map(|b| b.norm_squared()).sum()
    }

    fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        let scaled: Result<Vec<T>> = self.blocks.iter()
            .map(|b| b.scale(scalar.clone()))
            .collect();
        Ok(Self { blocks: scaled?, shape: self.shape })
    }

    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        // ライブラリコードなのでpanicではなくエラーで返す
        anyhow::ensure!(self.shape == other.shape, "Block shapes must match");
        let result: Result<Vec<T>> = self.blocks.iter()
            .zip(other.blocks.iter())
            .map(|(s, o)| s.axpby(a.clone(), o, b.clone()))
            .collect();
        Ok(Self { blocks: result?, shape: self.shape })
    }

    fn inner_product(&self, other: &Self) -> Result<AnyScalar> {
        anyhow::ensure!(self.shape == other.shape, "Block shapes must match");
        let mut sum = AnyScalar::F64(0.0);
        for (s, o) in self.blocks.iter().zip(other.blocks.iter()) {
            sum = sum + s.inner_product(o)?;
        }
        Ok(sum)
    }

    // 他の必須メソッド（GMRESには不要）
    // - `conj` はブロックごとに委譲して実装
    // - それ以外は当面「未対応」としてResultでエラーを返す（panicしない）
}
```

### Phase 3: TensorIndexトレイトの実装

`TensorLike`のスーパートレイト`TensorIndex`も実装が必要：

```rust
impl<T: TensorLike> TensorIndex for BlockTensor<T> {
    type Index = T::Index;

    fn external_indices(&self) -> Vec<Self::Index> {
        // ブロックテンソルとしては外部インデックスは持たない
        // （各ブロックが独自のインデックスを持つ）
        vec![]
    }

    fn replaceind(&self, old: &Self::Index, new: &Self::Index) -> Result<Self> {
        let replaced: Result<Vec<T>> = self.blocks.iter()
            .map(|b| b.replaceind(old, new))
            .collect();
        Ok(Self { blocks: replaced?, shape: self.shape })
    }

    fn replaceinds(&self, old_indices: &[Self::Index], new_indices: &[Self::Index]) -> Result<Self> {
        let replaced: Result<Vec<T>> = self.blocks.iter()
            .map(|b| b.replaceinds(old_indices, new_indices))
            .collect();
        Ok(Self { blocks: replaced?, shape: self.shape })
    }
}
```

### Phase 4: ユニットテスト

**ファイル**: `crates/tensor4all-core/src/block_tensor.rs` 内の `#[cfg(test)]` モジュール

#### テスト0（推奨）: `BlockTensor` の不変条件

- `rows * cols != blocks.len()` のとき `try_new` がエラーになること
- `axpby/inner_product` が shape 不一致でエラーになること

#### テスト1: 恒等行列によるGMRES（Dense行列ブロック）

初期段階では、各ブロックを通常のDense配列（`TensorDynLen::from_dense_f64` 等）で作る。
将来的にブロックがMPS/MPOになっても、`BlockTensor<T>` 側の実装は同じでよい（各ブロック型が `TensorLike` のベクトル空間操作を提供していればGMRESは動く）。

```
x = [x1, x2]^T (2x1ブロック、各ブロック2x1)
b = [b1, b2]^T (2x1ブロック)

A = I (4x4恒等行列、2x2ブロック構造)
A = [[I2, 0], [0, I2]]

Ax = x なので、b = x のとき解は x = b
```

#### テスト2: 対角ブロック行列

```
A = [[D1, 0], [0, D2]] (対角ブロック)
D1 = diag(2, 3), D2 = diag(4, 5)

b = [b1, b2]^T = [[2, 3]^T, [4, 5]^T]
x = [[1, 1]^T, [1, 1]^T]
```

#### テスト3: 非対角ブロック行列

```
A = [[I, B], [0, I]] (上三角ブロック)
B = [[1, 0], [0, 1]] (単位行列)

x = [x1, x2]^T
Ax = [x1 + Bx2, x2]^T

b = [[2, 3]^T, [1, 1]^T] のとき
x2 = [1, 1]^T, x1 = [2, 3]^T - B*[1, 1]^T = [1, 2]^T
```

## API使用例

```rust
use tensor4all_core::block_tensor::BlockTensor;
use tensor4all_core::krylov::{gmres, GmresOptions};

// 2x1ブロック構造のxとb
let b = BlockTensor::new(vec![b1, b2], (2, 1));
let x0 = BlockTensor::new(vec![zero1, zero2], (2, 1));

// ブロック行列Aの作用を定義
let apply_a = |x: &BlockTensor<TensorDynLen>| -> Result<BlockTensor<TensorDynLen>> {
    // A = [[A11, A12], [A21, A22]] の作用を計算
    let x1 = x.get(0, 0);
    let x2 = x.get(1, 0);

    let y1 = a11.apply(x1)?.axpby(1.0, &a12.apply(x2)?, 1.0)?;
    let y2 = a21.apply(x1)?.axpby(1.0, &a22.apply(x2)?, 1.0)?;

    Ok(BlockTensor::new(vec![y1, y2], (2, 1)))
};

let options = GmresOptions::default();
let result = gmres(apply_a, &b, &x0, &options)?;

// result.solution は BlockTensor<TensorDynLen>
let x1_solution = result.solution.get(0, 0);
let x2_solution = result.solution.get(1, 0);
```

## ファイル構成

```
crates/tensor4all-core/src/
├── lib.rs                 # mod block_tensor; を追加
├── block_tensor.rs        # 新規: BlockTensor型の実装
└── krylov.rs              # 変更なし（既存のGMRESをそのまま使用）
```

## 実装の優先順位

1. **必須**: `BlockTensor`構造体とコンストラクタ
2. **必須**: ベクトル空間操作（`norm_squared`, `scale`, `axpby`, `inner_product`）
3. **必須**: `TensorIndex`トレイトの基本実装
4. **必須**: ユニットテスト
5. **任意**: 他の`TensorLike`メソッド（`factorize`, `contract`等）

## 注意点

- `BlockTensor`の各ブロックは同じ型`T`である必要がある
- `Clone`トレイトが必要（GMRESの内部でベクトルを複製するため）
- 2次元ブロック行列（例: 2x2ブロック）も行優先で1次元配列として格納
- ブロック同士のインデックスの整合性はユーザー責任（`BlockLinearOperator`の定義時）

## 参考ファイル

- `/Users/ken/git_for_collab4/tensor4all-rs/crates/tensor4all-core/src/krylov.rs` - 既存GMRES実装
- `/Users/ken/git_for_collab4/tensor4all-rs/crates/tensor4all-core/src/tensor_like.rs` - TensorLikeトレイト定義
