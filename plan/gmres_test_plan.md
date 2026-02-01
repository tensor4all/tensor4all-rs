# GMRES with MPO/MPS Test Implementation Plan

## 目的

`krylov.rs` で実装された GMRES を用いて、A が MPO、x, b が MPS 形式のままで線型方程式 `Ax = b` を解けるようにする。

## 現状の分析

### tensor4all-itensorlike クレート

`TensorTrain` 型（MPS/MPO のラッパー）が既に実装されており、GMRES に必要なメソッドの多くが揃っている。

**既存の実装：**

| メソッド | 実装状況 | ファイル/行 |
|---------|---------|------------|
| `inner(&self, other: &Self) -> AnyScalar` | ✅ 実装済み | [tensortrain.rs:581-615](crates/tensor4all-itensorlike/src/tensortrain.rs#L581-L615) |
| `norm_squared(&self) -> f64` | ✅ 実装済み | [tensortrain.rs:620-622](crates/tensor4all-itensorlike/src/tensortrain.rs#L620-L622) |
| `norm(&self) -> f64` | ✅ 実装済み | [tensortrain.rs:627-629](crates/tensor4all-itensorlike/src/tensortrain.rs#L627-L629) |
| `add(&self, other: &Self) -> Result<Self>` | ✅ 実装済み | [tensortrain.rs:683-717](crates/tensor4all-itensorlike/src/tensortrain.rs#L683-L717) |
| `contract(&self, other: &Self, options)` | ✅ 実装済み | [contract.rs](crates/tensor4all-itensorlike/src/contract.rs) |
| `scale(&self, scalar) -> Result<Self>` | ❌ 未実装 | - |
| `axpby(&self, a, other, b) -> Result<Self>` | ❌ 未実装 | - |

### 重要: Index 設計（GMRES が前提とする「同一ベクトル空間」）

GMRES (`tensor4all-core/src/krylov.rs`) は反復のたびに `axpby` / `inner_product` を呼ぶため、
`apply_a(x)` が返す `T` は常に **`x` と同じ external indices（ID と ConjState を含む）** を持つ必要がある。

MPO×MPS の素朴な実装では、MPO の「入力物理 index」と「出力物理 index」が別物（別 ID）になりがちで、
そのままだと `apply_a` が別空間へ写像してしまい GMRES が破綻する。

このプランでは **Plan (B)** として、物理 index を **directed index (Ket/Bra)** にして
「入力/出力を同じ ID で持ち、ConjState だけを反転させる」ことで、MPO 適用後もベクトル空間が閉じるようにする。

補足: 現状の `DynIndex` は `ConjState::Undirected` 固定で `Ket/Bra` を表現できないため、Plan (B) を採用するには
`tensor4all-core` に directed index 型を追加し、`TensorTrain` がそれを使えるようにする必要がある。

### krylov.rs の GMRES

`gmres<T, F>` は以下を要求：
- `T: TensorLike`
- `F: Fn(&T) -> Result<T>` （線形演算子の適用）

GMRES で使用される `TensorLike` のメソッド：
- `axpby(a, other, b)` - 線形結合: `a * self + b * other`
- `scale(scalar)` - スカラー倍
- `inner_product(other)` - 内積
- `norm()` - ノルム
- `clone()` - クローン

## 実装アプローチ

**既存の `TensorTrain` 型を拡張して `TensorLike` を実装する**

利点：
1. `inner`, `norm`, `add` は既に実装済み
2. 内部で `TreeTN` を使用しており、効率的な実装
3. ITensorMPS.jl 互換の API

## 実装詳細

### Phase 1: TensorTrain に不足メソッドを追加

#### Step 1.1: `scale` メソッドを追加

```rust
// crates/tensor4all-itensorlike/src/tensortrain.rs に追加

impl TensorTrain {
    /// Scale the tensor train by a scalar.
    ///
    /// Only one tensor (the first) is scaled to avoid scalar^n scaling.
    pub fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut tensors = Vec::with_capacity(self.len());
        for site in 0..self.len() {
            let tensor = self.tensor(site);
            if site == 0 {
                tensors.push(tensor.scale(scalar.clone())?);
            } else {
                tensors.push(tensor.clone());
            }
        }

        Self::new(tensors)
    }
}
```

#### Step 1.2: `axpby` メソッドを追加

```rust
impl TensorTrain {
    /// Compute a * self + b * other.
    pub fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        let scaled_self = self.scale(a)?;
        let scaled_other = other.scale(b)?;
        scaled_self.add(&scaled_other)
    }
}
```

### Phase 2: TensorTrain に TensorLike を実装

```rust
// crates/tensor4all-itensorlike/src/tensortrain.rs に追加

use tensor4all_core::{TensorLike, TensorIndex, AllowedPairs, FactorizeOptions, FactorizeResult, FactorizeError, DirectSumResult};

impl TensorIndex for TensorTrain {
    type Index = DynIndex;

    fn external_indices(&self) -> Vec<Self::Index> {
        // 重複実装を避けるため、内部 TreeTN の TensorIndex 実装に委譲する
        // (TreeTN は TensorIndex を実装済み)
        self.as_treetn().external_indices()
    }

    fn num_external_indices(&self) -> usize {
        // 重複実装を避けるため、内部 TreeTN の TensorIndex 実装に委譲する
        self.as_treetn().num_external_indices()
    }

    fn replaceind(&self, old: &Self::Index, new: &Self::Index) -> anyhow::Result<Self> {
        // 重複実装を避けるため、内部 TreeTN の replaceind に委譲する
        // 置換後は直交性追跡が壊れる可能性があるので canonical_form は None に落とす
        let treetn = self.as_treetn().replaceind(old, new)?;
        Ok(Self::from_inner(treetn, None))
    }

    fn replaceinds(&self, old: &[Self::Index], new: &[Self::Index]) -> anyhow::Result<Self> {
        let treetn = self.as_treetn().replaceinds(old, new)?;
        Ok(Self::from_inner(treetn, None))
    }
}

impl TensorLike for TensorTrain {
    // GMRES で必要なメソッド
    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> anyhow::Result<Self> {
        TensorTrain::axpby(self, a, other, b).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn scale(&self, scalar: AnyScalar) -> anyhow::Result<Self> {
        TensorTrain::scale(self, scalar).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn inner_product(&self, other: &Self) -> anyhow::Result<AnyScalar> {
        Ok(self.inner(other))
    }

    fn norm_squared(&self) -> f64 {
        TensorTrain::norm_squared(self)
    }

    fn conj(&self) -> Self {
        // コーディング規約: library code で unwrap/expect を避ける
        // conj は Result を返せないので「失敗しない実装」に寄せる:
        // - clone して各 site tensor を conj に置き換える
        // - 内部の構造不整合があれば debug_assert しつつ clone を返す
        let mut result = self.clone();
        for site in 0..result.len() {
            let t = result.tensor(site).conj();
            result.set_tensor(site, t);
        }
        result
    }

    // 使用しないが実装が必要なメソッド
    fn factorize(&self, _left_inds: &[Self::Index], _options: &FactorizeOptions)
        -> std::result::Result<FactorizeResult<Self>, FactorizeError>
    {
        Err(FactorizeError::UnsupportedStorage("TensorTrain does not support factorize"))
    }

    fn contract(tensors: &[&Self], _allowed: AllowedPairs<'_>) -> anyhow::Result<Self> {
        anyhow::bail!("Use TensorTrain::contract() instead")
    }

    fn contract_connected(tensors: &[&Self], allowed: AllowedPairs<'_>) -> anyhow::Result<Self> {
        Self::contract(tensors, allowed)
    }

    fn direct_sum(&self, _other: &Self, _pairs: &[(Self::Index, Self::Index)])
        -> anyhow::Result<DirectSumResult<Self>>
    {
        anyhow::bail!("Use TensorTrain::add() instead")
    }

    fn outer_product(&self, _other: &Self) -> anyhow::Result<Self> {
        anyhow::bail!("TensorTrain does not support outer_product")
    }

    fn permuteinds(&self, _new_order: &[Self::Index]) -> anyhow::Result<Self> {
        anyhow::bail!("TensorTrain does not support permuteinds")
    }

    fn diagonal(input: &Self::Index, output: &Self::Index) -> anyhow::Result<Self> {
        // 単一サイトの恒等テンソルとして実装
        let delta = TensorDynLen::diagonal(input, output)?;
        Self::new(vec![delta]).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn scalar_one() -> anyhow::Result<Self> {
        Self::new(vec![]).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn ones(indices: &[Self::Index]) -> anyhow::Result<Self> {
        let t = TensorDynLen::ones(indices)?;
        Self::new(vec![t]).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn onehot(index_vals: &[(Self::Index, usize)]) -> anyhow::Result<Self> {
        let t = TensorDynLen::onehot(index_vals)?;
        Self::new(vec![t]).map_err(|e| anyhow::anyhow!("{}", e))
    }
}
```

### Phase 3: GMRES テストの実装

#### Plan (B): Directed indices による「ベクトル空間の閉性」確保

このテストでは `apply_a(x)` が常に `x` と同じ external indices を返す必要がある。
そのために以下の方針を採用する:

1. **Directed index 型を追加**（例: `DirectedDynIndex`）
    - `IndexLike::conj_state()` が `Ket/Bra` を返せる
    - `conj()` で `Ket <-> Bra` をトグルできる
    - ID は同一のまま ConjState だけが反転する
2. **MPS (x, b, x0)** は物理 index をすべて `Ket` 側で持つ
3. **Identity MPO** は各サイトで
    - 入力物理 index = `Bra`（ID は site と同一）
    - 出力物理 index = `Ket`（ID は site と同一）
    を持つ rank-4 tensor（+ bond indices）で構成する
4. MPO×MPS の縮約では `Bra` と `Ket` が contractable なので入力は正しく縮約され、出力は `Ket` のまま残る
    → `apply_a` の出力は `x` と同一の external indices（Ket）になり、GMRES が破綻しない

```rust
// crates/tensor4all-itensorlike/examples/test_gmres_mps.rs

//! Test: GMRES solver with MPS/MPO format
//!
//! A = Identity MPO
//! x = MPS with all elements = 1
//! b = A * x = x (since A = I)
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_mps --release

use tensor4all_core::krylov::{gmres, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen, TensorLike};
use tensor4all_itensorlike::{TensorTrain, ContractOptions};

fn main() -> anyhow::Result<()> {
    let n = 3;
    let phys_dim = 2;

    println!("=== Test: GMRES with MPS ===");
    println!("N = {}, phys_dim = {}", n, phys_dim);

    // 1. サイトインデックスを作成
    let sites: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // 2. Identity MPO を構築
    let mpo = create_identity_mpo(&sites)?;

    // 3. x_true (全要素 1 の MPS) を構築
    let x_true = create_ones_mps(&sites)?;

    // 4. b = A * x_true を計算（恒等演算子なので b = x_true）
    let b = mpo.contract(&x_true, &ContractOptions::zipup())?;

    // 5. 初期推定 (ゼロ MPS)
    let x0 = create_zero_mps(&sites)?;

    // 6. apply_a クロージャを定義
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        mpo.contract(x, &ContractOptions::zipup())
            .map_err(|e| anyhow::anyhow!("{}", e))
    };

    // 7. GMRES で解く
    let options = GmresOptions {
        max_iter: 50,
        rtol: 1e-10,
        max_restarts: 5,
        verbose: true,
    };

    let result = gmres(&apply_a, &b, &x0, &options)?;

    // 8. 結果検証
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Residual: {:.6e}", result.residual_norm);

    // x_sol と x_true の差のノルムを計算
    let diff = result.solution.axpby(
        AnyScalar::F64(1.0),
        &x_true,
        AnyScalar::F64(-1.0)
    )?;
    let error = diff.norm();
    println!("Error: {:.6e}", error);

    assert!(result.converged, "GMRES should converge");
    assert!(error < 1e-8, "Solution error too large: {}", error);

    println!("=== Test PASSED ===");
    Ok(())
}

/// Create an identity MPO
fn create_identity_mpo(sites: &[DynIndex]) -> anyhow::Result<TensorTrain> {
    // 各サイトに恒等テンソルを作成
    // ...
    todo!()
}

/// Create an MPS with all elements = 1
fn create_ones_mps(sites: &[DynIndex]) -> anyhow::Result<TensorTrain> {
    // 各サイトに全1ベクトルを作成
    // ...
    todo!()
}

/// Create a zero MPS
fn create_zero_mps(sites: &[DynIndex]) -> anyhow::Result<TensorTrain> {
    // 各サイトに全0ベクトルを作成
    // ...
    todo!()
}
```

## 実装チェックリスト

### Phase 1: TensorTrain への操作追加
- [x] `scale` メソッドを `tensortrain.rs` に追加
- [x] `axpby` メソッドを `tensortrain.rs` に追加
- [x] 単体テストを追加

### Phase 2: TensorLike の実装
- [x] `TensorIndex` を `TensorTrain` に実装
- [x] `TensorLike` を `TensorTrain` に実装
- [x] GMRES に必要なメソッドを実装
- [x] 使用しないメソッドは適切なエラーを返す
- [x] TensorLike 実装のテストを追加

### Phase 3: GMRES テスト
- [x] `examples/test_gmres_mps.rs` を作成
- [x] `create_identity_mpo` ヘルパー関数を実装
- [x] `create_ones_mps` ヘルパー関数を実装
- [x] GMRES による求解とテスト
- [x] `cargo run -p tensor4all-itensorlike --example test_gmres_mps --release` で動作確認

## 実装結果と発見した課題

### 成功点
- TensorTrain に `scale`, `axpby` メソッドを追加
- `TensorIndex`, `TensorLike` トレイトを実装
- Identity MPO を使った簡単なケースで GMRES が動作

### 現在の制約
1. **インデックス管理が複雑**: MPO適用後に `replaceinds` で出力インデックスを入力インデックスに置換する必要がある
2. **初期推定の制約**: x0=0 ではなく x0=b を使用しないと SVD が収束しない
3. **縮約方法**: `fit` メソッドを使用（`naive`, `zipup` は SVD エラー発生）
4. **ボンドインデックスの一致**: 異なる MPS 間で `add`/`inner` を行うにはボンドインデックスが一致している必要がある

### 次のステップ: Directed Index の設計
計画に記載の通り、`DynIndex` は `ConjState::Undirected` 固定のため、MPO×MPS の縮約後に出力が自動的に同じベクトル空間に戻らない。

**解決策案:**
1. `DirectedDynIndex` 型を追加（`Ket`/`Bra` を区別）
2. MPO の入力インデックスを `Bra`、出力を `Ket` として構築
3. MPS のインデックスを `Ket` として構築
4. 縮約時に `Bra` と `Ket` が自動的にマッチし、出力は `Ket` のまま残る

これにより、`replaceinds` の手動呼び出しが不要になり、GMRES がより自然に動作する。

## 技術的注意点

### 1. ボンド次元の増加

`axpby` (= `add`) を呼ぶたびにボンド次元が増加する：
- `a * self + b * other` → ボンド次元は `D_self + D_other`

GMRES の各イテレーションでボンド次元が増加するため、truncation が必要になる可能性がある。

**対策案：**
- `axpby` 後に自動的に truncation を適用するオプションを追加
- または、GMRES オプションに `max_bond_dim` を追加

### 2. 内積の計算

`TensorTrain::inner` は既に効率的に実装されている（サイトごとに縮約、`sim_linkinds` でリンクインデックスを区別）。

### 3. MPO の適用

`TensorTrain::contract` を使用して MPO を MPS に適用する。

## 参考ファイル

- [krylov.rs](crates/tensor4all-core/src/krylov.rs) - GMRES 実装
- [tensortrain.rs](crates/tensor4all-itensorlike/src/tensortrain.rs) - TensorTrain 実装
- [contract.rs](crates/tensor4all-itensorlike/src/contract.rs) - 縮約実装
- [linsolve.rs](crates/tensor4all-itensorlike/src/linsolve.rs) - 線形ソルバー（参考）

## 将来の拡張

1. **Truncation 付き GMRES**: ボンド次元の増加を制御
2. **非自明な MPO のテスト**: 対角行列、Pauli 演算子など
3. **複素数 MPS/MPO のテスト**
4. **性能ベンチマーク**: 密テンソル版との比較
5. **linsolve.rs との比較**: 既存の局所更新スイープ方式との性能比較
