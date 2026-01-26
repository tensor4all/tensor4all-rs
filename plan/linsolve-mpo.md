# SquareLinsolveUpdater の MPO 対応分析

## 問題の概要

GitHub Issue #185: `SquareLinsolveUpdater` が x と b が MPO (Matrix Product Operator) の場合に対応していない。

**エラー**: `Index count mismatch between init (5) and RHS (9)`

## 根本原因

現在のコードは x と b が MPS (Matrix Product State) であることを暗黙的に仮定しており、**external indices (batch indices)** という概念が存在しない。

## MPS と MPO の違い

### MPS の場合
- x: 各サイトに 1 つの physical index σ
- b: 各サイトに 1 つの physical index σ (x と同じ ID)
- A: input indices σ, output indices σ'

### MPO の場合
- x: 各サイトに 2 つの physical indices (σ_in, σ_out)
  - σ_in: A の input side と縮約される
  - σ_out: **external indices** (縮約されずに残る)
- b: 各サイトに 2 つの physical indices (σ_in, σ_out)
  - σ_in: **external indices** (x と同じ)
  - σ_out: A の output side と一致
- A: input indices, output indices

## MPS を仮定している箇所

### 1. `solve_local` (updater.rs:511-527) - 直接的なエラー箇所

```rust
let rhs_local = if init_indices.len() == rhs_indices.len() {
    // ...
} else {
    return Err(anyhow::anyhow!(
        "Index count mismatch between init ({}) and RHS ({})",
        ...
    ));
};
```

**問題**: x (init) と b (rhs) のインデックス数が等しいことを仮定。

- MPS: site indices + bond indices → 数が一致
- MPO: external indices の扱いが不明確で、縮約後のインデックス数が合わない

### 2. `ProjectedState::compute_environment` (projected_state.rs:178) - 根本的な問題

```rust
let bra_ket = T::contract(&[&bra_conj, tensor_ket], AllowedPairs::All)?;
```

**問題**: `AllowedPairs::All` で b† と reference_state (=x) の**全ての共通インデックス**を縮約。

- MPS: site index σ が 1 つ → 縮約されて OK
- MPO: site indices (σ_in, σ_out) が 2 つ
  - 縮約すべき: bond 方向のインデックス
  - **縮約すべきでない**: external indices

### 3. `ProjectedState::local_constant_term` (projected_state.rs:108-110)

```rust
let tensor_refs: Vec<&T> = all_tensors.iter().collect();
T::contract(&tensor_refs, AllowedPairs::All)
```

同様に `AllowedPairs::All` で全ての共通インデックスを縮約。external indices も縮約されてしまう。

### 4. `ProjectedOperator::compute_environment` (projected_operator.rs:291-295)

```rust
let mut tensor_refs: Vec<&T> = vec![&transformed_ket, tensor_op, &transformed_bra_conj];
tensor_refs.extend(child_envs.iter());
T::contract(&tensor_refs, AllowedPairs::All)
```

3-chain 縮約では operator を介するため直接的な問題は少ないが、bra_state と ket_state が同じ external indices を持つ場合、それらも縮約されてしまう可能性がある。

### 5. external indices という概念の欠如

`TreeTN::site_space(node)` は全ての site indices を返すが:
- 「どのインデックスが A と縮約されるべきか」
- 「どのインデックスが external (縮約されずに残る) か」

の区別がない。

## MPO での Ax = b の正しい構造

```
A: オペレーター
   - input indices: s_in (x と縮約)
   - output indices: s_out (結果に残る)

x: 解 (MPO)
   - A と縮約されるインデックス: s_in と同じ
   - external indices: 縮約されずに残る

b: RHS (MPO)
   - A の output と一致するインデックス: s_out と同じ
   - external indices: x と同じ (同じ ID)
```

**重要**: A の input 側の site indices と x の indices を比較するときは、**common indices** を使う必要がある。

## 必要な修正の方向性

### 1. external indices の概念を導入

x, b に対して「どのインデックスが A と縮約され、どれが external か」を識別する仕組みが必要。

オプション:
- `TreeTN` に `external_indices` フィールドを追加
- `SquareLinsolveUpdater` に external indices の情報を渡す
- A の input/output mapping から自動的に判定

### 2. 縮約時に common indices を明示的に指定

`AllowedPairs::All` ではなく、縮約すべきインデックスを明示的に指定:
- `AllowedPairs::Only(indices)` で縮約するインデックスを指定
- または `AllowedPairs::Excluding(indices)` で external indices を除外

### 3. A の input 側と x の common indices を使用

現在:
```rust
// x と b の全インデックスを比較
init_indices.len() == rhs_indices.len()
```

修正後:
```rust
// A の input indices と x の common indices を取得
let contracted_indices = common_indices(&a_input_indices, &x_indices);
// external indices = x の indices - contracted_indices
let external_indices = x_indices.difference(&contracted_indices);
```

### 4. インデックス構造の検証を修正

`solve_local` で単純な len 比較ではなく、正しいインデックス構造の検証を行う:
- x の external indices と b の external indices が一致するか
- x の縮約されるインデックスと A の input が一致するか
- b の対応するインデックスと A の output が一致するか

## 関連ファイル

- `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
- `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs`
- `crates/tensor4all-treetn/src/linsolve/square/local_linop.rs`
- `crates/tensor4all-treetn/src/linsolve/common/projected_operator.rs`
