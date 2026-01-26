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

### 2. `ProjectedState::compute_environment` (projected_state.rs:178) - bra/ket が逆 (バグ)

```rust
let bra_conj = tensor_bra.conj();  // tensor_bra は self.rhs (= b) から
let bra_ket = T::contract(&[&bra_conj, tensor_ket], AllowedPairs::All)?;
// tensor_ket は reference_state から
```

**問題**: `<b|ref>` を計算しているが、`ProjectedOperator` との一貫性から `<ref|b>` を計算すべき。

- `ProjectedOperator`: `<ref|H|x>` を計算（bra = ref†, ket = x）
- `ProjectedState`: `<ref|b>` を計算すべき（bra = ref†, ket = b）
- 現在のコード: `<b|ref>` を計算している（bra = b†, ket = ref）→ **逆**

正しくは：
```rust
let bra_conj = tensor_ref.conj();  // ref† を作る
let ket = tensor_b;                 // b
let bra_ket = T::contract(&[&bra_conj, &ket], ...)?;  // <ref|b>
```

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

## External Indices の縮約タイミング

nsite=1 update のダイアグラム分析 (diagram.jpg 参照):

```
方程式: <x|H|x> = <x|b>  (T_ref を使用)

左側 (環境計算):          右側 (ローカル update):
x* ─□─ ─□─□             x* ╭─□─ ─□─□
    │   │ │                 │
H  ─□─ ─□─□             x  ╰─□─ ─□─□
    │   │ │
x  ─□─ ─□─□
    ╰───╯ ╰─ external indices
```

### 環境内での external indices の扱い

- **環境計算**: external indices は縮約される (正しい)
  - x* と x が同じ external index ID を持つ → 縮約
  - ref と b が同じ external index ID を持つ → 縮約

- **ローカル update**: external indices は縮約しない (正しい動作)
  - active site の x には external indices が開いたまま残る
  - active site の b にも external indices が開いたまま残る
  - これらは同じ ID なので、local linear system の構造として一致する

### 現状の動作確認

現在のコードでは、active site において external indices は開いたまま残っている。
なぜなら環境テンソルには active site の indices が含まれず、
`local_constant_term` で active site の b テンソルと環境を縮約しても、
b の external indices は縮約対象がないため残る。

## 初期検証: External Indices の一致チェック

x と b の external indices は各サイトで（集合として）一致している必要がある。
これを `SquareLinsolveUpdater` の初期化時に検証すべき。

### 計算方法

```rust
// x の external indices = x.site_indices - input_mapping.true_index
// b の external indices = b.site_indices - output_mapping.true_index
```

- `input_mapping[node].true_index`: x のうち A と縮約されるインデックス
- `output_mapping[node].true_index`: b のうち A の output に対応するインデックス

### 検証ロジック (追加予定)

```rust
fn validate_external_indices<T, V>(
    state: &TreeTN<T, V>,           // x
    rhs: &TreeTN<T, V>,             // b
    input_mapping: &HashMap<V, IndexMapping<T::Index>>,
    output_mapping: &HashMap<V, IndexMapping<T::Index>>,
) -> Result<()> {
    for node in state.node_names() {
        // x の site indices から input_mapping の true_index を除く
        let x_site_ids: HashSet<_> = state.site_space(&node)
            .map(|s| s.iter().map(|i| i.id().clone()).collect())
            .unwrap_or_default();
        let x_contracted = input_mapping.get(&node)
            .map(|m| m.true_index.id().clone());
        let x_external: HashSet<_> = x_site_ids.iter()
            .filter(|id| Some(*id) != x_contracted.as_ref())
            .cloned().collect();

        // b についても同様
        let b_site_ids: HashSet<_> = rhs.site_space(&node)
            .map(|s| s.iter().map(|i| i.id().clone()).collect())
            .unwrap_or_default();
        let b_contracted = output_mapping.get(&node)
            .map(|m| m.true_index.id().clone());
        let b_external: HashSet<_> = b_site_ids.iter()
            .filter(|id| Some(*id) != b_contracted.as_ref())
            .cloned().collect();

        if x_external != b_external {
            return Err(anyhow!(
                "External indices mismatch at node {:?}: x has {:?}, b has {:?}",
                node, x_external, b_external
            ));
        }
    }
    Ok(())
}
```

### 追加場所

1. `with_index_mappings()`: 早期エラーとして返す
2. `verify()`: レポートにも含める

## 必要な修正の方向性

### 1. external indices の概念を導入

x, b に対して「どのインデックスが A と縮約され、どれが external か」を識別する仕組みが必要。

オプション:
- `TreeTN` に `external_indices` フィールドを追加
- `SquareLinsolveUpdater` に external indices の情報を渡す
- A の input/output mapping から自動的に判定

### 2. A の input 側と x の common indices を使用

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

### 3. インデックス構造の検証を修正

`solve_local` で単純な len 比較ではなく、正しいインデックス構造の検証を行う:
- x の external indices と b の external indices が一致するか
- x の縮約されるインデックスと A の input が一致するか
- b の対応するインデックスと A の output が一致するか

## 関連ファイル

- `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
- `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs`
- `crates/tensor4all-treetn/src/linsolve/square/local_linop.rs`
- `crates/tensor4all-treetn/src/linsolve/common/projected_operator.rs`
