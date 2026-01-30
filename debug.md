# Issue #192: SVD convergence failure in linsolve at N>=5 — 原因分析

## 直接原因

SVDに渡されている2x2行列の全要素が **NaN** になっている。SVDアルゴリズムの収束問題ではなく、入力データが既に壊れている。

```
=== SVD FAILED: matrix size 2x2 ===
NaN, NaN
NaN, NaN
```

## NaN発生メカニズム

数値のオーバーフロー（`f64::MAX` ~ 1.8e308 を超える）により Inf → NaN が生じている。

## 根本原因: 環境テンソル `<ref|rhs>` のノルム爆発

問題箇所は `ProjectedState::local_constant_term`（[projected_state.rs:78](crates/tensor4all-treetn/src/linsolve/square/projected_state.rs#L78)）における環境テンソルの計算。

### 発生の流れ

1. **All-ones MPS（bond_dim=20）** をQR正規化すると、正規中心（site0）に全ノルムが集中する
   - site0のノルム ~ `sqrt(2^5) * 20^4` = 9.05e5

2. `SquareLinsolveUpdater::update`（[updater.rs:735](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L735)）では、`solve_local` でGMRESを呼ぶ際に局所的なRHS `b_local` を計算する

3. `b_local` の計算には **reference_state**（bra側）と **rhs**（ket側）の部分内積 `<ref|rhs>` で構成される環境テンソルが使われる

4. 環境計算で、正規中心のテンソル同士が掛け合わされる:
   - `ref[site0]` x `rhs[site0]` ~ 9e5 x 9e5 = **8.1e11**

5. これがGMRESの右辺に影響し、解のノルムも巨大になる

6. 解をSVD分解（`factorize_tensor_to_treetn_with`、[decompose.rs:131](crates/tensor4all-treetn/src/treetn/decompose.rs#L131)）してMPSに戻す際、Left canonical形式で分解するため右側テンソル（`S*Vh`）にノルムが蓄積される

7. **次のステップ** で reference_state が更新され（`sync_reference_state_region`、[updater.rs:582](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L582)）、巨大ノルムのテンソルが再び環境計算に使われることでノルムが **指数関数的に増大** する

## ステップごとのノルム変化（N=5, sweep 1）

| ステップ | region | site0 | site1 | site2 | site3 | site4 |
|---------|--------|-------|-------|-------|-------|-------|
| 初期 | - | 9.05e5 | 1.00 | 1.00 | 1.41 | 1.00 |
| step 0 | [0,1] | 9.05e5 | 1.00 | 1.00 | 1.41 | 1.00 |
| step 1 | [1,2] | 9.05e5 | **8.19e11** | 1.00 | 1.41 | 1.00 |
| step 2 | [2,3] | 9.05e5 | 8.19e11 | **6.71e23** | 1.00 | 1.00 |
| step 3 | [3,4] | 9.05e5 | 8.19e11 | 6.71e23 | **4.50e47** | 1.00 |
| step 4 | [4,3] | 9.05e5 | 8.19e11 | 6.71e23 | 4.50e47 | 1.00 |
| step 5 | [3,2] | 9.05e5 | 8.19e11 | **4.90e55** | 1.00 | 1.00 |
| step 6 | [2,1] | 9.05e5 | **5.99e51** | 1.00 | 1.00 | 1.00 |
| step 7 | [1,0] | **1.10e42** | 1.00 | 1.00 | 1.00 | 1.00 |

指数が約2倍ずつ増加（5→11→23→47）。環境の内積で前ステップのノルムが二乗されるため。

## N=4で成功し N>=5 で失敗する理由

N=4の場合、MPS全体のノルムが `sqrt(2^4) * 20^3` ~ 32000 と比較的小さく、sweep内の指数爆発が `f64` の範囲内に収まる。N=5以上では `20^4 = 160000` 以上のノルムが正規中心に集中し、指数爆発がオーバーフローに至る。

## 再現方法

```bash
cargo run -p tensor4all-treetn --example repro_linsolve_single_run --release -- 5 20 identity ones 1 0
```

ブランチ: `feature/linsolve-benchmark`

## 関連コード箇所

| ファイル | 行 | 役割 |
|---------|-----|------|
| [updater.rs](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L735) | 735-764 | `update`: solve_local → factorize のメインループ |
| [updater.rs](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L750) | 750-753 | factorize_options: SVD固定、`rtol` 未設定 |
| [projected_state.rs](crates/tensor4all-treetn/src/linsolve/square/projected_state.rs#L78) | 78-118 | `local_constant_term`: 環境テンソルを含むRHS計算 |
| [projected_state.rs](crates/tensor4all-treetn/src/linsolve/square/projected_state.rs#L142) | 142-196 | `compute_environment`: `<ref\|rhs>` 環境テンソル構築 |
| [decompose.rs](crates/tensor4all-treetn/src/treetn/decompose.rs#L131) | 131-305 | `factorize_tensor_to_treetn_with`: Left canonical SVD分解ループ |
| [svd.rs](crates/tensor4all-core/src/defaults/svd.rs#L249) | 249-315 | `svd_truncated_usvh`: NaN行列を受け取りSVD失敗 |

## Step 0 / Step 1 の詳細トレース

### Sweep plan（N=5, 2-site sweep, center="site0"）

```
step 0: nodes=["site0", "site1"], new_center="site1"
step 1: nodes=["site1", "site2"], new_center="site2"
step 2: nodes=["site2", "site3"], new_center="site3"
step 3: nodes=["site3", "site4"], new_center="site4"
step 4: nodes=["site4", "site3"], new_center="site3"  ← 折り返し
step 5: nodes=["site3", "site2"], new_center="site2"
step 6: nodes=["site2", "site1"], new_center="site1"
step 7: nodes=["site1", "site0"], new_center="site0"
```

### Step 0: region=["site0","site1"], new_center="site1"

#### 初期状態

QR正規化（`CanonicalForm::Unitary`）後のMPS。正規中心site0に全ノルムが集中:

```
site0: dims=[2, 1],   norm=9.05e5  ← 正規中心
site1: dims=[2, 1, 1], norm=1.00    （直交）
site2: dims=[2, 2, 1], norm=1.00    （直交）
site3: dims=[2, 1, 2], norm=1.41    （直交）
site4: dims=[2, 1],   norm=1.00    （直交）
```

#### 1. `contract_region`

site0（norm=9e5）× site1（norm=1）→ bond を縮約して局所テンソルに。norm ~ 9e5。

#### 2. `solve_local`（GMRES）

`a0=1, a1=0` → 演算子は恒等写像 `I*x = b`。

- **local RHS** = `local_constant_term`:
  - rhs[site0] × rhs[site1] × 環境（site2側）
  - 環境は region の外側。site2以降のref/rhsのnormはすべて ~1 → 環境 ~1
  - → local RHS ≈ norm ~9e5
- GMRES解 ≈ local RHS（恒等写像）→ norm ~9e5

#### 3. `factorize_tensor_to_treetn_with` — ルート選択の問題

2ノード `["site0","site1"]` のトポロジーで分解ツリーのルートを決定する（[decompose.rs:198-210](crates/tensor4all-treetn/src/treetn/decompose.rs#L198-L210)）:

```rust
let root = adj.iter().max_by(|a, b| {
    a.1.len().cmp(&b.1.len())          // degree が大きいほう
        .then_with(|| b.0.cmp(a.0))    // tie-break: 名前が小さいほう
})
```

両ノード degree=1（相互に1本のedgeのみ）→ tie-break で **root = "site0"**。

分解は post-order（葉→ルート）で処理:

```
1. node = site1（葉）
   left_inds = site1の物理index + 外部bond(site1→site2)
   SVD(Canonical::Left): left = U (直交), right = S*Vh (ノルム保持)
   → node_tensors["site1"] = U        … norm ~1
   → current_tensor = S*Vh            … norm ~9e5

2. node = site0（root） → 残りテンソルをそのまま受け取る
   → node_tensors["site0"] = current_tensor (= S*Vh) … norm ~9e5
```

#### 4. Step 0 の結果

```
site0: S*Vh → norm = 9.05e5  （非直交、全ノルムを保持）
site1: U   → norm = 1.00     （直交）
```

**`set_canonical_center(["site1"])` が呼ばれるが、実際のノルムは site0 にある。**
正規中心は site1 と宣言されているが、site0 が非直交でノルムを持つ — 正規形の矛盾。

原因: `factorize_tensor_to_treetn_with` のルート（= ノルムの受け手）が "site0" であり、sweep plan の `new_center` = "site1" と一致しない。

---

### Step 1: region=["site1","site2"], new_center="site2" — ノルム爆発

#### 入力状態（step 0 の結果）

```
site0: norm=9.05e5  （非直交 — S*Vh を保持）
site1: norm=1.00    （直交 U）← 宣言上の正規中心
site2: norm=1.00    （直交）
...
```

#### 1. `contract_region`

site1（norm=1）× site2（norm=1）→ norm ~1。**問題なし。**

#### 2. `solve_local`（GMRES）← ノルム爆発の起点

local RHS = `local_constant_term`（[projected_state.rs:78-118](crates/tensor4all-treetn/src/linsolve/square/projected_state.rs#L78-L118)）:

```
b_local = rhs_local × environments
```

- **rhs_local**: rhs[site1] × rhs[site2]。norm ~1。
- **環境テンソル**（site0 → site1 方向）:

  `compute_environment`（[projected_state.rs:142](crates/tensor4all-treetn/src/linsolve/square/projected_state.rs#L142)）で `<ref|rhs>` をsite0で計算:

  ```
  env(site0→site1) = conj(ref[site0]) × rhs[site0]
  ```

  - `ref[site0]` = reference_state の site0。step 0 の `after_step` → `sync_reference_state_region`（[updater.rs:582](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L582)）により ket_state からコピー済み → **norm = 9.05e5**
  - `rhs[site0]` = 元のrhsの site0（不変）→ **norm = 9.05e5**
  - → **env ≈ 9.05e5 × 9.05e5 = 8.19e11**

- **b_local** = rhs_local(~1) × env(**8.19e11**) ≈ **8.19e11**

GMRESは恒等写像を解く → 解 ≈ b_local → **norm ≈ 8.19e11**

#### 3. `factorize_tensor_to_treetn_with`

2ノード `["site1","site2"]` → 同様に root = "site1"（tie-break）。

```
1. node = site2（葉）→ U, norm ~1
2. node = site1（root）→ S*Vh, norm ~8.19e11
```

#### 4. Step 1 の結果

```
site0: norm=9.05e5
site1: norm=8.19e11  ← 1 → 8.19e11 に爆発！ = (9.05e5)²
site2: norm=1.00
...
```

---

### 爆発の連鎖

以降のステップでも同じメカニズムが繰り返される:

- **step 2** (region=[2,3]): site1 の norm = 8.19e11 が環境に入る
  - env = ref[site0](9e5) × rhs[site0](9e5) × ref[site1](8.19e11) × rhs[site1](1) ≈ 8.19e11 × 8.19e11 ≈ 6.71e23
  - → site2: **6.71e23**

- **step 3** (region=[3,4]): 同様にsite2のnormが環境に
  - → site3: **4.50e47** ≈ (6.71e23)²

指数の倍々ゲーム: 5 → 11 → 23 → 47 → ... → overflow → NaN

---

### 2つの複合問題

#### 問題1: `factorize_tensor_to_treetn_with` のルート選択が `new_center` と不一致

[decompose.rs:198-210](crates/tensor4all-treetn/src/treetn/decompose.rs#L198-L210) でルートは degree 最大 + 名前 tie-break で決まる。2ノードの場合、常に名前が小さいノードがルートになる。

| step | region | root（実際） | new_center（期待） | 一致? |
|------|--------|-------------|-------------------|-------|
| 0 | [site0, site1] | site0 | site1 | **不一致** |
| 1 | [site1, site2] | site1 | site2 | **不一致** |
| 2 | [site2, site3] | site2 | site3 | **不一致** |
| 3 | [site3, site4] | site3 | site4 | **不一致** |
| 4 | [site4, site3] | site3 | site3 | 一致 |
| 5 | [site3, site2] | site2 | site2 | 一致 |
| 6 | [site2, site1] | site1 | site1 | 一致 |
| 7 | [site1, site0] | site0 | site0 | 一致 |

forward sweep（step 0-3）ではすべて不一致。ノルムがルート（= sweep 進行方向と逆のノード）に残り、正規中心にノルムが配置されない。

本来、`new_center` がルート（= S*Vh の受け手）になるべき。そうすれば正規中心にノルムが集中し、環境テンソルの入力は直交テンソル（norm ~1）のみとなり、ノルム爆発が起きない。

#### 問題2: 環境テンソル `<ref|rhs>` でノルムが二乗される

- reference_state の非直交テンソル（norm=N）× rhs の非直交テンソル（norm=N）= env(norm=N²)
- これが GMRES 解の norm に反映され、次の step でさらに二乗
- 結果: 指数の倍々（5 → 11 → 23 → 47 → ... → overflow → NaN）

正規形が正しく維持されていれば、環境に入るテンソルはすべて直交（norm ~1）なので、この問題は発生しない。

## 修正方針

### 方針A（推奨）: `factorize_tensor_to_treetn_with` にルートノードを指定可能にする

**根本原因の直接修正。** 現在のルート選択ロジック（degree最大 + 名前tie-break）を変更し、呼び出し元が `new_center` をルートとして指定できるようにする。

変更箇所:
- [decompose.rs:131](crates/tensor4all-treetn/src/treetn/decompose.rs#L131): `factorize_tensor_to_treetn_with` のAPIに `root` パラメータを追加（`Option<V>` で後方互換性を保持）
- [updater.rs:750-755](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L750-L755): `factorize_tensor_to_treetn_with` 呼び出し時に `step.new_center` をルートとして渡す

期待効果:
- ルート = new_center → S*Vh（ノルム保持テンソル）が正規中心に配置される
- 環境テンソルの入力がすべて直交テンソル（norm ~1）になる
- ノルム爆発が根本的に解消

```rust
// 修正イメージ（updater.rs）
let decomposed = factorize_tensor_to_treetn_with(
    &solved_local,
    &topology,
    factorize_options,
    Some(&step.new_center),  // ルートノードを指定
)?;
```

### 方針B: `Canonical::Right` の使用

`factorize_tensor_to_treetn_with` で `Canonical::Left`（左直交）が固定されている（[decompose.rs:241-245](crates/tensor4all-treetn/src/treetn/decompose.rs#L241-L245)）。forward sweep では `Canonical::Right`（右直交: left=U*S, right=Vh）にすれば、ルートが名前順で選ばれてもノルムが葉側（= sweep 進行方向 = new_center 側）に行く。

ただし、sweep方向に応じてcanonical方向を切り替える必要があり、forward/reverseの判定ロジックが複雑になる。方針Aのほうがシンプル。

### 方針C: reference_state の正規化

各ステップ前に reference_state を正規化（canonical formに変換）すれば、環境テンソルの入力が常に norm ~1 となりノルム爆発を防げる。

欠点: 毎ステップの正規化はコストが高く、本来不要な処理。正規形が正しく維持されていればこの問題は起きないので、対症療法に過ぎない。

### 方針D: factorize_options に rtol を渡す

現在 `max_rank` のみ設定されている（[updater.rs:750-753](crates/tensor4all-treetn/src/linsolve/square/updater.rs#L750-L753)）。ユーザー指定の `rtol` を反映すれば、冗長な特異値が除去されノルム蓄積が緩和される可能性。

ただし、ノルム爆発の根本原因（ルートと正規中心の不一致）は解消しないため、大きなノルムでは依然問題が残る。方針Aと併用すべき。

### 方針E: 初期状態の事前トランケーション

all-ones MPS（bond_dim=20）のように冗長な初期状態を事前にトランケートして適正なbond_dimにすれば、正規中心のノルムが小さくなり爆発が起きにくくなる。

ただしユーザー側の回避策であり、ライブラリ側の修正ではない。

### 推奨

**方針A** を主とし、方針Dを併用するのが最善。方針Aでルート選択を修正すれば、正規形が正しく維持され、環境テンソルのノルム問題が根本的に解消される。方針Dはそれに加えて不要なランクを削減し数値安定性を向上させる。
