# Issue #192: linsolve での SVD エラー（NaN入力）修正プラン（日本語）

対象: Issue #192（SVD convergence failure in linsolve at N≥5）

ブランチ: `feature/linsolve-benchmark`

## 0. 目的 / ゴール

- `SquareLinsolveUpdater` を用いた linsolve 実行で、N≥5（例: N=5, bond_dim=20）において発生する
  - `SvdError(ComputationError(...))`
  - 実際には SVD 入力が NaN に壊れている
  という不具合を解消する。
- 正規形（canonical form）を sweep の各ステップで一貫して保ち、環境テンソル `<ref|rhs>` のノルム爆発 → overflow → NaN を防ぐ。
- ユーザーが指定した `TruncationOptions`（`rtol`, `max_rank`, `form`）が factorize 側に正しく反映されるようにする。

## 1. 非ゴール

- 既存アルゴリズム（GMRES、projected operator/state）の理論的改良や高速化はこの対応では行わない。
- 例外的に巨大ノルムな初期状態を「ユーザー側で回避」させる（初期トランケーション必須化等）方針にはしない。

## 2. 再現手順（現状）

以下で N=5 で再現:

```bash
cargo run -p tensor4all-treetn --example repro_linsolve_single_run --release -- 5 20 identity ones 1 0
```

期待: `(a0*I + a1*A)x=b` で a0=1, a1=0（恒等）なので、`x=b` のまま正常に動作。

実際: sweep 中に factorize へ渡る 2x2 行列が NaN になり、SVD が失敗。

## 3. 原因（debug.md まとめ）

### 3.1 直接原因

- SVD に渡される小行列が NaN 埋め尽くし。
- SVD の「収束しない」のではなく、入力が overflow/NaN で壊れている。

### 3.2 根本原因

- `ProjectedState::local_constant_term` が使う環境テンソル `<ref|rhs>` が、正規形の破綻により指数爆発。
- 破綻のトリガは、`factorize_tensor_to_treetn_with` の **root 選択が sweep plan の `new_center` と一致しない** こと。
  - 2ノード領域では tie-break により「名前が小さい側」が root になりがち。
  - `Canonical::Left` 固定の分解では、`S*Vh`（ノルム保持側）が root に残り、結果として **ノルムが new_center に移動しない**。
  - `subtree.set_canonical_center([new_center])` と実際のノルム配置が矛盾し、次ステップの `<ref|rhs>` でノルムが二乗され続ける。

### 3.3 追加の悪化要因

- `SquareLinsolveUpdater::update` 内の factorize 設定が `max_rank` のみで、`rtol` が伝播していない。
  - 不要なランク/特異値が残りやすく、数値安定性が悪化しうる。

## 4. 修正方針（推奨）

### 方針A（本命）: factorize の root を `new_center` に一致させる

**狙い**: 正規中心（canonical center）にノルムが集まる（あるいは正規形の前提が満たされる）ようにし、環境テンソルへ巨大ノルムが混入しないようにする。

- `factorize_tensor_to_treetn_with` に「root を明示指定」できるルートを追加する。
- `SquareLinsolveUpdater::update` からは、必ず `root = step.new_center` を渡す。

実装方針（互換性は無視）:

- 既存の factorize API を破壊的に変更し、root を **必須** にする。
  - `factorize_tensor_to_treetn_with(..., options, root: &V)`
  - （必要なら）`factorize_tensor_to_treetn(..., root: &V)` も同様に root 必須
- linsolve の local update では必ず `root = step.new_center` を渡す。
- decompose の返り値 TreeTN は `canonical_center = {root}` を必ずセットする。

これにより「new root に必ず正規化中心（ノルム保持側）が移る」ことを API レベルで強制する。

### 方針D（併用）: factorize に `rtol` を伝播

- `SquareLinsolveUpdater::update` で `self.options.truncation.rtol()` を `FactorizeOptions::with_rtol` で設定。
- `max_rank` と併用。

※これは根本原因（root/new_center不一致）を直すものではないが、数値安定性を上げる補助として有効。

## 5. 具体的な実装ステップ（コード変更の順序）

### Step 1: root 必須の factorize API に変更（破壊的変更）

- 対象: `crates/tensor4all-treetn/src/treetn/decompose.rs`
- 変更:
  - `factorize_tensor_to_treetn_with(..., options, root: &V)` に変更（root 必須）
  - root 自動選択（degree/tie-break）は廃止し、呼び出し側が root を決める

受け入れ条件:
- root を指定した場合、post-order の最後（root）に `S*Vh` が残り、ノルム保持が `new_center` に乗る。

### Step 2: linsolve updater 側で root=new_center を必ず使用

- 対象: `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
- `factorize_tensor_to_treetn_with(..., root=&step.new_center)` を必ず使用

加えて、subtree 内の `ortho_towards` も `new_center` 方向へ更新し、
「メタデータ上の中心だけ移動して実体が追従しない」状況を避ける。

受け入れ条件:
- forward sweep（step0-3）で正規中心とノルム保持側が一致し、debug.md で観測された矛盾が消える。

### Step 3: factorize options に rtol/max_rank を反映

- 対象: `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
- `FactorizeOptions::svd()` の生成後に:
  - `with_max_rank(...)`
  - `with_rtol(...)`（`self.options.truncation.rtol()` が `Some` の場合）

受け入れ条件:
- `TruncationOptions` に設定した rtol が実際の分解に反映される。

### Step 4: 退行テスト（再現コードをテスト化）

- 対象候補:
  - `crates/tensor4all-treetn/tests/issue192_regression.rs`（integration test）
- 内容:
  - N=5, bond_dim=20, identity MPO, rhs=ones, a0=1,a1=0 を 1 sweep 以上回し
  - panic/Err にならないこと
  - 途中のテンソルが NaN/Inf を含まないこと（可能ならチェック）

※ テストは public API を使う（内部表現に依存しない）。

### Step 5: 追加検証

- `repro_linsolve_single_run` で N=5,6,10 を再実行
- `a0=0,a1=1`（Aを使うケース）や `rhs=ax` でも sanity check

## 6. リスクと注意点

- root 指定は「分解木の向き」そのものを変えるため、他のアルゴリズム（例えば canonicalization/truncate）と整合しているか確認が必要。
- forward/backward sweep の折り返し（step4以降）は root/new_center が一致するケースが多いが、一般の木構造ではより複雑になりうる。
- 既存コードが暗黙に「rootはヒューリスティックでよい」前提を持っている場合、挙動変化が出る可能性がある。

## 7. 完了条件（Definition of Done）

- N=5 の最小再現で SVD エラーが消える（NaN入力が発生しない）。
- N=10 でも同様に走る（少なくとも `a0=1,a1=0, identity/ones` のケース）。
- 追加した regression test が CI でパス。
- `cargo fmt --all` と `cargo clippy --workspace` が通る（実装開始後に実施）。

---

補足: このプランは [debug.md](../debug.md) の観測（root/new_center不一致→ノルム爆発→NaN）を前提にしており、実装前にもう一度「root指定でノルム配置が改善する」ことをログで確認できると確度が上がる。
