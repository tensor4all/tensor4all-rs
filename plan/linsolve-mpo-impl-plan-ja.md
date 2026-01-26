# MPO版 linsolve 対応：実装プラン（日本語）

対象：`tensor4all-treetn` の `SquareLinsolveUpdater` を、未知数 `x` と右辺 `b` が **MPO** の場合に対応させる。

関連：
- 既存分析：`plan/linsolve-mpo.md`
- 再現例：`crates/tensor4all-treetn/examples/test_linsolve_mpo_identity.rs`
- 実装：`crates/tensor4all-treetn/src/linsolve/square/{updater.rs,projected_state.rs,local_linop.rs}`
- 実装：`crates/tensor4all-treetn/src/linsolve/common/projected_operator.rs`

---

## 0. 目標仕様を先に固定する（最重要）

### 想定する方程式
- 連立：\(A X = B\)
- ここで
  - `A`: MPO（作用素）
  - `X`: MPO（未知の作用素）
  - `B`: MPO（右辺の作用素）

### 1ノード（サイト）ごとのインデックス役割
ノード `v` について：
- `X.site_space(v)` は **2つ**の site index（`σ_in`, `σ_ext`）を持つ
  - `σ_in`: `A` の input 側と縮約される（「縮約対象」）
  - `σ_ext`: **external index**（縮約されずに残る）
- `B.site_space(v)` も **2つ**の site index（`τ_out`, `σ_ext`）を持つ
  - `τ_out`: `A` の output 側と一致し縮約される（「縮約対象」）
  - `σ_ext`: **external index**（`X` と一致する必要）

### 必須整合条件（ノードごと）
- `X` の external indices と `B` の external indices が一致（ID と dim）
- `X` の縮約対象 index は `input_mapping[v].true_index` により特定できる
- `B` の縮約対象 index は `output_mapping[v].true_index` により特定できる

> 目的：`TreeTN` 本体に「external index」概念を追加しなくても、**mapping から外部/縮約対象を判定**できる形にする。

---

## 1. `<ref|H|x>` と `<ref|b>` の対応関係をチェックする（最初にやる）

MPO 対応に入る前に、まず **「bra/ket の向き」が設計どおりか**を機械的に検査する。
ここがズレていると RHS 側の open indices が崩れて、`solve_local` の整列が必ず破綻しやすい。

### 期待する対応関係
- `ProjectedOperator`：`<ref|H|x>`（bra = `reference_state`†, ket = `state`(=x)）
- `ProjectedState`：`<ref|b>`（bra = `reference_state`†, ket = `rhs`(=b)）

### 実装するチェック（アイデア）
場所候補：
- `SquareLinsolveUpdater::before_step`（環境構築・local solve に入る直前に必ず通る）
- もしくは `SquareLinsolveUpdater::verify()`（レポートとして見たい場合）

チェック内容（第一段階は “観測” と “早期失敗” に寄せる）：
- ある step（最初の step でよい）で `rhs_local_raw` を作った直後に、
  - 「`rhs_local_raw` の open indices が `init` と整合するはず」という前提が成り立つか確認する
  - 成り立たない場合は、余分/不足の index ID/dim を列挙して **bra/ket の逆転疑い**を含むヒントを出す

**合格条件**
- 「`ProjectedState` の環境が `<ref|b>` になっているか（or 逆転しているか）」を、
  実行時にすぐ判断できる（少なくとも、失敗時に説明可能なログが出る）

---

## 2. 再現と観測（最初の 30 分でやる）

### 再現手順
- `cargo run -p tensor4all-treetn --example test_linsolve_mpo_identity --release`

### 観測したい情報（最初の失敗ステップ）
`SquareLinsolveUpdater::solve_local` 内で：
- `init.external_indices()` の ID/dim（列）
- `rhs_local_raw.external_indices()` の ID/dim（列）
- region（更新ノード集合）
- `rhs_local_raw` に「余分に残っている（or 欠けている）」インデックスの具体的なID/dim

**合格条件**
- 「なぜ `init=5, rhs=9` なのか」を、どの index が余分/不足かで説明できる

---

## 3. 早期バリデーション（fail fast）を入れる

場所候補：
- `SquareLinsolveUpdater::with_index_mappings(...)` 直後
- もしくは `SquareLinsolveUpdater::verify()` に加えて、`with_index_mappings` からも呼ぶ

### バリデーション項目（ノードごと）
1) **site_space の個数**
- MPOモードでは `X.site_space(v).len()==2` かつ `B.site_space(v).len()==2`
  - 将来拡張を考えるなら「>=2 で external を集合として扱う」でもよいが、まずは 2 固定で良い

2) **mapping が site_space に含まれている**
- `input_mapping[v].true_index` が `X.site_space(v)` に含まれる（ID一致）
- `output_mapping[v].true_index` が `B.site_space(v)` に含まれる（ID一致）

3) **external index の一致**
- `external_x(v) = X.site_space(v) - {input_mapping[v].true_index}`
- `external_b(v) = B.site_space(v) - {output_mapping[v].true_index}`
- `external_x(v) == external_b(v)`（ID/dim）

**合格条件**
- sweep に入る前に、MPO構造が不正なら「どのノードで何が不一致か」を明確にエラー表示できる

---

## 4. `ProjectedState` の bra/ket 方向を確定して修正する（重要）

`ProjectedOperator` が `<ref|H|x>` を作るなら、`ProjectedState` は `<ref|b>` を作るのが自然。

やること：
- `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs` の環境計算が
  - `<b|ref>` になっていないか
  - `<ref|b>` を作っているか
を確認し、必要なら修正。

**合格条件**
- MPS既存テストに悪影響がない
- MPOケースで `rhs_local_raw` の open indices が期待に近づく（少なくとも「外部が勝手に消える/増える」挙動が減る）

---

## 5. external indices の縮約タイミングを検証する（環境 vs ローカル）

`plan/linsolve-mpo.md` のダイアグラム整理のとおり、MPO の external indices は
「環境では縮約されてよいが、ローカル update（active site/region）では開いたまま残る」
という挙動が期待される。

### 期待する挙動
- **環境計算**（open region の外側を潰す側）:
  - `x*` と `x` が同じ external index ID を持つ部分は **縮約されてよい**
  - `ref` と `b` が同じ external index ID を持つ部分も **縮約されてよい**
- **ローカル update**（open region 内の local linear system を解く側）:
  - active site/region の `x` の external indices は **残る**
  - active site/region の `b` の external indices も **残る**
  - これら external は **同じ ID** を持ち、local system の index 構造として整合する必要がある

### チェック観点（実装前提の検証項目）
- `ProjectedState::local_constant_term` が返す `rhs_local_raw` について、
  - active region の external indices が「残っている」こと
  - その external indices が `init` 側（x の local tensor）の external と一致すること
- 逆に、active region の外側では external が環境内で縮約されており、
  `rhs_local_raw` に不要な external が “余分に残っていない” こと

**合格条件**
- `rhs_local_raw` と `init` の「open indices のズレ」を
  external の縮約タイミングの観点で説明でき、次の修正（AllowedPairs 制御/整列ロジック）に落とせる

---

## 6. `AllowedPairs::All` による external indices の “縮約しすぎ” を点検・制御する

元分析で指摘されている通り、`AllowedPairs::All` は「共通IDのインデックスを全て縮約」するため、
設計上 “残すべき external indices” まで縮約してしまう（または逆に、残り方が想定とズレる）
リスクがある。

### 点検対象（重要）
- `ProjectedState::local_constant_term`（`projected_state.rs`）
  - 環境テンソルと `b` の縮約で external が意図せず消えていないか
- `ProjectedState::compute_environment`（`projected_state.rs`）
  - `<ref|b>` の 2-chain 構築で external の扱いが設計通りか
- `ProjectedOperator::compute_environment`（`projected_operator.rs`）
  - 3-chain でも external が共通IDである場合に縮約されうるため、期待と合うか

### 方針（実装でやることの方向性）
- MPOモードでは、縮約対象を “All” ではなく
  - 「縮約して良い index（例：link/bond + contracted side）」と
  - 「縮約してはいけない index（external）」を分離して扱う
- まずはログ/検証で “どの index が縮約されているか” を確定し、
  必要なら `AllowedPairs` の与え方を制御する（external を除外するなど）

**合格条件**
- external indices が “消えた/残った” の原因が `AllowedPairs::All` 起因かどうか切り分けできる
- 切り分け結果に基づいて、次の `solve_local` 整列（または ProjectedState 側）修正に繋げられる

---

## 7. `solve_local` の「len一致」前提を撤廃し、構造に基づき整列する

現状は：
- `init.external_indices().len() == rhs_local_raw.external_indices().len()` でなければ即エラー

MPO対応では：
- 「external / contracted」の区別を用いて、`rhs_local_raw` を `init` と同じ index リストへ整列できるかを試みる

方針：
1) まず `init` が期待する external index リスト（順序付き）を基準にする
2) `rhs_local_raw` が同一集合なら `permuteinds` で合わせる
3) 集合が違う場合は、
   - “余分な index” と “不足している index” を列挙して、
   - **原因推定（ProjectedStateの環境が外部を縮約し過ぎ/不足、mapping不整合など）**を添えてエラーにする

**合格条件**
- MPO identity の最小例で、少なくとも sweep 1 の最初の local step を通過できる
  - ここまで到達しない場合は、3（ProjectedState）か 2（validation）の不備を疑う

---

## 8. テスト戦略（段階的）

1) まず example（再現）で「index mismatch が消える」ことを確認
2) 次に小さいテストを追加
   - external index validation のユニットテスト
   - `ProjectedState` が `<ref|b>` を作ることのテスト（open indices の集合一致など）
3) 最後に MPO identity solve が「ある程度進む」ことを確認するテスト

**注意**
- 収束 tolerance の緩和は基本しない（必要ならユーザー合意を取る）

---

## 9. 実装順序（推奨）

1) `<ref|H|x>` と `<ref|b>` になっているかのチェックを入れて観測（まず向きを確定）
2) 観測（ログ）で `5 vs 9` の内訳を特定
3) 早期バリデーション導入
4) external indices の縮約タイミング（環境 vs ローカル）を検証
5) `AllowedPairs::All` による縮約しすぎの有無を点検（必要なら制御方針を決める）
6) `ProjectedState` bra/ket 修正（必要なら）
7) `solve_local` の index 整列ロジック改善
8) テスト追加・整備
9) ~~未解決~~ **対応済み** Section 11：`ProjectedOperator::apply` の `hx` の index 構造を入力 `x` と同じ空間に揃える（重複ID・余分IDを出さない）

---

## 10. 変更対象ファイル（目安）

- `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
  - validation 呼び出し
  - `solve_local` の整列／エラー改善
- `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs`
  - bra/ket 方向の修正
- `crates/tensor4all-treetn/src/linsolve/common/projected_operator.rs`
  - 必要なら外部/縮約の扱いの確認
  - **Section 11**：`apply` が返す `hx` の index を入力 `x` と同一空間に揃える（重複ID・余分IDを出さない）
- `crates/tensor4all-treetn/src/linsolve/square/local_linop.rs`
  - **Section 11**：`ProjectedOperator::apply` 修正に応じた `hx` 整列まわりの見直し（必要なら）
- `crates/tensor4all-treetn/examples/test_linsolve_mpo_identity.rs`
  - 追加の観測/再現補助（必要なら）

---

## 11. ~~未解決~~ **対応済み**：`ProjectedOperator::apply` の `hx` の index 構造を入力 `x` と揃える（重複ID・余分IDを出さない）

### 実装サマリ（2025-01）

- **ProjectedOperator::apply**（`projected_operator.rs`）  
  - MPO-with-mappings 時、**unique temp indices**（`temp_in` / `temp_out`）を使用。  
    - 入力: `true_index` → `temp_in`（`internal_index` を v に載せず重複を防止）。  
    - OP: `s_in` → `temp_in`、`s_out` → `temp_out` で clone して replace。  
  - 縮約後 `temp_out` → `true_index` で戻す。  
  - 結果の **bra 境界 bond → ket** 置換、および **`v` の index 順への `permuteinds`** で整列。
- **Example**（`test_linsolve_mpo_identity`）  
  - オペレータ A 用に `create_identity_mpo_operator_only` を追加。  
    - A のテンソルは `[s_out, s_in]`（+ bonds）のみとし、**`true_site`（external）を持たない**。  
  - これにより、`apply` 出力で `true_site` の重複が発生しなくなった。

### 現象（対応前）

`test_linsolve_mpo_identity` 実行時に、`LocalLinOp::apply` 内で以下が発生する：

- 入力 `x`（region を contract した local tensor）：**7 インデックス**、7 種の ID
- `ProjectedOperator::apply` が返す `hx`：**10 インデックス**
  - **重複ID**：ある ID が複数回現れる（例：site 用 ID が 2 回ずつ）
  - **余分なID**：`x` に存在しない ID が `hx` に含まれる（例：環境由来の別 ID）

このため `x` と `hx` の index 集合が一致せず、`permuteinds`／`axpby` が使えず、GMRES の `b - A*x` も破綻する。

### 目標

**`ProjectedOperator::apply` が返す `hx` の index 構造を「入力 `x` と同じ空間」に揃える。**

- **重複IDを出さない**：`hx` の各 ID は高々 1 回だけ現れる。
- **余分なIDを出さない**：`hx` の external indices は、`x` のそれと集合として一致する（bond・site の対応も設計どおり）。

つまり、`hx` は `x` と同じ index リスト（順序は permute で合わせる想定）を持ち、`LocalLinOp::apply` 側で「index 整列」や「index mismatch エラー」に頼らずに `a₀*x + a₁*H*x` を計算できるようにする。

### 検討観点（実装は行わない）

1. **重複IDの原因**
   - `apply` 内の contract で、**site indices**（とくに MPO の input/output 両方に現れる true_index）が、replaceind 後も複数残って開いたままになっていないか。
   - `output_mapping` による `internal → true` 置換の結果、同じ true_index が異なるテンソル同士で重複して現れ、contract で縮約されずに `hx` に残っていないか。

2. **余分なIDの原因**
   - 環境テンソルが持つ **bra 側の index**（bond 以外の site など）が、`LocalLinOp` の boundary bond の ket への置換後も残り、`hx` に含まれていないか。
   - `AllowedPairs::All` により「縮約すべきでない index」が縮約されず開いたまま残り、`x` には無い ID として出ていないか。

3. **修正の入れどころ**
   - `ProjectedOperator::apply`：contract の入力テンソル構成、replaceind の適用順・対象、および **返すテンソルの external indices が `x` と一致する**ことを保証するロジック。
   - 必要に応じて `compute_environment` や `LocalLinOp::apply` の boundary bond 置換との整合（「同じ空間」の定義は、**bond = ket 側、site = x の site_space に一致**とする）。

### 合格条件（達成済み）

- ✅ `test_linsolve_mpo_identity` が `LocalLinOp::apply` の index mismatch で落ちない。
- ✅ `hx.external_indices()` の ID 集合が `x.external_indices()` と一致し、重複・余分が無い。  
- ✅ 既存の MPS 向け linsolve テストに悪影響がない。

