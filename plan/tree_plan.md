# Tree Network Zip-up Contraction 実装プラン

## 概要

現在のzip-up contraction実装を、線形チェーン（TensorTrain）から任意のTree Network構造に拡張する。アルゴリズムは「葉からルートへ向かって中間テンソル（環境テンソル）を累積しながら進む」方式を採用する。

## 現状の実装状況

### 既存実装
- **TensorTrain (線形チェーン)**: `contract_zipup_itensors_like` で実装済み
  - ITensors.jlスタイルの累積R方式
  - 環境テンソルRを保持しながら順次処理
- **TreeTN (汎用ツリー)**: `contract_zipup_with` で実装済み
  - 現在は「各edgeでchild tensorをfactorizeしてparentに吸収」する方式
  - 中間テンソルを保持しない

### 課題
- TreeTNのzip-upは、各edgeで独立に処理しているため、効率が悪い可能性がある
- プランで要求される「中間テンソルを保持しながら進む」方式と異なる

## アルゴリズム設計

### 基本方針
1. **Post-order DFSでedgeを取得**: 葉からルートへ向かう順序
2. **中間テンソル（環境テンソル）を保持**: 各ノードで累積された中間テンソルを管理
3. **段階的なcontractionとfactorization**: 葉では2つのcoreをcontract、内部ノードでは中間テンソル+2つのcoreをcontract

### 詳細アルゴリズム

#### ステップ1: 前処理
1. トポロジーの検証: `same_topology()` で両ネットワークの構造が一致することを確認
2. 内部インデックスの分離: `sim_internal_inds()` で両ネットワークのbond indicesを独立化
3. ルートノードの決定: 指定されたcenterノードをルートとする
4. Post-order DFS edgeの取得: `edges_to_canonicalize_by_names(center)` で葉→ルートのedge順序を取得

#### ステップ2: 葉ノードの処理
各葉ノード（source）について：

```
入力:
  - A[source]: ネットワークAのsourceノードのテンソル
  - B[source]: ネットワークBのsourceノードのテンソル
  - destination: sourceの親ノード

処理:
  1. C_temp = contract(A[source], B[source])  // 2つのcoreをcontract
  2. (C[source], R) = factorize(C_temp, left_inds=site_indices(source) + bond_to_destination)
     // 左因子はsite indices + destinationへのbond、右因子は環境テンソルR
  3. 結果のbond indexでedgeを更新: replace_edge_bond(edge(source, destination), new_bond)
  4. 中間テンソルRをdestinationに登録（まだcontractionは取らない）
  5. C[source]を結果ネットワークのsourceノードに保存
```

**注意点:**
- `site_indices(source)` は、sourceノードの物理インデックス（bond indicesを除く）
- `bond_to_destination` は、sourceからdestinationへのbond index
- この時点では、destinationのテンソルとRのcontractionは**行わない**
 - 既存のTreeTNと同様に、edge bondの更新（`replace_edge_bond`）が必要

#### ステップ3: 内部ノードの処理
各内部ノード（source、ただし葉ではない）について：

```
入力:
  - R_accumulated: sourceから来た累積中間テンソル（既に登録済み）
  - A[source]: ネットワークAのsourceノードのテンソル
  - B[source]: ネットワークBのsourceノードのテンソル
  - destination: sourceの親ノード

処理:
  1. テンソルリストを準備: [R_accumulated..., A[source], B[source]]
  2. C_temp = contract([R_accumulated..., A[source], B[source]], AllowedPairs::All)
     // 最適な順番でcontractionが自動的に取られる
  3. (C[source], R_new) = factorize(C_temp, left_inds=site_indices(source) + bond_to_destination)
  4. 結果のbond indexでedgeを更新: replace_edge_bond(edge(source, destination), new_bond)
  5. 中間テンソルR_newをdestinationに登録（まだcontractionは取らない）
  6. C[source]を結果ネットワークのsourceノードに保存
```

**注意点:**
- `T::contract(&[tensors...], AllowedPairs::All)` は、自動的に最適な順序でcontractionを実行する
- 複数の中間テンソルは先にまとめず、**一括でcontract**する方針（順序最適化を活かす）
 - edge bondの更新（`replace_edge_bond`）が必要

#### ステップ4: ルートノードの処理
ルートノード（全ての中間テンソルが集約される）について：

```
入力:
  - R_list: ルートに接続された全ての子ノードから来た中間テンソルのリスト
  - A[root]: ネットワークAのルートノードのテンソル
  - B[root]: ネットワークBのルートノードのテンソル

処理:
  1. テンソルリストを準備: [R_list..., A[root], B[root]]
  2. C_temp = contract([R_list..., A[root], B[root]], AllowedPairs::All)
     // 最適な順番でcontractionが自動的に取られる
  3. C[root] = C_temp  // ルートではfactorizationは不要（これが最終結果）
  4. C[root]を結果ネットワークのルートノードに保存
```

**注意点:**
- ルートノードは複数の子を持つ可能性がある（星型構造など）
- 全ての中間テンソルを一度にcontractする必要がある
- ルートではfactorizationは不要（これが最終的なテンソル）

#### ステップ5: 最終結果の正規化中心の設定
全ての処理が完了した後、結果TreeTNの正規化中心を設定する：

```
処理:
  1. 結果TreeTNが構築された後、指定されたcenterノードを正規化中心として設定
  2. result.set_canonical_center(std::iter::once(center.clone()))?;
  3. これにより、結果のTreeTNは指定されたcenterを正規化中心として持つ
```

**注意点:**
- 既存の`contract_zipup_with`実装と同様に、最後に正規化中心を設定する
- `set_canonical_center`は、centerノードが結果に存在する場合のみ成功する
- 正規化中心の設定により、結果のTreeTNは正規化された状態になる
- edgeの `ortho_towards` は **centerに向くように更新**する（既存実装と整合）

## 実装詳細

### データ構造

```rust
// 中間テンソルを管理するための構造
struct IntermediateTensors {
    // 各ノードに登録された中間テンソルのリスト
    // キー: ノード名, 値: そのノードに集約される中間テンソルのリスト
    accumulated: HashMap<V, Vec<TensorDynLen>>,
}

// または、よりシンプルに
// HashMap<V, Vec<TensorDynLen>> として直接管理
```

### 関数シグネチャ（仮）

```rust
impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Tree Network用のzip-up contraction（中間テンソル累積方式）
    pub fn contract_zipup_tree_accumulated(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self> {
        // 実装
    }
}
```

### 実装ステップ

#### Phase 1: 基本構造の実装
1. **前処理部分**
   - トポロジー検証
   - 内部インデックス分離
   - Post-order DFS edge取得

2. **中間テンソル管理の実装**
   - `HashMap<V, Vec<TensorDynLen>>` で各ノードに集約される中間テンソルを管理
   - 中間テンソルの追加・取得関数

#### Phase 2: 葉ノード処理の実装
1. **葉ノードの判定**
   - Post-order DFS順で処理するため、最初に処理されるノードが葉
   - または、`graph.neighbors(node).count() == 1` で判定（ルート以外）

2. **葉ノードでのcontraction + factorization**
   - `T::contract(&[a, b], AllowedPairs::All)` で2つのcoreをcontract
   - `factorize_with()` でSVD分解
   - 左因子を結果に保存、右因子（環境テンソル）を親ノードに登録

#### Phase 3: 内部ノード処理の実装
1. **内部ノードの判定**
   - 葉でもルートでもないノード

2. **累積中間テンソルの取得**
   - そのノードに登録された中間テンソルを取得
   - 複数ある場合は、それらを先にcontractして1つにまとめる

3. **contraction + factorization**
   - `T::contract(&[r_accumulated, a, b], AllowedPairs::All)` でcontract
   - `factorize_with()` でSVD分解
   - 左因子を結果に保存、右因子を親ノードに登録

#### Phase 4: ルートノード処理の実装
1. **ルートノードでの最終処理**
   - 全ての中間テンソルを取得
   - `T::contract(&[r_list..., a_root, b_root], AllowedPairs::All)` でcontract
   - 結果をそのまま保存（factorization不要）

2. **最終結果の正規化中心の設定**
   - 結果TreeTNが構築された後、`result.set_canonical_center(std::iter::once(center.clone()))?` を呼び出す
   - centerノードが結果に存在することを確認してから設定
   - これにより、結果のTreeTNは指定されたcenterを正規化中心として持つ

#### Phase 5: エッジケースの処理
1. **単一ノードネットワーク**
   - ルート = 葉のケース
   - 直接 `contract(A[root], B[root])` を実行

2. **2ノードネットワーク**
   - ルートと葉のみ
   - 葉でfactorize、ルートで最終contract

3. **星型構造（ルートに複数の葉が接続）**
   - 各葉から中間テンソルが集約
   - ルートで全てをcontract

## 技術的な考慮事項

### 1. Contraction順序の最適化
- `T::contract(&[tensors...], AllowedPairs::All)` は自動的に最適な順序を選択する
- ただし、中間テンソルが複数ある場合の順序は明示的に制御する必要がある可能性がある

### 2. Factorizationオプション
- `FactorizeOptions` で `rtol` と `max_rank` を指定
- `CanonicalForm::Left`/`Unitary`/`LU`/`CI` の**3形態すべてに対応**
  - `Unitary`/`LU`/`CI` それぞれに対応する `FactorizeAlg` を使用
  - left factorが「site indices + bond」を保持する方針は維持

### 3. インデックス管理
- Site indicesの抽出: `external_indices()` からbond indicesを除外
- Bond indicesの識別: 隣接ノードとの共通インデックス

### 4. メモリ効率
- 中間テンソルは必要最小限の期間のみ保持
- 処理済みノードのテンソルは早期に解放

## テスト計画

### ユニットテスト
1. **単一ノード**: 2つのテンソルを直接contract
2. **2ノードチェーン**: 葉→ルートの処理
3. **3ノードチェーン**: 葉→内部→ルートの処理
4. **星型構造**: ルートに3つ以上の葉が接続
5. **分岐構造**: 内部ノードに複数の子が接続

### 統合テスト
1. **既存のzip-up実装との結果比較**
   - 線形チェーンで `contract_zipup_itensors_like` と結果が一致するか
2. **Naive contractionとの結果比較**
   - `contract_naive` で得られる結果と数値的に一致するか（truncation誤差を除く）

### パフォーマンステスト
1. **大規模ネットワークでのベンチマーク**
   - ノード数、bond dimension、物理次元を変えて計測
2. **メモリ使用量の計測**
   - 中間テンソルの保持によるメモリ増加を確認

## 既存コードとの統合

### 関数の配置
- `crates/tensor4all-treetn/src/treetn/contraction.rs` に追加
- 既存の `contract_zipup_with` は残す（後方互換性のため）
- 新しい関数名: `contract_zipup_accumulated` または `contract_zipup_tree_accumulated`

### オプションの拡張
- `ContractionOptions` に新しいメソッドを追加する可能性
- または、既存の `ContractMethod::Zipup` の実装を置き換える

## 実装の優先順位

1. **高優先度**: 基本アルゴリズムの実装（Phase 1-4）
2. **中優先度**: エッジケースの処理（Phase 5）
3. **低優先度**: パフォーマンス最適化、メモリ効率化

## 参考資料

- ITensors.jlのMPO zip-up実装
- 既存の `contract_zipup_itensors_like` 実装（線形チェーン用）
- `contract_zipup_with` 実装（現在のTreeTN用）

## 注意事項

- 実装は段階的に進める（各Phaseごとにテスト）
- 既存の実装との互換性を維持
- エラーハンドリングを適切に実装（各ステップでエラーチェック）
