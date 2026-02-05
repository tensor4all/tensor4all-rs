# Restart GMRES with Truncation 実装プラン

## 概要

テンソルトレイン（MPS/MPO）表現を用いた線形方程式 `Ax = b` を解くためのrestart GMRES法を実装する。積極的なtruncationを行いながらも収束を維持できる手法。

## 動機

現在の `gmres_with_truncation` は単一のGMRESサイクル内でtruncationを適用する。これによりKrylov基底の正規直交性が失われ、解の精度が劣化する可能性がある。restart版では：
- 各GMRESサイクル内で積極的なtruncation（小さいボンド次元）が可能
- リスタート時に真の残差を再計算することで、truncation誤差を「リセット」
- 外側の反復により徐々に収束

## アルゴリズム

```
入力: apply_a, b, x0（初期推定）, options, truncate
出力: x（近似解）

for outer_iter in 0..max_outer_iters:
    # 真の残差を計算
    r = b - apply_a(x0)
    r = truncate(r)

    residual_norm = ||r|| / ||b||
    if residual_norm < rtol:
        return x0  # 収束

    # Ax' = r を gmres_with_truncation で解く
    # （内側GMRESは小さいmax_iter、積極的なtruncation）
    # 原則: 内側GMRESの初期推定は 0（補正量を解く）
    x' = gmres_with_truncation(apply_a, r, x'_0=0, inner_options, truncate)

    # 解を更新
    x0 = truncate(x0 + x')

return x0
```

## プラン検証メモ（懸念点と提案）

この設計は「truncateが非線形で、内側GMRESの推定残差が信用できない」状況に対して、外側で **真の残差** を毎回再計算して収束判定するため、方向性として妥当。

ただし、実装をスムーズにし、混乱を避けるために以下を明確化しておくと良い。

- **`truncate` の型の整合性**
    - 既存の `gmres_with_truncation` は `Tr: Fn(&mut T) -> Result<()>`（in-place）なので、本APIも同じ型に揃えると使いやすい。
    - `Fn(&T) -> Result<T>`（out-of-place）でも設計可能だが、余計なclone/アロケーションが増えやすい。
- **“restart” の用語の混乱回避**
    - 既存 `gmres` / `gmres_with_truncation` には `max_restarts` があり、これは通常の GMRES(m) のリスタート（Krylov基底の作り直し）。
    - 本プランの外側ループは「補正方程式を繰り返し解く外側反復（inexact/flexibleに近い）」なので、README/Docに「既存restartとは別物」と明記するか、関数名・オプション名で区別する。
- **内側GMRESの品質ノブ（停滞対策）**
    - 外側が真の残差で止まるのは強いが、内側が粗すぎると外側が停滞する。
    - `min_reduction`（外側残差が一定率以上下がらなければ打ち切り/警告）を初期から入れる価値が高い。
    - 追加ノブ候補：`inner_rtol`（内側の目標精度。外側残差に対する相対目標でも良い）。
- **`GmresResult.iterations` の意味**
    - `iterations` を「内側合計反復回数」にするのか「外側回数」にするのかを仕様として決めておく。
    - 必要なら `outer_iterations` を追加する。

## 主要な設計決定

### 1. 関数シグネチャ

```rust
pub fn restart_gmres_with_truncation<T, F, Tr>(
    apply_a: F,
    b: &T,
    x0: Option<&T>,
    options: RestartGmresOptions,
    truncate: Tr,
) -> Result<GmresResult<T>>
where
    T: TensorLike + Clone,
    F: Fn(&T) -> Result<T>,
    Tr: Fn(&mut T) -> Result<()>,
```

補足：`truncate` は既存の `gmres_with_truncation` と同じく in-place を想定する。

### 2. オプション構造体

```rust
pub struct RestartGmresOptions {
    /// 外側リスタート反復の最大回数
    pub max_outer_iters: usize,  // デフォルト: 20

    /// 収束判定の相対許容誤差（真の残差に基づく）
    pub rtol: f64,  // デフォルト: 1e-10

    /// 内側GMRESサイクルあたりの最大反復数
    pub inner_max_iter: usize,  // デフォルト: 10

    /// 内側GMRES内でのリスタート回数（通常0または1）
    pub inner_max_restarts: usize,  // デフォルト: 0

    /// 停滞検出（外側残差がこの率以上減らなければ停滞とみなす）
    /// 例: 0.99 なら「1%未満しか減らない状態」が続くと停滞
    pub min_reduction: Option<f64>,

    /// 内側GMRESの目標精度（必要なら）
    /// 外側残差に対する相対目標として解釈しても良い
    pub inner_rtol: Option<f64>,

    /// 詳細出力
    pub verbose: bool,  // デフォルト: false
}
```

### 3. 収束判定

各外側反復で計算される**真の残差** `||b - A*x|| / ||b||` に基づいて収束を判定する。内側GMRESの残差は（truncationにより不正確な可能性があるため）使用しない。

### 4. 初期推定の扱い

- `x0 = None` の場合：ゼロベクトルから開始（`b.scale(0.0)` などを使用）
- `x0 = Some(x)` の場合：与えられた初期推定から開始

補足：内側で解く補正 `x'` の初期推定は **原則0**（`x'_0 = 0`）とする。
（`x'_0 = r` のようなヒューリスティックもあり得るが、デフォルトにはしない。）

### 5. 内側GMRESの設定

内側GMRESは以下のように設定：
- 小さい `max_iter`（例：5-20）でKrylov基底のサイズを制限
- `max_restarts = 0`（内側GMRESではリスタートしない、外側ループに任せる）
- 積極的なtruncation（小さいボンド次元）を適用

## 実装ステップ

### ステップ1: オプション構造体の追加

[krylov.rs](../crates/tensor4all-core/src/krylov.rs) に `RestartGmresOptions` を追加：
- 上記のフィールドを持つ構造体を定義
- `Default` トレイトを実装
- ビルダーメソッドを追加（`.with_max_outer_iters()` など）

### ステップ2: `restart_gmres_with_truncation` の実装

場所：[krylov.rs](../crates/tensor4all-core/src/krylov.rs)（`gmres_with_truncation` の後）

実装詳細：
1. 初期推定を処理（Noneの場合はゼロベクトルを作成）
2. 相対許容誤差のため初期残差ノルム `||b||` を計算
3. 外側ループ：
   - 残差 `r = b - A*x0` を計算しtruncate
   - 収束判定：`||r|| / ||b|| < rtol`
   - RestartGmresOptionsから内側GmresOptionsを作成
    - `gmres_with_truncation(apply_a, &r, x'_0=0, inner_opts, truncate)` を呼び出し
   - 解を更新：`x0 = truncate(x0 + x')`
4. 最終解、総反復数、残差ノルム、収束フラグを含む `GmresResult` を返す

### ステップ3: ゼロベクトルのヘルパー追加

`b` と同じ構造を持つゼロベクトルを作成する方法が必要：
- 案A：`b.scale(AnyScalar::zero())` - シンプルだが問題が生じる可能性
- 案B：`TensorLike::zeros_like(&self) -> Self` メソッドを追加
- 案C：内側GMRESにゼロ初期推定として `None` を渡す

推奨：最初は案A（`b.scale(0.0)`）を使用。問題が生じた場合は `zeros_like` メソッドを追加。

補足：ここで必要なのは「外側の初期推定 `x0=None` のときのゼロベクトル」と「内側補正 `x'_0=0`」。いずれも原則ゼロで良い。

### ステップ4: ユニットテストの追加

`krylov.rs` にテストを追加：

1. **基本収束テスト**：単純な対角系、収束を確認
2. **比較テスト**：小さい系でリスタートなしGMRESと比較
3. **ボンド次元制御テスト**：積極的なtruncationでの解の品質を確認
4. **反復回数テスト**：外側反復が正しく追跡されることを確認

### ステップ5: MPS/MPO統合テストの追加

新規ファイル [test_restart_gmres_mps.rs](../crates/tensor4all-itensorlike/examples/test_restart_gmres_mps.rs) を作成：

1. 既存の `test_gmres_mps.rs` と同じ系でテスト
2. `gmres_with_truncation` と `restart_gmres_with_truncation` の収束挙動を比較
3. リスタート版がより小さいボンド次元で動作することを確認

## API変更の要約

`tensor4all-core` に追加される新しい公開項目：
- `RestartGmresOptions` 構造体
- `restart_gmres_with_truncation` 関数

既存APIへの変更なし。

## テスト戦略

1. **ユニットテスト**：密なテンソルでアルゴリズムの正確性を検証
2. **統合テスト**：様々な演算子でMPS/MPOの挙動を検証
3. **比較テスト**：リスタート版と非リスタート版の残差減少を比較

## 未解決の質問

1. **停滞検出**：停滞（外側ループで進展なし）を検出して対処すべきか？
   - 提案：停滞検出のため `min_reduction` オプションを追加

2. **適応的内側反復**：収束率に基づいて `inner_max_iter` を適応させるべきか？
   - 初期決定：シンプルに保つ。ユーザーが手動で調整可能。

3. **反復回数の報告**：`GmresResult.iterations` は内側合計か外側回数か？
    - 提案：内側合計を `iterations` にし、外側回数は別フィールド（例：`outer_iterations`）で返す。

4. **前処理**：前処理（プリコンディショナー）をサポートすべきか？
   - 初期決定：最初の実装には含めない。後で追加可能。
