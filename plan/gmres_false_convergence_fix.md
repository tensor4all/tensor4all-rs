# Issue #207 解決プラン: GMRES with Truncation の改善

## 概要

`gmres_with_truncation` の false convergence 問題を根本的に解決し、実行ごとの結果変動の原因を調査する。

## 問題の整理

### 問題 1: False Convergence
- `gmres_with_truncation` が Hessenberg 残差 ~1e-16 で収束と報告するが、真の残差は ~0.4
- 原因: SVD truncation が Krylov 基底の直交性を破壊し、Hessenberg 行列が不正確になる
- 既存の iterative reorthogonalization では完全には防げない

### 問題 2: 実行ごとの結果変動
- 同じコードでも実行ごとに結果が異なる
- `DynIndex::new_dyn()` が `ThreadRng` で ID を生成 (`index.rs:337-339`)
- ただし Issue #192 の調査で、ランダム ID 自体は計算結果に影響しないことが判明済み
- 真の原因の特定が必要

---

## フェーズ 1: 非決定性の原因調査

### 仮説

MPS/MPO テストで `ContractOptions::fit()` を使用しており、fit アルゴリズム内部で `sim_internal_inds()` が新しい ID を生成する (`contraction.rs:567-568`)。また、Rust の `std::collections::HashMap` はプロセスごとにランダムなハッシュ seed を使うため、キーが同じでもイテレーション順序が実行ごとに変わり得る。これにより、縮約・加算の累積順序が微妙に異なり、浮動小数点丸め誤差が増幅して結果が変動する可能性がある。

加えて、並列化（Rayon や BLAS 由来の並列 reduction）がある場合、実行スケジューリングにより加算順序が変わり、同様の非決定性が出る可能性がある。

### 調査手順

0. まずスレッド数を固定して再現性を確認する（並列 reduction 要因の切り分け）
    - `RAYON_NUM_THREADS=1`（必要なら BLAS 系のスレッドも 1 に固定）
1. `test_issue207_restart_gmres` を 5 回実行し、数値出力（真の残差、反復回数など）を記録して変動を定量化
2. `ContractOptions::fit()` を `ContractOptions::zipup()` に一時的に変更して比較
3. fit でのみ変動する場合、縮約順序（`HashMap` 等）の非決定性による丸め誤差と結論し、必要なら順序を固定する最小実験を行う
    - 例: キー集合をソートしてから処理する／`BTreeMap` や順序安定な map を局所的に使う

### 対象ファイル

- `crates/tensor4all-treetn/src/treetn/contraction.rs` - `sim_internal_inds()`
- `crates/tensor4all-core/src/defaults/index.rs` - `generate_id()` (L337)
- `crates/tensor4all-itensorlike/examples/test_issue207_restart_gmres.rs`

---

## フェーズ 2: `gmres_with_truncation` に true residual チェックを追加

### 設計方針

**アプローチ C+B**: 収束判定時の必須チェック + オプションの定期チェック

最小限のコストで false convergence を防ぐ。Hessenberg 残差が rtol 以下になった時点で、`||b - A*x|| / ||b||` を計算して真に収束しているか確認する。

補足: truncation を伴う実装では Arnoldi 直交性が破れ得るため、Hessenberg 残差だけでの収束判定は信頼できない。したがって「収束と返すなら、真の残差（または truncate 後の残差）でも rtol 未満」を保証するのが目的。

### Step 2a: `GmresOptions` に `check_true_residual` フィールドを追加

**ファイル**: `crates/tensor4all-core/src/krylov.rs` (L35-64)

```rust
pub struct GmresOptions {
    pub max_iter: usize,
    pub rtol: f64,
    pub max_restarts: usize,
    pub verbose: bool,
    /// true の場合、収束判定時に真の残差 ||b - A*x|| / ||b|| を計算して検証する。
    /// truncation による Krylov 基底の直交性喪失に起因する false convergence を防止する。
    /// 追加コスト: 収束検出時に apply_a を 1 回余分に呼ぶ。
    /// デフォルト: false
    pub check_true_residual: bool,
}
```

- `Default` 実装: `check_true_residual: false`（後方互換性のため）

### Step 2b: `gmres_with_truncation` の収束判定ロジックを修正

**ファイル**: `crates/tensor4all-core/src/krylov.rs` (L303-521)

修正箇所は 3 箇所:

#### (1) 内部ループの収束判定（L473-482）

```rust
if rel_res < options.rtol {
    let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
    x = update_solution_truncated(&x, &v_basis[..=j], &y, &truncate)?;

    if options.check_true_residual {
        let ax_check = apply_a(&x)?;
        let r_check = b.axpby(AnyScalar::F64(1.0), &ax_check, AnyScalar::F64(-1.0))?;
        let true_rel_res = r_check.norm() / b_norm;
        if true_rel_res < options.rtol {
            // 真に収束
            return Ok(GmresResult { residual_norm: true_rel_res, converged: true, .. });
        }
        // False convergence 検出 → restart cycle へ break
        break;
    } else {
        return Ok(GmresResult { residual_norm: rel_res, converged: true, .. });
    }
}
```

#### (2) Lucky breakdown ケース（L492-503）

同様に true residual チェックを追加。ただし lucky breakdown は既に true residual を計算しているため変更不要。

#### (3) Restart cycle 開始時の収束判定（L342-349）

ここでは既に `r = b - A*x` を計算しているため、変更不要（既に true residual ベース）。

### Step 2c: `restart_gmres_with_truncation` の内部 GMRES で有効化

**ファイル**: `crates/tensor4all-core/src/krylov.rs` (L718-723)

```rust
let inner_options = GmresOptions {
    max_iter: options.inner_max_iter,
    rtol: options.inner_rtol.unwrap_or(0.1),
    max_restarts: options.inner_max_restarts + 1,
    verbose: options.verbose,
    check_true_residual: true,  // restart 文脈では常に有効
};
```

### 設計判断: True residual 計算時の truncate

- `restart_gmres_with_truncation` と一貫して、residual `r = b - A*x` の計算後に **truncate を適用してから** ノルムを評価する
- これにより restart wrapper と内部チェックの判定基準が統一される

注意: ここで評価する量は厳密な意味での「非truncate の真の残差」ではなく、「表現制約（truncate）込みで評価した residual」となる。ドキュメント上は用語を揃え、必要ならログ出力などでは `checked residual` のように区別して混乱を避ける。

---

## フェーズ 3: テスト

### Step 3a: ユニットテスト追加

**ファイル**: `crates/tensor4all-core/src/krylov.rs` (L946-1542)

`test_gmres_with_truncation_detects_false_convergence`:
- 直交性を壊す truncation 関数を用意
- 主眼は `check_true_residual: true` の安全性保証: 「`converged=true` を返すなら、チェックした residual が rtol 未満」を必ず満たす
- `check_true_residual: false` 側は挙動が問題設定に依存して不安定になりやすいため、厳密に「必ず false convergence する」前提のテストにはしない（必要なら弱いアサーションに留める）

### Step 3b: Issue #207 統合テスト更新

**ファイル**: `crates/tensor4all-itensorlike/examples/test_issue207_restart_gmres.rs`

- `gmres_with_truncation` + `check_true_residual: true` のテストケースを追加
- 3 つの方式を比較:
  1. `gmres_with_truncation`（チェックなし）
  2. `gmres_with_truncation`（true residual チェックあり）
  3. `restart_gmres_with_truncation`

### Step 3c: 再現性テスト

- 同一プロセス内で 5 回実行し、結果の変動を記録
- `zipup` と `fit` 両方で比較

---

## フェーズ 4: 検証

```bash
cargo fmt --all
cargo clippy --workspace
cargo test --workspace
cargo run -p tensor4all-itensorlike --example test_issue207_restart_gmres --release
```

---

## パフォーマンスへの影響

| シナリオ | 追加コスト |
|---------|-----------|
| 収束成功（false conv なし） | `apply_a` 1 回（収束確認時のみ） |
| False convergence 検出 | `apply_a` 1 回 + restart cycle |
| `check_true_residual: false` | コスト 0（現状と同じ） |

`apply_a` が MPS/MPO の場合は計算コストが大きいが、false convergence を黙って受け入れるよりは遥かに良い。

---

## 修正対象ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `crates/tensor4all-core/src/krylov.rs` | `GmresOptions` 拡張、`gmres_with_truncation` 修正、`restart_gmres_with_truncation` 更新、ユニットテスト追加 |
| `crates/tensor4all-itensorlike/examples/test_issue207_restart_gmres.rs` | 統合テスト追加 |
