# tensor4all-hdf5-ffi Refactor Design

Issue: https://github.com/tensor4all/tensor4all-rs/issues/211

## Goals

1. **Ubuntu CI安定化** - 現在のworkaroundを置き換え、根本的に安定動作
2. **メンテナンス性向上** - hdf5-metno upstreamへの追従を容易に
3. **コード品質向上** - hdf5-metnoの実績ある実装をベースに

## Scope

**このタスクのゴール:** 別リポジトリ(shinaoka/tensor4all-hdf5-ffi)でテストを通すこと。tensor4all-rsへの統合は後のステップ。

## Approach

### リポジトリ構成

```
shinaoka/tensor4all-hdf5-ffi/
├── Cargo.toml              # ワークスペース
├── .github/
│   └── workflows/
│       └── ci.yml          # Ubuntu 22.04/24.04 マトリックス
├── hdf5/                   # hdf5-metnoのhdf5クレートベース
│   ├── Cargo.toml
│   ├── src/
│   └── tests/
├── hdf5-types/             # hdf5-metnoのhdf5-typesベース
│   ├── Cargo.toml
│   └── src/
└── README.md
```

### 依存関係

- `hdf5-metno-sys` を外部依存として使用（自前実装しない）
- `hdf5-types` は hdf5-metno からコピー

### 削除する機能

- MPI対応 (`mpio` feature)
- Filterプラグイン (H5Z、圧縮)
- hdf5-derive（マクロ）
- hdf5-src（ソースビルド）

### 維持する機能

上記以外のhdf5-metnoの構造・機能はすべて維持：
- High-level API (File, Group, Dataset, Attribute, Dataspace, Datatype)
- Handle管理、エラー処理、sync
- hdf5-types全体
- テストコード（削除機能関連は除外）

### runtime-loading（dlopen）対応

hdf5-metnoはビルド時リンクのみなので、runtime-loading機能を追加。

```
hdf5/src/
├── sys/                    # FFI抽象化レイヤー
│   ├── mod.rs              # feature flagで切り替え
│   ├── link.rs             # hdf5-metno-sysをre-export
│   └── runtime.rs          # libloadingでdlopen
```

切り替え方式:
```rust
#[cfg(not(feature = "runtime-loading"))]
pub use link::*;

#[cfg(feature = "runtime-loading")]
pub use runtime::*;
```

### CI構成

```yaml
strategy:
  matrix:
    os: [ubuntu-22.04, ubuntu-24.04]
    feature: [default, runtime-loading]
```

### テスト戦略

1. hdf5-metnoの既存テストをベース（削除機能関連は除外）
2. 以前失敗したテストケースを追加（回帰防止）
3. Ubuntu 22.04/24.04 両方でパス必須
4. link / runtime-loading 両モードでパス必須

## Implementation Steps

### Phase 1: リポジトリ準備
1. shinaoka/tensor4all-hdf5-ffi リポジトリ作成
2. hdf5-metno から `hdf5` と `hdf5-types` をコピー
3. 不要クレート削除（hdf5-derive, hdf5-src）
4. Cargo.toml調整（hdf5-metno-sysを依存として設定）

### Phase 2: 機能削除
5. MPI関連コード削除（`mpio` feature）
6. Filter関連削除（H5Z、圧縮プラグイン）
7. 関連テスト削除/調整
8. `cargo test` がローカルで通ることを確認

### Phase 3: runtime-loading追加
9. `sys/` モジュール追加（link/runtime切り替え）
10. `hdf5_sys::*` 参照を `crate::sys::*` に変更
11. runtime.rs 実装（現tensor4all-hdf5-ffiベース）
12. 両モードでテスト通過確認

### Phase 4: CI整備
13. GitHub Actions workflow追加
14. Ubuntu 22.04/24.04 マトリックステスト
15. 以前失敗したテストケース追加
16. CI全パス確認

## Success Criteria

- Ubuntu 22.04/24.04 両方でCIパス
- link / runtime-loading 両モードでテストパス
- hdf5-metnoの既存テスト（削除機能以外）がパス

## Related

- PR #210: 現在のworkaround
- https://github.com/shinaoka/hdf5-rust/pull/2: hdf5-metno Ubuntu 24.04テスト
- https://github.com/aldanber/hdf5-rust (hdf5-metno)
