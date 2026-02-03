# Tensor4all.jl 独立リポジトリ化設計

## 概要

Tensor4all.jl を tensor4all-rs から独立したリポジトリに分離し、GitHub URL から直接 `Pkg.add` / `Pkg.dev` できるようにする。

## 背景

- 現状: `tensor4all-rs/julia/Tensor4all.jl/` にネストされており、`Pkg.add(url="...", subdir="julia/Tensor4all.jl")` でインストール可能だが、ビルドには親ディレクトリの Rust コードが必要
- Julia/Python は Tier-2 ターゲットとして、Rust と独立したリリースサイクルで管理する方針

## 設計

### リポジトリ構造

**変更前**:
```
tensor4all-rs/
├── crates/
├── julia/Tensor4all.jl/  # ネスト
└── python/tensor4all/
```

**変更後**:
```
github.com/tensor4all/
├── tensor4all-rs/        # Rust (既存)
│   ├── crates/
│   └── python/tensor4all/
│
└── Tensor4all.jl/        # Julia (新規リポジトリ)
    ├── Project.toml
    ├── src/
    ├── deps/
    │   ├── build.jl
    │   └── libtensor4all_capi.{so,dylib,dll}
    └── test/
```

### 共有ライブラリの配置

`deps/` ディレクトリに配置:
- パッケージごとに独立（衝突リスクなし）
- `Pkg.rm()` で自動削除
- sparse-ir-rs/SparseIR.jl と同じパターン

### Rust ソース取得ロジック

優先順位:
1. 環境変数 `TENSOR4ALL_RS_PATH` が設定されていればそれを使用
2. 兄弟ディレクトリ `../tensor4all-rs/` が存在すればそれを使用
3. どちらもなければ GitHub から clone

```julia
const LOCAL_RUST_DIR = get(ENV, "TENSOR4ALL_RS_PATH", nothing)
const SIBLING_RUST_DIR = joinpath(dirname(PACKAGE_DIR), "tensor4all-rs")

function find_rust_source()
    if LOCAL_RUST_DIR !== nothing && isdir(LOCAL_RUST_DIR)
        return LOCAL_RUST_DIR
    elseif isdir(SIBLING_RUST_DIR) && isfile(joinpath(SIBLING_RUST_DIR, "Cargo.toml"))
        return SIBLING_RUST_DIR
    else
        return nothing  # → GitHub から clone
    end
end
```

### バージョン指定

build.jl 内に定数として記載:
```julia
const TENSOR4ALL_RS_VERSION = "v0.2.0"  # タグまたはコミットハッシュ
const TENSOR4ALL_RS_REPO = "https://github.com/tensor4all/tensor4all-rs.git"
```

### cargo の取得

RustToolChain.jl を使用してユーザー環境に Rust がなくても動作:
```julia
using RustToolChain: cargo
run(`$(cargo()) build -p tensor4all-capi --release`)
```

### build.jl 実装

```julia
using RustToolChain: cargo
using Libdl

const DEPS_DIR = @__DIR__
const PACKAGE_DIR = dirname(DEPS_DIR)
const TENSOR4ALL_RS_VERSION = "v0.2.0"
const TENSOR4ALL_RS_REPO = "https://github.com/tensor4all/tensor4all-rs.git"
const LIB_NAME = "libtensor4all_capi." * Libdl.dlext

const LOCAL_RUST_DIR = get(ENV, "TENSOR4ALL_RS_PATH", nothing)
const SIBLING_RUST_DIR = joinpath(dirname(PACKAGE_DIR), "tensor4all-rs")

function find_rust_source()
    if LOCAL_RUST_DIR !== nothing && isdir(LOCAL_RUST_DIR)
        return LOCAL_RUST_DIR
    elseif isdir(SIBLING_RUST_DIR) && isfile(joinpath(SIBLING_RUST_DIR, "Cargo.toml"))
        return SIBLING_RUST_DIR
    else
        return nothing
    end
end

function build()
    rust_dir = find_rust_source()

    if rust_dir === nothing
        # GitHub から clone
        rust_dir = mktempdir()
        run(`git clone --depth 1 --branch $TENSOR4ALL_RS_VERSION $TENSOR4ALL_RS_REPO $rust_dir`)
        cleanup = true
    else
        println("Using local Rust source: $rust_dir")
        cleanup = false
    end

    # ビルド
    cd(rust_dir) do
        run(`$(cargo()) build -p tensor4all-capi --release`)
    end

    # 共有ライブラリをコピー
    src = joinpath(rust_dir, "target", "release", LIB_NAME)
    dst = joinpath(DEPS_DIR, LIB_NAME)
    cp(src, dst; force=true)
    println("Library installed to: $dst")

    # 一時ディレクトリの場合は削除
    if cleanup
        rm(rust_dir; recursive=true)
    end
end

build()
```

### ライブラリロード

```julia
module Tensor4all

using Libdl

function get_library_path()
    deps_lib = joinpath(@__DIR__, "..", "deps", "libtensor4all_capi." * Libdl.dlext)

    if isfile(deps_lib)
        return deps_lib
    else
        error("""
            libtensor4all_capi not found.
            Run: using Pkg; Pkg.build("Tensor4all")
            """)
    end
end

const LIBPATH = get_library_path()

end
```

## 使用シナリオ

| シナリオ | コマンド | Rust ソース |
|---------|---------|-------------|
| GitHub から add | `Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl")` | GitHub clone |
| 兄弟ディレクトリで開発 | `Pkg.dev(path="~/projects/Tensor4all.jl")` | `~/projects/tensor4all-rs/` |
| 任意パスで開発 | `ENV["TENSOR4ALL_RS_PATH"]="/path/to/tensor4all-rs"; Pkg.build("Tensor4all")` | 指定パス |

## 移行手順

1. **新リポジトリ作成**: GitHub で `tensor4all/Tensor4all.jl` を作成
2. **ファイル移動**:
   - `tensor4all-rs/julia/Tensor4all.jl/*` → `Tensor4all.jl/`
   - Project.toml に RustToolChain 依存を追加
   - deps/build.jl を新ロジックに書き換え
3. **tensor4all-rs の整理**:
   - `julia/` ディレクトリを削除
   - README.md のインストール手順を更新
4. **動作確認**: add mode / dev mode の両方をテスト

## 将来の拡張

crates.io に tensor4all-capi を公開後:
- GitHub clone の代わりに crates.io から取得
- ビルド時間の短縮（依存クレートのキャッシュ活用）

## 参考

- [sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs) / [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl)
- [RustToolChain.jl](https://github.com/AtelierArith/RustToolChain.jl)
