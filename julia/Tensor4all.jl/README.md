# Tensor4all.jl

Julia wrapper for the tensor4all Rust library.

## セットアップ手順

このJuliaパッケージを使用するには、まずRustライブラリをビルドする必要があります。

### 1. Rustのインストール

Rustがインストールされていない場合、以下のコマンドでインストールできます：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

インストール後、新しいターミナルを開くか、以下のコマンドで環境変数を読み込みます：

```bash
source ~/.cargo/env
```

### 2. Rustライブラリのビルド

#### 方法A: 直接ビルドスクリプトを実行（activate不要）

Juliaパッケージのディレクトリに移動して、ビルドスクリプトを直接実行します：

```bash
cd /Users/ken/git_for_collab4/tensor4all-rs/julia/Tensor4all.jl
julia deps/build.jl
```

この方法では、Pkgモードでactivateする必要はありません。

#### 方法B: Pkgモードでactivateしてビルド

Julia REPLでパッケージを開発モードで使用する場合：

```julia
using Pkg

# パッケージディレクトリに移動（またはパスを指定）
Pkg.activate(".")  # 現在のディレクトリをアクティベート
# または絶対パスで：
# Pkg.activate("/Users/ken/git_for_collab4/tensor4all-rs/julia/Tensor4all.jl")

# ビルドを実行
Pkg.build("Tensor4all")
```

または、別の環境から開発モードで追加する場合：

```julia
using Pkg
Pkg.develop(path="/Users/ken/git_for_collab4/tensor4all-rs/julia/Tensor4all.jl")
Pkg.build("Tensor4all")
```

### 3. テストの実行

ビルドが完了したら、テストを実行できます：

#### 方法A: 直接実行（activate不要）

```bash
cd /Users/ken/git_for_collab4/tensor4all-rs/julia/Tensor4all.jl
julia --project=. test/runtests.jl
```

#### 方法B: Pkgモードで実行

```julia
using Pkg
Pkg.activate(".")  # または Pkg.activate("/path/to/Tensor4all.jl")
Pkg.test("Tensor4all")
```

## トラブルシューティング

### エラー: "tensor4all-capi library not found"

このエラーは、`deps/`ディレクトリにライブラリファイルが存在しないことを示しています。

解決方法：
1. Rustがインストールされていることを確認：`cargo --version`
2. ビルドスクリプトを実行：`julia deps/build.jl`
3. `deps/libtensor4all_capi.dylib`（macOSの場合）が存在することを確認

### エラー: "Could not find cargo"

Rustがインストールされていないか、PATHに含まれていません。

解決方法：
1. Rustをインストール（上記の手順を参照）
2. 新しいターミナルを開くか、`source ~/.cargo/env`を実行

## パッケージの使用

### 開発モードで使用する場合

パッケージを開発・テストする場合、まずactivateします：

```julia
using Pkg
Pkg.activate(".")  # パッケージディレクトリで実行
using Tensor4all
```

### 通常のパッケージとして使用する場合

別のプロジェクトから使用する場合：

```julia
using Pkg
Pkg.add(url="/path/to/Tensor4all.jl")  # または Pkg.develop(path="...")
using Tensor4all
```

## 使用方法

```julia
using Tensor4all

# Indexの作成
i = Index(5)
j = Index(3; tags="Site,n=1")

# Tensorの作成
data = rand(5, 3)
t = Tensor([i, j], data)

# データの取得
retrieved = data(t)
```

詳細な使用方法については、各モジュールのドキュメントを参照してください。
