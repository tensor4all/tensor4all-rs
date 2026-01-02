# tensor4all-rs ライセンス分析レポート

## 現状の確認

### 1. ライセンス設定

- **Cargo.toml**: `license = "MIT OR Apache-2.0"` ✅
- **LICENSE ファイル**: **存在しない** ❌
- **コード内のライセンスヘッダー**: **存在しない** ⚠️
- **README での言及**: ITensors.jl への言及あり ✅

### 2. 実装の独立性

コードを確認した結果：
- ✅ **実装は独自**: ITensors.jl のコードを直接移植したものではない
- ✅ **API 設計の参考**: ITensors.jl の API を参考にしているが、実装は Rust で独自に書かれている
- ✅ **関数名の借用**: 一部の関数名（例: `sim`, `replaceinds`）は ITensors.jl から借用
- ✅ **アルゴリズムの概念**: 数学的概念（SVD、QR分解、テンソル縮約など）は一般的なもの

## 問題点と推奨事項

### ❌ 必須対応項目

#### 1. LICENSE ファイルの追加

**問題**: `MIT OR Apache-2.0` と宣言しているが、LICENSE ファイルが存在しない

**対応**: 以下の2つのファイルを追加する必要があります：

**LICENSE-MIT**:
```
MIT License

Copyright (c) 2024 tensor4all collaboration

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**LICENSE-APACHE**:
```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   [Apache License 2.0 の全文をここに含める]

   Copyright 2024 tensor4all collaboration

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### ⚠️ 推奨対応項目

#### 2. README へのクレジット強化

**現状**: ITensors.jl への言及はあるが、ライセンス情報が不足

**推奨**: README に以下のセクションを追加：

```markdown
## Acknowledgments

This implementation is inspired by and designed to be compatible with:

- **ITensors.jl** (https://github.com/ITensor/ITensors.jl)
  - License: Apache License 2.0
  - Copyright: The Simons Foundation, Inc.
  - We have borrowed API design concepts and function names, but the implementation is independently written in Rust.

- **QSpace v4** (MATLAB/C++)
  - License: Apache License 2.0
  - Copyright: Andreas Weichselbaum
  - We have borrowed concepts for quantum number symmetries and block-sparse tensor organization.

## License

This project is dual-licensed under either:

- **MIT License** (see [LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** (see [LICENSE-APACHE](LICENSE-APACHE))

at your option.
```

#### 3. コード内のコメント強化（オプション）

主要なファイルに、元の実装への言及を追加：

```rust
//! Tensor network implementation for Rust
//!
//! This crate is inspired by ITensors.jl (https://github.com/ITensor/ITensors.jl)
//! and QSpace v4, but the implementation is independently written in Rust.
//! API design and function names are borrowed for compatibility, but the
//! actual implementation is original.
```

#### 4. NOTICE ファイルの追加（オプション、推奨）

Apache License 2.0 を使用する場合、NOTICE ファイルを追加することで、元の実装への適切なクレジットを提供できます：

**NOTICE**:
```
tensor4all-rs
Copyright 2024 tensor4all collaboration

This product includes software inspired by:

ITensors.jl
Copyright 2021 The Simons Foundation, Inc.
Licensed under Apache License 2.0
https://github.com/ITensor/ITensors.jl

QSpace v4
Copyright 2024 Andreas Weichselbaum
Licensed under Apache License 2.0
```

## 法的リスク評価

### ✅ 低リスクの理由

1. **実装が独自**: ITensors.jl のコードを直接移植したものではない
2. **アルゴリズムは一般的**: SVD、QR分解、テンソル縮約は標準的な数学的アルゴリズム
3. **API 設計の借用**: API 設計の概念は著作権保護されない（アイデアは保護されない）
4. **関数名の借用**: 関数名は短い表現であり、著作権保護の対象外の可能性が高い

### ⚠️ 注意点

1. **明確な言及**: 元の実装への言及を明確にすることで、透明性を確保
2. **ライセンスファイル**: `MIT OR Apache-2.0` と宣言している以上、両方のライセンスファイルが必要
3. **商用利用**: 商用利用を予定する場合は、法的レビューを推奨

## 推奨される対応手順

### 即座に対応（必須）

1. ✅ `LICENSE-MIT` ファイルを作成
2. ✅ `LICENSE-APACHE` ファイルを作成（Apache License 2.0 の全文を含む）

### 短期対応（推奨）

3. ✅ README に Acknowledgments セクションを追加
4. ✅ `NOTICE` ファイルを作成（Apache License 2.0 を使用する場合）

### 長期対応（オプション）

5. ⚠️ 主要なファイルにライセンスヘッダーを追加
6. ⚠️ コントリビューションガイドラインにライセンス方針を明記

## 結論

### 現状の評価

- **法的リスク**: **低** ✅
  - 実装は独自であり、直接的なコードの移植ではない
  - アルゴリズムは一般的な数学的概念

- **コンプライアンス**: **要改善** ⚠️
  - LICENSE ファイルが存在しない（必須）
  - 元の実装への言及はあるが、より明確にできる（推奨）

### 推奨アクション

1. **必須**: LICENSE-MIT と LICENSE-APACHE ファイルを追加
2. **推奨**: README に Acknowledgments セクションを追加
3. **推奨**: NOTICE ファイルを追加（Apache License 2.0 の要件を満たすため）

これらの対応により、ライセンスコンプライアンスを完全に満たし、元の実装への適切なクレジットを提供できます。

---

**免責事項**: この分析は法的アドバイスではありません。具体的な法的問題については、適格な法的専門家に相談してください。

