#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
generated="$tmp/tensor4all_capi.h"
committed="$root/crates/tensor4all-capi/include/tensor4all_capi.h"

python3 "$root/scripts/test-generate-capi-header.py"
python3 "$root/scripts/generate-capi-header.py" --output "$generated"

grep -Fq 'Generated with cbindgen:0.29.2' "$generated"
diff -u "$committed" "$generated"

cat >"$tmp/header.c" <<'EOF'
#include "crates/tensor4all-capi/include/tensor4all_capi.h"
int main(void) { return 0; }
EOF
cat >"$tmp/header.cpp" <<'EOF'
#include "crates/tensor4all-capi/include/tensor4all_capi.h"
int main() { return 0; }
EOF

"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -I"$root" -fsyntax-only "$tmp/header.c"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$root" -fsyntax-only "$tmp/header.cpp"
