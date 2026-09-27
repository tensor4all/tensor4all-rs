#!/usr/bin/env python3
"""Generate the C header using the normal-library export attributes.

cbindgen does not evaluate cfg_attr (mozilla/cbindgen#183). Resolve the one
supported condition in a temporary source tree; never rewrite repository files.
Unknown cfg_attr forms fail so new conditions require an explicit review here.
"""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


EXPECTED_VERSION = "cbindgen 0.29.2"
CONDITIONAL_ATTRIBUTE = re.compile(r"#\[\s*cfg_attr\b[^\]]*\]", re.DOTALL)
LIBRARY_EXPORT = "#[cfg_attr(not(test),unsafe(no_mangle))]"


def normal_library_source(source: str, path: str = "<source>") -> str:
    """Resolve only our test-dependent C export attribute, rejecting others."""
    def resolve(match: re.Match[str]) -> str:
        if re.sub(r"\s+", "", match.group()) != LIBRARY_EXPORT:
            raise ValueError(f"{path}: unsupported conditional attribute: {match.group()}")
        return "#[unsafe(no_mangle)]"

    return CONDITIONAL_ATTRIBUTE.sub(resolve, source)


def generate(root: Path, output: Path) -> None:
    """Generate through cbindgen's supported source-file input mode."""
    version = subprocess.check_output(["cbindgen", "--version"], text=True).strip()
    if version != EXPECTED_VERSION:
        raise ValueError(f"expected {EXPECTED_VERSION}, found {version}")
    crate = root / "crates/tensor4all-capi"
    with tempfile.TemporaryDirectory(prefix="tensor4all-capi-header-") as temporary:
        source = Path(temporary) / "src"
        shutil.copytree(crate / "src", source)
        for path in source.rglob("*.rs"):
            path.write_text(normal_library_source(path.read_text(), str(path.relative_to(source))))
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cbindgen", str(source / "lib.rs"), "--config", str(crate / "cbindgen.toml"),
             "--output", str(output)],
            check=True,
        )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=root / "crates/tensor4all-capi/include/tensor4all_capi.h")
    args = parser.parse_args()
    generate(root, args.output.resolve())


if __name__ == "__main__":
    main()
