# tensor4all

Unified re-export crate that provides access to all tensor4all functionality through a single dependency.

## Features

- Re-exports core types from `tensor4all-core`
- Re-exports tensor train functionality from `tensor4all-itensorlike`
- Re-exports tree tensor networks from `tensor4all-treetn`
- Single dependency for full tensor4all functionality

## Usage

```rust
use tensor4all::prelude::*;

// All tensor4all types are available through this crate
let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let tensor = TensorDynLen::random(&mut rng, &[i, j]);
```

## License

MIT License
