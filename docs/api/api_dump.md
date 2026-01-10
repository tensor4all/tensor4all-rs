# api-dump

## src/main.rs

### ` fn main() -> Result < () , Box < dyn std :: error :: Error > >`

### ` fn expand_glob(base: & Path, pattern: & str) -> Vec < PathBuf >`

### ` fn process_crate(crate_path: & Path, output_dir: & Path) -> Result < () , Box < dyn std :: error :: Error > >`

### ` fn parse_file(path: & Path) -> Result < Vec < FuncInfo > , Box < dyn std :: error :: Error > >`

### ` fn extract_items(items: & [Item], funcs: & mut Vec < FuncInfo >)`

### ` fn extract_fn_info(item_fn: & ItemFn) -> FuncInfo`

### ` fn extract_impl_items(item_impl: & ItemImpl, funcs: & mut Vec < FuncInfo >)`

### ` fn extract_trait_items(item_trait: & ItemTrait, funcs: & mut Vec < FuncInfo >)`

### ` fn vis_to_string(vis: & Visibility) -> String`

### ` fn sig_to_string(sig: & Signature) -> String`

### ` fn fn_arg_to_string(arg: & FnArg) -> String`

### ` fn type_to_string(ty: & syn :: Type) -> String`

### ` fn extract_doc_summary(attrs: & [Attribute]) -> Option < String >`

