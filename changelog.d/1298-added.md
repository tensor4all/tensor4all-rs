- `tensor4all-treetn` adds `LinearOperator::to_state_network` and
  `StateOperatorNetwork` (#1298): an operator on any subset of a state's nodes,
  written as a plain `TreeTN` on the state's whole tree, with true input
  indices, identity gaps, bonds routed along the state's paths, and one scalar
  type. It is the operand a `PartitionedTreeTN` contraction such as
  `contract_adaptive` needs, without rewriting indices or promoting dtypes by
  hand.
