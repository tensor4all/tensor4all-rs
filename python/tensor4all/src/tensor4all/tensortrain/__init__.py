"""TensorTrain subpackage for tensor4all.

Provides tensor train (Matrix Product State) operations.

Examples
--------
>>> from tensor4all.tensortrain import TensorTrainF64, TensorTrainC64
>>>
>>> # Create a constant tensor train
>>> tt = TensorTrainF64.constant([2, 3, 2], 1.0)
>>> print(tt.sum())  # 12.0
>>>
>>> # Arithmetic operations
>>> tt2 = TensorTrainF64.constant([2, 3, 2], 2.0)
>>> tt3 = tt + tt2
>>> print(tt3.sum())  # 36.0
"""

from ._tensortrain import TensorTrainF64, TensorTrainC64

__all__ = ["TensorTrainF64", "TensorTrainC64"]
