import torch

if torch.cuda.is_available():
    DEFAULT_TENSOR_TYPE = torch.cuda.DoubleTensor
else:
    DEFAULT_TENSOR_TYPE = torch.DoubleTensor

DEFAULT_DTYPE = torch.double
