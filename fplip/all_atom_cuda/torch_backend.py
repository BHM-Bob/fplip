"""
PyTorch CUDA Compute Backend

GPU-accelerated compute backend using PyTorch.
PyTorch provides built-in cdist and extensive CUDA support.

Requirements:
    - NVIDIA GPU with CUDA support
    - PyTorch with CUDA (pip install torch --index-url https://download.pytorch.org/whl/cu121)

Note:
    PyTorch has a built-in torch.cdist function which is highly optimized.
    This backend may be preferred when PyTorch is already available in the
    environment (e.g., for machine learning workflows).

GPU-Centric Mode:
    All computations stay on GPU. Results are returned as torch tensors.
    Data is only transferred back to CPU when explicitly requested via to_numpy(force=True).
"""

from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    pass

from fplip.all_atom_cuda.backend import ComputeBackend


class TorchBackend(ComputeBackend):
    """PyTorch CUDA compute backend.

    GPU-Centric Design:
    - All computations stay on GPU
    - Results are returned as torch tensors (not numpy arrays)
    - CPU conversion only happens when to_numpy(force=True) is called

    Parameters
    ----------
    device : str
        PyTorch device string (default: 'cuda:0')
    """

    def __init__(self, device: str = 'cuda:0'):
        try:
            import torch
            self._torch = torch
            self._device = torch.device(device)
            self.bool = torch.bool            
            if not self._device.type.startswith('cuda'):
                import warnings
                warnings.warn(
                    f"TorchBackend initialized with non-CUDA device '{device}'. "
                    "GPU acceleration will not be available.",
                    RuntimeWarning
                )
        except ImportError:
            raise ImportError(
                "PyTorch is not installed. Install it with: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

    @property
    def name(self) -> str:
        return "torch"

    @property
    def is_gpu(self) -> bool:
        return self._device.type.startswith('cuda')

    def to_device(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transfer array to GPU device.

        If arr is already a tensor on the correct device, return as-is.
        """
        if isinstance(arr, self._torch.Tensor):
            if arr.device == self._device:
                return arr
            return arr.to(self._device)
        return self._torch.from_numpy(np.asarray(arr)).to(self._device)

    def to_numpy(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert array to numpy (CPU) if force=True, otherwise keep on device.

        GPU-Centric Mode:
        - If force=False (default): returns tensor on GPU
        - If force=True: transfers to CPU and returns numpy array
        """
        if isinstance(arr, self._torch.Tensor):
            return arr.detach().cpu().numpy()
        return arr

    def cdist(self, A: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute pairwise distance matrix. Result stays on GPU."""
        A = self.to_device(A)
        B = self.to_device(B)

        if A.shape[0] == 0 or B.shape[0] == 0:
            return self._torch.empty((A.shape[0], B.shape[0]), device=self._device)

        return self._torch.cdist(A, B)

    def norm(self, arr: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> torch.Tensor:
        """Compute vector norm. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.linalg.norm(arr, dim=axis)

    def arccos(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Element-wise inverse cosine. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.arccos(arr)

    def dot(self, A: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Matrix multiplication. Result stays on GPU."""
        A = self.to_device(A)
        B = self.to_device(B)
        if A.ndim == 1 and B.ndim == 1:
            return self._torch.dot(A, B)
        elif A.ndim == 2 and B.ndim == 2:
            return self._torch.mm(A, B)
        else:
            return self._torch.tensordot(A, B, dims=([-1], [-2]))

    def argwhere(self, mask: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Find indices where mask is True. Result stays on GPU."""
        mask = self.to_device(mask)
        return self._torch.nonzero(mask, as_tuple=True)

    def cross(self, A: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Element-wise cross product. Result stays on GPU."""
        A = self.to_device(A)
        B = self.to_device(B)
        return self._torch.cross(A, B, dim=-1)

    def clip(self, arr: Union[np.ndarray, torch.Tensor], min_val: float, max_val: float) -> torch.Tensor:
        """Clip array values. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.clamp(arr, min_val, max_val)

    def mean(self, arr: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> torch.Tensor:
        """Compute mean along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.mean(arr, dim=axis)

    def svd(self, arr: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Singular Value Decomposition. Results stay on GPU."""
        arr = self.to_device(arr)
        U, S, Vh = self._torch.linalg.svd(arr)
        return U, S, Vh

    def sum(self, arr: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> torch.Tensor:
        """Sum along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.sum(arr, dim=axis)

    def maximum(self, a: Union[np.ndarray, torch.Tensor], b: float) -> torch.Tensor:
        """Element-wise maximum of array and scalar. Result stays on GPU."""
        a = self.to_device(a)
        return self._torch.maximum(a, self._torch.tensor(b, device=self._device))
    
    def max(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute maximum along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.max(arr)  # pyright: ignore[reportReturnType]

    def min(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute minimum along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.min(arr)  # pyright: ignore[reportReturnType]

    def sqrt(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Element-wise square root. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.sqrt(arr)

    def where(self, condition: Union[np.ndarray, torch.Tensor], x, y) -> torch.Tensor:
        """Element-wise conditional selection. Result stays on GPU."""
        condition = self.to_device(condition)
        # Ensure condition is boolean type
        if condition.dtype != self._torch.bool:
            condition = condition.to(self._torch.bool)
        # Handle scalar x and y
        if (not isinstance(x, (int, float))) and (not isinstance(x, self._torch.Tensor)):
            x = self._torch.tensor(x, device=self._device, dtype=self._torch.float64)
        if (not isinstance(y, (int, float))) and (not isinstance(y, self._torch.Tensor)):
            y = self._torch.tensor(y, device=self._device, dtype=self._torch.float64)
        return self._torch.where(condition, x, y)
    
    def where_on_condition(self, condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Element-wise conditional selection. Result stays on GPU."""
        return self._torch.where(condition)

    def degrees(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert angles from radians to degrees. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.rad2deg(arr)

    def expand_dims(self, arr: Union[np.ndarray, torch.Tensor], axis: int) -> torch.Tensor:
        """Expand the shape of an array. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._torch.unsqueeze(arr, dim=axis)
