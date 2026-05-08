"""
CuPy Compute Backend

GPU-accelerated compute backend using CuPy.
CuPy provides a NumPy-compatible API that runs on NVIDIA GPUs via CUDA.

Requirements:
    - NVIDIA GPU with CUDA support
    - CuPy package (pip install cupy-cuda12x or cupy-cuda11x)

Note:
    CuPy does not provide a built-in cdist function, so we implement
    distance matrix computation using the expansion:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b

GPU-Centric Mode:
    All computations stay on GPU. Results are returned as cupy arrays.
    Data is only transferred back to CPU when explicitly requested via to_numpy(force=True).
"""

from typing import Optional, Tuple, Union

import numpy as np
try:
    import cupy
except ImportError:
    pass

from fplip.all_atom_cuda.backend import ComputeBackend


class CuPyBackend(ComputeBackend):
    """CuPy GPU compute backend.

    GPU-Centric Design:
    - All computations stay on GPU
    - Results are returned as cupy arrays (not numpy arrays)
    - CPU conversion only happens when to_numpy(force=True) is called

    Parameters
    ----------
    device_id : int
        CUDA device ID (default: 0)
    """

    def __init__(self, device_id: int = 0):
        try:
            import cupy as cp
            self._cp = cp
            cp.cuda.Device(device_id).use()
            self._device_id = device_id
        except ImportError:
            raise ImportError(
                "CuPy is not installed. Install it with: "
                "pip install cupy-cuda12x (for CUDA 12.x) or "
                "pip install cupy-cuda11x (for CUDA 11.x)"
            )

    @property
    def name(self) -> str:
        return "cupy"

    @property
    def is_gpu(self) -> bool:
        return True

    def to_device(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Transfer array to GPU device.

        If arr is already a cupy array, return as-is.
        """
        if isinstance(arr, self._cp.ndarray):
            return arr
        return self._cp.asarray(arr)

    def to_numpy(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> Union[np.ndarray, 'cupy.ndarray']:
        """Convert array to numpy (CPU).
        """
        if isinstance(arr, self._cp.ndarray):
            return arr.get()
        return arr

    def cdist(self, A: Union[np.ndarray, 'cupy.ndarray'], B: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Compute pairwise distance matrix. Result stays on GPU."""
        A = self.to_device(A)
        B = self.to_device(B)

        if A.shape[0] == 0 or B.shape[0] == 0:
            return self._cp.empty((A.shape[0], B.shape[0]))

        A2 = self._cp.sum(A ** 2, axis=1, keepdims=True)
        B2 = self._cp.sum(B ** 2, axis=1, keepdims=True)
        dist_sq = A2 + B2.T - 2.0 * self._cp.dot(A, B.T)
        dist_sq = self._cp.maximum(dist_sq, 0.0)
        return self._cp.sqrt(dist_sq)

    def norm(self, arr: Union[np.ndarray, 'cupy.ndarray'], axis: Optional[int] = None) -> 'cupy.ndarray':
        """Compute vector norm. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.linalg.norm(arr, axis=axis)

    def arccos(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Element-wise inverse cosine. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.arccos(arr)

    def dot(self, A: Union[np.ndarray, 'cupy.ndarray'], B: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Matrix multiplication. Result stays on GPU."""
        A = self.to_device(A)
        B = self.to_device(B)
        return self._cp.dot(A, B)

    def argwhere(self, mask: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Find indices where mask is True. Result stays on GPU."""
        mask = self.to_device(mask)
        row, col = self._cp.nonzero(mask)
        return row.tolist(), col.tolist()

    def cross(self, A: Union[np.ndarray, 'cupy.ndarray'], B: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Element-wise cross product. Result stays on GPU."""
        A = self.to_device(A)
        B = self.to_device(B)
        return self._cp.cross(A, B)

    def clip(self, arr: Union[np.ndarray, 'cupy.ndarray'], min_val: float, max_val: float) -> 'cupy.ndarray':
        """Clip array values. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.clip(arr, min_val, max_val)

    def mean(self, arr: Union[np.ndarray, 'cupy.ndarray'], axis: Optional[int] = None) -> 'cupy.ndarray':
        """Compute mean along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.mean(arr, axis=axis)

    def svd(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> Tuple['cupy.ndarray', 'cupy.ndarray', 'cupy.ndarray']:
        """Singular Value Decomposition. Results stay on GPU."""
        arr = self.to_device(arr)
        U, S, Vh = self._cp.linalg.svd(arr)
        return U, S, Vh

    def sum(self, arr: Union[np.ndarray, 'cupy.ndarray'], axis: Optional[int] = None) -> 'cupy.ndarray':
        """Sum along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.sum(arr, axis=axis)

    def maximum(self, a: Union[np.ndarray, 'cupy.ndarray'], b: float) -> 'cupy.ndarray':
        """Element-wise maximum of array and scalar. Result stays on GPU."""
        a = self.to_device(a)
        return self._cp.maximum(a, b)
    
    def max(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Compute maximum along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.max(arr)
    
    def min(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Compute minimum along axis. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.min(arr)

    def sqrt(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Element-wise square root. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.sqrt(arr)

    def where(self, condition: Union[np.ndarray, 'cupy.ndarray'], x, y) -> 'cupy.ndarray':
        """Element-wise conditional selection. Result stays on GPU."""
        condition = self.to_device(condition)
        # Handle scalar x and y
        if not isinstance(x, self._cp.ndarray):
            x = self._cp.asarray(x)
        if not isinstance(y, self._cp.ndarray):
            y = self._cp.asarray(y)
        return self._cp.where(condition, x, y)
    
    def where_on_condition(self, condition: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Element-wise conditional selection. Result stays on GPU."""
        condition = self.to_device(condition)
        idx = self._cp.where(condition)[0]
        return (idx.tolist(), )

    def degrees(self, arr: Union[np.ndarray, 'cupy.ndarray']) -> 'cupy.ndarray':
        """Convert angles from radians to degrees. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.degrees(arr)

    def expand_dims(self, arr: Union[np.ndarray, 'cupy.ndarray'], axis: int) -> 'cupy.ndarray':
        """Expand the shape of an array. Result stays on GPU."""
        arr = self.to_device(arr)
        return self._cp.expand_dims(arr, axis)
