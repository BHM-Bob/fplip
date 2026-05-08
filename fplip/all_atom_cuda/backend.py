"""
Compute Backend Abstract Layer

Provides a unified API for numerical computation backends (NumPy, CuPy, PyTorch).
All-Atom-CUDA uses this abstraction to switch between CPU and GPU computation
without modifying interaction detection logic.

GPU-Centric Mode (Default):
    - Input: numpy arrays (from CPU) or device arrays
    - Computation: on the backend's device (CPU/GPU)
    - Output: device arrays (stay on GPU for GPU backends)
    
    This minimizes CPU-GPU data transfers and maximizes performance.
    Data is only converted back to CPU when explicitly requested.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class ComputeBackend(ABC):
    """Abstract base class for compute backends.

    GPU-Centric Design:
    - All computations stay on the device (GPU for CUDA backends)
    - Results are returned as device arrays, not numpy arrays
    - CPU conversion only happens when explicitly requested via to_numpy()
    
    This design maximizes performance by minimizing CPU-GPU data transfers.
    The interaction detection code works with device arrays directly.
    """

    @abstractmethod
    def to_device(self, arr: np.ndarray):
        """Transfer a numpy array to the compute device.

        Parameters
        ----------
        arr : np.ndarray
            Input numpy array

        Returns
        -------
        Device array (backend-specific type)
        """
        pass

    @abstractmethod
    def to_numpy(self, arr, force: bool = False) -> np.ndarray:
        """Transfer a device array back to numpy (CPU).

        In GPU-centric mode, this should only be called when explicitly
        needed (e.g., for creating Interaction objects or final output).

        Parameters
        ----------
        arr : device array or np.ndarray
            Input array (on device or already numpy)
        force : bool
            If True, always convert to numpy (for final output)
            If False, may return device array for GPU backends

        Returns
        -------
        np.ndarray or device array
            Array on CPU (if force=True or CPU backend) or device
        """
        pass

    @abstractmethod
    def cdist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix between two sets of points.

        Parameters
        ----------
        A : np.ndarray, shape [M, 3]
            First set of coordinates
        B : np.ndarray, shape [N, 3]
            Second set of coordinates

        Returns
        -------
        np.ndarray, shape [M, N]
            Distance matrix where result[i, j] = ||A[i] - B[j]||
        """
        pass

    @abstractmethod
    def norm(self, arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Compute vector norm.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        axis : int, optional
            Axis along which to compute norm

        Returns
        -------
        np.ndarray
            Norm values
        """
        pass

    @abstractmethod
    def arccos(self, arr: np.ndarray) -> np.ndarray:
        """Element-wise inverse cosine.

        Parameters
        ----------
        arr : np.ndarray
            Input values in [-1, 1]

        Returns
        -------
        np.ndarray
            Angles in radians
        """
        pass

    @abstractmethod
    def dot(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication.

        Parameters
        ----------
        A : np.ndarray
            First matrix
        B : np.ndarray
            Second matrix

        Returns
        -------
        np.ndarray
            Matrix product
        """
        pass

    @abstractmethod
    def argwhere(self, mask: np.ndarray) -> np.ndarray:
        """Find indices where mask is True.

        Parameters
        ----------
        mask : np.ndarray, bool
            Boolean mask

        Returns
        -------
        np.ndarray, shape [K, ndim]
            Indices of True elements
        """
        pass

    @abstractmethod
    def cross(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Element-wise cross product.

        Parameters
        ----------
        A : np.ndarray, shape [..., 3]
            First vector(s)
        B : np.ndarray, shape [..., 3]
            Second vector(s)

        Returns
        -------
        np.ndarray, shape [..., 3]
            Cross product
        """
        pass

    @abstractmethod
    def clip(self, arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Clip array values to [min_val, max_val].

        Parameters
        ----------
        arr : np.ndarray
            Input array
        min_val : float
            Minimum value
        max_val : float
            Maximum value

        Returns
        -------
        np.ndarray
            Clipped array
        """
        pass

    @abstractmethod
    def mean(self, arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Compute mean along axis.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        axis : int, optional
            Axis along which to compute mean

        Returns
        -------
        np.ndarray
            Mean values
        """
        pass

    @abstractmethod
    def svd(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition.

        Parameters
        ----------
        arr : np.ndarray, shape [M, N]
            Input matrix

        Returns
        -------
        U : np.ndarray
        S : np.ndarray
        Vh : np.ndarray
        """
        pass

    @abstractmethod
    def sum(self, arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Sum along axis.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        axis : int, optional
            Axis along which to sum

        Returns
        -------
        np.ndarray
            Sum values
        """
        pass

    @abstractmethod
    def maximum(self, a: np.ndarray, b: float) -> np.ndarray:
        """Element-wise maximum of array and scalar.

        Parameters
        ----------
        a : np.ndarray
            Input array
        b : float
            Scalar value

        Returns
        -------
        np.ndarray
            Element-wise maximum
        """
        pass
    
    @abstractmethod
    def min(self, arr: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def max(self, arr: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sqrt(self, arr: np.ndarray) -> np.ndarray:
        """Element-wise square root.

        Parameters
        ----------
        arr : np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Square root values
        """
        pass

    @abstractmethod
    def where(self, condition: np.ndarray, x, y) -> np.ndarray:
        """Element-wise conditional selection.

        Parameters
        ----------
        condition : np.ndarray, bool
            Boolean condition array
        x : scalar or np.ndarray
            Value(s) to use where condition is True
        y : scalar or np.ndarray
            Value(s) to use where condition is False

        Returns
        -------
        np.ndarray
            Array with elements from x where condition is True, else from y
        """
        pass
    
    @abstractmethod
    def where_on_condition(self, condition: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Element-wise conditional selection. Result stays on GPU."""
        pass

    @abstractmethod
    def degrees(self, arr: np.ndarray) -> np.ndarray:
        """Convert angles from radians to degrees.

        Parameters
        ----------
        arr : np.ndarray
            Angles in radians

        Returns
        -------
        np.ndarray
            Angles in degrees
        """
        pass

    @abstractmethod
    def expand_dims(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Expand the shape of an array by inserting a new axis.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        axis : int
            Position in the expanded axes where the new axis is placed

        Returns
        -------
        np.ndarray
            Array with expanded shape
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        """Whether this backend runs on GPU."""
        pass
