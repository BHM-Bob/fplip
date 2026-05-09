"""
NumPy Compute Backend

CPU-based compute backend using NumPy and SciPy.
This backend serves as the reference implementation and fallback
when GPU backends are not available.

Note:
    This backend provides the same API as GPU backends but operates
    entirely on CPU. It is useful for:
    - Baseline testing and validation
    - Systems without GPU
    - Debugging and development
    - Comparison with GPU-accelerated results
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist

from fplip.all_atom_cuda.backend import ComputeBackend


class NumPyBackend(ComputeBackend):
    """NumPy CPU compute backend.

    This backend uses NumPy and SciPy for all computations.
    It serves as the reference implementation for correctness testing
    and provides a fallback when GPU is not available.

    Parameters
    ----------
    None
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def is_gpu(self) -> bool:
        return False
    
    def arange(self, stop: int) -> np.ndarray:
        return np.arange(stop)

    def full(self, shape: Tuple[int], fill: Union[bool, int], dtype: Union[bool, int] = None) -> np.ndarray:
        return np.full(shape, fill, dtype=dtype)

    def to_device(self, arr: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Convert input to numpy array.

        For NumPy backend, this simply ensures the input is a numpy array.

        Parameters
        ----------
        arr : np.ndarray, list, or tuple
            Input array or array-like object

        Returns
        -------
        np.ndarray
            Numpy array
        """
        if isinstance(arr, np.ndarray):
            return arr
        return np.asarray(arr)

    def to_numpy(self, arr: Union[np.ndarray, list, tuple], force: bool = False) -> np.ndarray:
        """Convert array to numpy (no-op for NumPy backend).

        Parameters
        ----------
        arr : np.ndarray or array-like
            Input array
        force : bool, optional
            Ignored for NumPy backend (default: False)

        Returns
        -------
        np.ndarray
            Numpy array
        """
        if isinstance(arr, np.ndarray):
            return arr
        return np.asarray(arr)

    def cdist(self, A: Union[np.ndarray, list, tuple], B: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Compute pairwise distance matrix using scipy.

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
        A = self.to_device(A)
        B = self.to_device(B)

        if A.shape[0] == 0 or B.shape[0] == 0:
            return np.empty((A.shape[0], B.shape[0]))

        return scipy_cdist(A, B, metric='euclidean')

    def norm(self, arr: Union[np.ndarray, list, tuple], axis: Optional[int] = None) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.linalg.norm(arr, axis=axis) # type: ignore

    def arccos(self, arr: Union[np.ndarray, list, tuple]) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.arccos(arr)

    def dot(self, A: Union[np.ndarray, list, tuple], B: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Matrix or vector dot product.

        Parameters
        ----------
        A : np.ndarray
            First matrix/vector
        B : np.ndarray
            Second matrix/vector

        Returns
        -------
        np.ndarray
            Dot product result
        """
        A = self.to_device(A)
        B = self.to_device(B)
        return np.dot(A, B)

    def argwhere(self, mask: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, ...]:
        """Find indices where mask is True.

        Parameters
        ----------
        mask : np.ndarray, bool
            Boolean mask

        Returns
        -------
        tuple of np.ndarray
            Indices of True elements (one array per dimension)
        """
        mask = self.to_device(mask)
        return np.nonzero(mask)

    def cross(self, A: Union[np.ndarray, list, tuple], B: Union[np.ndarray, list, tuple]) -> np.ndarray:
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
        A = self.to_device(A)
        B = self.to_device(B)
        return np.cross(A, B)

    def clip(self, arr: Union[np.ndarray, list, tuple], min_val: float, max_val: float) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.clip(arr, min_val, max_val)

    def mean(self, arr: Union[np.ndarray, list, tuple], axis: Optional[int] = None) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.mean(arr, axis=axis)

    def svd(self, arr: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        arr = self.to_device(arr)
        return np.linalg.svd(arr, full_matrices=False)

    def sum(self, arr: Union[np.ndarray, list, tuple], axis: Optional[int] = None) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.sum(arr, axis=axis)

    def maximum(self, a: Union[np.ndarray, list, tuple], b: float) -> np.ndarray:
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
        a = self.to_device(a)
        return np.maximum(a, b)

    def max(self, arr: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Compute maximum value.

        Parameters
        ----------
        arr : np.ndarray
            Input array

        Returns
        -------
        np.ndarray or scalar
            Maximum value
        """
        arr = self.to_device(arr)
        return np.max(arr)

    def min(self, arr: Union[np.ndarray, list, tuple], dim: Optional[int] = None) -> np.ndarray:
        """Compute minimum value.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        dim : int, optional
            Dimension along which to compute minimum value

        Returns
        -------
        np.ndarray or scalar
            Minimum value
        """
        arr = self.to_device(arr)
        return np.min(arr, axis=dim)

    def sqrt(self, arr: Union[np.ndarray, list, tuple]) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.sqrt(arr)

    def where(self, condition: Union[np.ndarray, list, tuple], x, y) -> np.ndarray:
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
        condition = self.to_device(condition)
        # Handle scalar x and y
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        return np.where(condition, x, y)

    def where_on_condition(self, condition: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, ...]:
        """Find indices where condition is True.

        Parameters
        ----------
        condition : np.ndarray, bool
            Boolean condition array

        Returns
        -------
        tuple of np.ndarray
            Indices where condition is True
        """
        return self.argwhere(condition)

    def degrees(self, arr: Union[np.ndarray, list, tuple]) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.degrees(arr)

    def expand_dims(self, arr: Union[np.ndarray, list, tuple], axis: int) -> np.ndarray:
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
        arr = self.to_device(arr)
        return np.expand_dims(arr, axis)
