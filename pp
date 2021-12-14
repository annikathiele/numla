from typing import Callable, Iterator
from collections import deque
import numpy as np




def idx_zero(js : list[int], n : int) -> int:
    """Indexbügel: flatten the d-dimensional index into its traversal order, first index is 0
    Parameters:
        js : list[int]
            indices of discretised point (dimension d inferred as len(js))
        n : int
            Frequenct of discretisation
    Returns:
        m : int
            flattened index
    """
    d = len(js)
    m = 0
    n = n-1
    for k in range(d - 1, 0, -1):
        m += js[k]
        m *= n # n-1
    m += js[0]
    return m

def inv_idx_zero(m : int, d : int, n : int) -> list[int]:
    """Retrieve the original indices from the flattened one, first index is 0
    Parameters:
        m : int
            flattened index
        d : int
            dimension
        n : int
            frequency of discretisation
    Returns:
        js : list[int]
            original indices
    """
    js = []
    n = n-1
    for _ in range(d):
        js.append(m % n) # n-1
        m = m // n # n-1
    return js

def idx(js : list[int], n : int) -> int:
    """Indexbügel: flatten the d-dimensional index into its traversal order, first index is 1
    Parameters:
        js : list[int]
            indices of discretised point (dimension d inferred as len(js))
        n : int
            Frequenct of discretisation
    Returns:
        m : int
            flattened index
    """
    return idx_zero(list(map((lambda t : t - 1), js)), n) + 1

def inv_idx(m : int, d : int, n : int) -> list[int]:
    """Retrieve the original indices from the flattened one, first index is 1
    Parameters:
        m : int
            flattened index
        d : int
            dimension
        n : int
            frequency of discretisation
    Returns:
        js : list[int]
            original indices
    """
    return list(map((lambda t : t + 1), inv_idx_zero(m - 1, d, n)))


def rhs(d, n, f):
    """Sample f at discretisation points and flatten
    Parameters:
        d : int
            dimension
        n : int
            frequency of discretisation
        f : callable[[numpy.ndarray], float]
            function f: (0, 1)^d -> RR
    Returns:
        b : numpy.ndarray
            sample of f
    Raises:
        ValueError
            d < 1 or n < 2
    """
    if d < 1 or n < 2 :
        raise ValueError
    return (
        np.array(list(map(
            (lambda m: f(np.array(inv_idx(m, d, n)) / n , d)), # inefficient as hell
            range(1, (n - 1) ** d + 1)
        ))) / (n ** 2)
    )





def compute_error(d, n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.
    Parameters
    ----------
    d : int
       Dimension of the space
    n : int
       Number of intersections in each dimension
    hat_u : array_like of ’numpy’
       Finite difference approximation of the solution of the Poisson problem
       at the discretization points
    u : callable
       Solution of the Poisson problem
       The calling signature is ’u(x)’. Here ’x’ is an array_like of ’numpy’.
       The return value is a scalar.
    Returns
    -------
    float
       maximal absolute error at the discretization points
    """
    sol_exact=[]
    for counter in range(1,(n-1)**d+1):
        sol_exact.append(u(inv_idx(counter,d,n)))

    norm=0

    for i in range(len(sol_exact)):
        dif = abs(sol_exact[i]-hat_u[i])
        if dif>=norm:
            norm = dif

    return norm



