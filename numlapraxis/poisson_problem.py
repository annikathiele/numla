from typing import Callable, Iterator
from collections import deque
import numpy as np

"""
Wir entnehmen die idx,inv, rhs Funktionen aus dem Wiki.
Compute_errors ist unsere Implementierung
"""

def idx_zero(js, n):
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

def inv_idx_zero(m, d, n):
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

def idx(js, n):
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

def inv_idx(m, d , n ) :
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
    b=[]
    for counter in range(1,(n-1)**d+1):
        punkt = inv_idx(counter,d,n)
        for i in punkt:
            i=i/n
        x=np.array(punkt)
        b.append(((1/n)**2)*f(x))
    return np.array(b)


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



    for i in range(1,(n-1)**d+1):
        sol=u(np.array(inv_idx(i,d,n))/n)
        sol_exact.append(sol)
    sol_array=np.array(sol_exact)
    norm=float(np.linalg.norm(sol_array-np.array(hat_u),ord=np.inf))

    return norm
