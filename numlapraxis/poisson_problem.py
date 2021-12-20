"""
    Serie 3
    Kurs: Praxisübung Numerische Lineare Algebra
    Programm: experiments_lu
    Authoren: Aron Ventura, Annika Thiele
    Datum: 20.12.2021
    Funktionen:
        idx_zero()
        inv_idx_zero()
        idx()
        inv_idx()
        rhs()
        compute_error()

"""

import numpy as np
import matplotlib.pyplot as plt
import block_matrix as bl
import linear_solvers as ls

 #pylint: disable=invalid-name
 #pylint: disable= pointless-string-statement

"""
Wir entnehmen die idx,inv Funktionen aus dem Wiki.
Compute_errors und rhs ist unsere Implementierung
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
    array=[]
    for counter in range(1,(n-1)**d+1):
        punkt = np.array(inv_idx(counter,d,n))/n
        array.append(((1/n)**2)*f(punkt))
    return np.array(array)


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

def graph_errors(f,u,n):
    """
    Diese Funktion erstellt einen Graphen, der die Fehler der approximierten
    Lösung für das Poisson Problem in ABhängigkeit von den
    Diskretisierungspunkten darstellt.
    Parameter
    ------
    f Callable Funktion, für die das Poisson Problem gelöst werden soll.
    u Callable Exakte Lösung des Poisson Problems
    n Maximales n für das der Fehler berechnet werden soll
    Returns
    -------
    None
    """
    xlist =[]
    for i in range(2,n):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        p,l,uu=A.get_lu()
        b = rhs(1, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=compute_error(1,x, hat_u, u)
        ylist.append(e)

    ylistt=[]
    for x in xlist:
        A=bl.BlockMatrix(2, x)
        p,l,uu=A.get_lu()
        b = rhs(2, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=compute_error(2,x, hat_u, u)
        ylistt.append(e)

    ylisttt=[]
    for x in xlist:
        A=bl.BlockMatrix(3, x)
        p,l,uu=A.get_lu()
        b = rhs(3, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=compute_error(3,x, hat_u, u)
        ylisttt.append(e)

    plt.plot(xlist, ylist, 'b-',label= 'd=1' )
    plt.plot(xlist, ylistt, 'b-',label= 'd=2' )
    plt.plot(xlist, ylisttt, 'b-',label= 'd=3)' )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
