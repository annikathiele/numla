"""
    Serie 3
    Kurs: Praxisübung Numerische Lineare Algebra
    Programm: linear_solvers
    Authoren: Aron Ventura, Annika Thiele
    Datum: 20.12.2021
    Funktionen:
        solve_lu()
"""


from scipy.linalg import solve_triangular
import numpy as np
import block_matrix as bl
 #pylint: disable=invalid-name

def main():
    """
    The main demonstrates the fundtionality od the functions.
    Returns
    -------
    None
    """
    A= bl.BlockMatrix(2,3)
    p,l,u= A.get_lu()
    b=np.array([2,4,4,2])
    x= solve_lu(p,l,u,b)
    print(x)

def solve_lu(p, l, u, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u.
    Parameters
    ----------
    p : numpy.ndarray
    permutation matrix of LU-decomposition
    l : numpy.ndarray
    lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
    upper triangular matrix of LU-decomposition
    b : numpy.ndarray
    vector of the right-hand-side of the linear system
    Returns
    -------
    x : numpy.ndarray
    solution of the linear system
    """
    Pb= np.matmul(np.transpose(p),b)
    y = solve_triangular(l,Pb, lower=True)
    x= solve_triangular(u,y)
    return x

if __name__ == "__main__":
    main()
