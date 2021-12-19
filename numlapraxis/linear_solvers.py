
import block_matrix as bl
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import seaborn as sns
import scipy.linalg as lg
from scipy.sparse.linalg import inv, norm

def main():
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
    y = solve_triangular(l,b, lower=True)
    x= solve_triangular(u,y)
    return x

if __name__ == "__main__":
    main()



class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations of the Laplace operator.
    Parameters ---------d : int Dimension of the space. n : int Number of intervals in each dimension.
    Attributes ---------d : int Dimension of the space. n : int Number of intervals in each dimension.
    Raises -----ValueError If d < 1 or n < 2.
    """
    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.sparse = None


    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.
    Returns ------scipy.sparse.csr_matrix The block_matrix in a sparse data format. """
        if self.sparse is None:
            matrix = sparse.csr_matrix(np.array([[2 * self.d]]))
            for _ in range(self.d):
                idt = sparse.identity(matrix.shape[0], format='csr', dtype='int8')
                zero = sparse.csr_matrix(matrix.shape, dtype='int8')
                rows = []
                for r in range(self.n-1):
                    entries = []
                    for c in range(self.n-1):
                        if c == r:
                            entries.append(matrix)
                        elif abs(c-r) == 1:
                            entries.append(-idt)
                        else:
                            entries.append(zero)
                    rows.append(sparse.hstack(entries, format='csr'))
                matrix = sparse.vstack(rows, format='csr')
            self.sparse = matrix
        return self.sparse


    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u
        Returns
        -------
        p : numpy.ndarray
        permutation matrix of LU-decomposition
        l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
        upper triangular matrix of LU-decomposition
        """
        a= self.get_sparse()
        p,l,u=lg.lu(a.todense())
        return p,l,u