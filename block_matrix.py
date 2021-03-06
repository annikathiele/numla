import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import seaborn as sns
import scipy.linalg as lg
from scipy.sparse.linalg import inv, norm
import poisson_problem_23 as pp
 #pylint: disable=invalid-name
"""
def main():
    vgl_cond()
    vgl_spar()
    A= BlockMatrix(3,3)
    print(A.get_sparse())
    p,l,u = A.get_lu()
    print(A.eval_sparsity_lu())
    print(A.get_cond())
    #graph_cond()
    graph_sparsity()
"""

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.
    Parameters ---------d : int Dimension of the space. n : int Number of
    intervals in each dimension.
    Attributes ---------d : int Dimension of the space. n : int Number of
    intervals in each dimension.
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
                idt = sparse.identity(matrix.shape[0], format='csc', dtype='int8')
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
                    rows.append(sparse.hstack(entries, format='csc'))
                matrix = sparse.vstack(rows, format='csc')
            self.sparse = matrix
        return self.sparse

    def eval_sparsity(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.
        Returns
        -------
        int
        Number of non-zeros
        float
        Relative number of non-zeros
        """
        csr = self.get_sparse()
        nnz = csr.count_nonzero()
        rel_nnz = nnz / (csr.shape[0] * csr.shape[1])
        return (nnz, rel_nnz)

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

    def eval_sparsity_lu(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.
        Returns
        -------
        int
        Number of non-zeros
        float
        Relative number of non-zeros
        """
        p,l,u = self.get_lu()
        count_l = np.count_nonzero(l)-(self.n-1)**(self.d)
        count_u = np.count_nonzero(u)
        nonzeros= count_l + count_u
        rel = float(nonzeros) / float((self.n-1)**(2*self.d))
        return nonzeros, rel

    def get_cond(self):
        """ Computes the condition number of the represented matrix.
        Returns
        -------
        float
        condition number with respect to the infinity-norm
        """
        A = self.get_sparse()
        A_inv = inv(A)
        cond= norm(A, np.inf)*norm(A_inv, np.inf)
        return cond

def graph_cond():
 
    """ Erstellt ein Plot, welcher f??r Dimension 1,2,3 die Kondition der Blockmatirx in Abh??ngigkeit von 
    der Anzahl an inneren Diskretisierungspunkten abbildet. 
    Returns
    -------
    None
    """

    nlist_one=[]
    nlist_two =[]
    nlist_three =[]
    Nlist_one=[]
    Nlist_two=[]
    Nlist_three=[]
    ylist_one = []
    ylist_two = []
    ylist_three = []
    ylist = [ylist_one , ylist_two , ylist_three ]
    nlist = [nlist_one , nlist_two , nlist_three ]
    Nlist = [Nlist_one , Nlist_two , Nlist_three ]
    rangelist = []
    for counter in range (1, 6):
        rangelist.append(counter*(1700/20))
    """
    for counter in range (1,4):
        rangelist.append(10**counter)
    """
    for dimension in range (1,4):
        for i in rangelist:
            n=int(i**(float(1)/float(dimension))+1)
            nlist[dimension-1].append(n)
            Nlist[dimension-1].append((n-1)**(dimension))
        for n in nlist[dimension-1]:
            A=BlockMatrix(dimension, n)
            ylist[dimension-1].append(A.get_cond())

    plt.plot(Nlist_one, ylist_one, 'b-',label= 'd=1' )
    plt.plot(Nlist_two, ylist_two, 'r-',label= 'd=2' )
    plt.plot(Nlist_three, ylist_three, 'y-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

def graph_sparsity():
    nlist_one=[]
    nlist_two =[]
    nlist_three =[]
    Nlist_one=[]
    Nlist_two=[]
    Nlist_three=[]
    ylist_one = []
    ylist_two = []
    ylist_three = []
    ylist = [ylist_one , ylist_two , ylist_three ]
    nlist = [nlist_one , nlist_two , nlist_three ]
    Nlist = [Nlist_one , Nlist_two , Nlist_three ]
    rangelist = []
    for counter in range (1, 6):
        rangelist.append(counter*(1700/20))
    """
    for counter in range (1,4):
        rangelist.append(10**counter)
    """
    for dimension in range (1,4):
        for i in rangelist:
            n=int(i**(float(1)/float(dimension))+1)
            nlist[dimension-1].append(n)
            Nlist[dimension-1].append((n-1)**(dimension))
        for n in nlist[dimension-1]:
            A=BlockMatrix(dimension, n)
            ylist[dimension-1].append(A.eval_sparsity_lu()[0])

    plt.plot(Nlist_one, ylist_one, 'b-',label= 'd=1' )
    plt.plot(Nlist_two, ylist_two, 'r-',label= 'd=2' )
    plt.plot(Nlist_three, ylist_three, 'y-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

"""
def graph_error():
   
    nlist_one=[]
    nlist_two =[]
    nlist_three =[]
    Nlist_one=[]
    Nlist_two=[]
    Nlist_three=[]
    ylist_one = []
    ylist_two = []
    ylist_three = []
    ylist = [ylist_one , ylist_two , ylist_three ]
    nlist = [nlist_one , nlist_two , nlist_three ]
    Nlist = [Nlist_one , Nlist_two , Nlist_three ]
    rangelist = []
    for counter in range (1, 4):
        rangelist.append(counter*(1700/20))
    for dimension in range (1,4):
        for i in rangelist:
            n=int(i**(float(1)/float(dimension))+1)
            nlist[dimension-1].append(n)
            Nlist[dimension-1].append((n-1)**(dimension))
        for n in nlist[dimension-1]:
            ylist[dimension-1].append(pp.compute_error(dimension, n, hat_u, u))

    plt.plot(Nlist_one, ylist_one, 'b-',label= 'd=1' )
    plt.plot(Nlist_two, ylist_two, 'r-',label= 'd=2' )
    plt.plot(Nlist_three, ylist_three, 'y-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
"""


def cond_hilmat(n):
    """ 
    Liefert zu einem gegebenen n die Kondition der HIlbertmatrix der Gr????e nxn.
    Input
    ______
    n (int) Gr????e der zu betrachtenden Hilbertmatrix
    Returns
    -------
    float Kondition der Hilbertmatrix der Gr????e nxn
    """
    cond= lg.norm(lg.hilbert(n), np.inf)*lg.norm(lg.invhilbert(n), np.inf)
    return cond

def vgl_cond():
    """ 
    Gibt f??r n=1 bis n=6 die Konditionen der Hilbert und Blockmatritzen der jeweiligen Gr????e aus.
    Input
    ______
    None
    Returns
    -------
    None
    """
    
    for counter in range(1,7):
        A= BlockMatrix(counter,5)
        print("d =", counter)
        print("Kondition A^d =", A.get_cond(), "Kondition Hilbert\
 =" , cond_hilmat(counter))

def vgl_spar():
    """ 
    Gibt f??r verschiedene Dimensionen und n die Sparsit??t der Blockmatrix und der LU Zerlegung aus.
    Input
    ______
    None
    Returns
    -------
    None
    """
    print("Dimension, n , Sparsity A^d , Sparsity LU")
    for d in range(1,4):
        for n in range(2,15):
            A= BlockMatrix(d,n)
            print(d,n, A.eval_sparsity()[0] , A.eval_sparsity_lu()[0])
"""
if __name__ == "__main__":
    main()
"""
