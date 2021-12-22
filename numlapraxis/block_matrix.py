
"""
    Serie 3
    Kurs: Praxisübung Numerische Lineare Algebra
    Programm: block_matrix
    Authoren: Aron Ventura, Annika Thiele
    Datum: 10.12.2021
    Funktionen:
        get_sparse()
        eval_sparsity()
        get_lu()
        eval_sparsity_lu()
        get_cond()
        graph_sparsity()
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import inv, norm
from scipy import sparse

 #pylint: disable=invalid-name
 #pylint: disable=unused-variable
 #pylint: disable= unbalanced-tuple-unpacking

def main():
    """
    Die Main Funktion demonstriert die Funktionalität der Funktionen.
    Returns
    -------
    None
    """
    print("Mit diesem Programm lässt sich die LU-Zerlegung, die Kondition \
und Sparsity unserer Koeffizientenmatrix bestimmen.")
    print("Wir nutzen dies um zwei Plots zu erstellen")
    print("Beide sind in Abhängigkeit von N mit Werten bis N=595 mit \
einer Schrittweite von 85")
    print("Wenn Sie den Plot zur Kondition sehen möchten, geben Sie bitte\
 die 0 ein.")
    print("1 führt Sie zum Plot über die Sparsity.")
    while True:
        try:
            print()
            auswahl = input("Welchen Plot möchten Sie sehen? ")
            auswahl = int(auswahl)
            if auswahl == 0:
                print()
                print("Hier der Plot zur Kondition von A für d = 1,2,3 in \
Abhängigkeit von N")
                graph_cond()
                break
            if auswahl ==1:
                print()
                print("Hier der Plot zur Sparsity von A und der LU-Zerlegung \
von A für d = 1,2,3 in Abhängigkeit von N")
                graph_sparsity()
                break
            print("Eingabe ungültig. Versuchen Sie es noch einmal...")
        except ValueError:
            print("Eingabe ungültug. Versuchen Sie es noch einmal...")


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
    Returns
    ------
    scipy.sparse.csr_matrix The block_matrix in a sparse data format. """
        if self.sparse is None:
            A=2*self.d
            for l in range(1,self.d+1):
                lstr=[A]*(self.n-1)
                D=sparse.block_diag(lstr)
                r=(self.n-2)*((self.n-1)**(l-1))

                lst=2*[-1*np.ones(r)]
                lste=[(self.n-1)**(l-1),-(self.n-1)**(l-1)]
                E=sparse.diags(lst,lste)
                A=D+E
            self.sparse =sparse.csr_matrix(A)
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

    """ Erstellt ein Plot, welcher für Dimension 1,2,3 die Kondition der
    Blockmatirx in Abhängigkeit von
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
    for counter in range (1, 8):
        rangelist.append(counter*(1700/20))
    for dimension in range (1,4):
        for i in rangelist:
            n=int(i**(float(1)/float(dimension))+1)
            nlist[dimension-1].append(n)
            Nlist[dimension-1].append((n-1)**(dimension))
        for n in nlist[dimension-1]:
            A=BlockMatrix(dimension, n)
            ylist[dimension-1].append(A.get_cond())

    plt.figure()
    plt.plot(Nlist_one, ylist_one, 'b-',label= 'd=1' )
    plt.plot(Nlist_two, ylist_two, 'r-',label= 'd=2' )
    plt.plot(Nlist_three, ylist_three, 'y-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Condition")
    plt.legend()
    plt.show()

def graph_sparsity():

    """ Erstellt ein Plot, welcher für Dimension 1,2,3 die Sparsität der
    Blockmatirx in Abhängigkeit von
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
    for counter in range (1, 8):
        rangelist.append(counter*(1700/20))
    for dimension in range (1,4):
        for i in rangelist:
            n=int(i**(float(1)/float(dimension))+1)
            nlist[dimension-1].append(n)
            Nlist[dimension-1].append((n-1)**(dimension))
        for n in nlist[dimension-1]:
            A=BlockMatrix(dimension, n)
            ylist[dimension-1].append(A.eval_sparsity_lu()[0])

    plt.figure()
    plt.plot(Nlist_one, ylist_one, 'b-',label= 'd=1' )
    plt.plot(Nlist_two, ylist_two, 'r-',label= 'd=2' )
    plt.plot(Nlist_three, ylist_three, 'y-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Sparsity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
