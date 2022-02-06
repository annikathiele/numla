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
import block_matrixfinal as bl
import scipy.sparse as sp
 #pylint: disable=invalid-name

def mainold():
    """
    The main demonstrates the fundtionality od the functions.
    Returns
    -------
    None
    """
    print("Mithilfe dieses Moduls kann man mit einer gegebenen LU-Zerlegung \
für A Gleichungssysteme der Form Ax = b nach x auflösen")
    print()
    print("Sei in diesem Beispiel A die Koeffizienten Matrix")
    d,n= bl.get_dn()
    print("Sei b=[1,1,...,1]")
    b=np.ones((n-1)**d)
    A= bl.BlockMatrix(d,n)
    p,l,u= A.get_lu()
    xx=solve_lu(p,l,u,b)[1]
    x=xx.pop()
    print()
    print("Dann löst", x," unsere Gleichung.")

def main():
    """
    The main demonstrates the fundtionality od the functions.
    Returns
    -------
    None
    """
    print("Mithilfe dieses Moduls kann man mit einer gegebenen LU-Zerlegung \
für A Gleichungssysteme der Form Ax = b nach x auflösen")
    print()
    print("Sei in diesem Beispiel A die Koeffizienten Matrix")
    d,n= bl.get_dn()
    print("Sei b=[1,1,...,1]")
    b=np.ones((n-1)**d)
    A= bl.BlockMatrix(d,n)
    A=A.get_sparse()
    x=solve_sor2(A,b,b)[1].pop()
    residual=solve_sor2(A,b,b)[2].pop()
    print()
    print("Dann löst",x," unsere Gleichung")
    print("Das Residuum für unsere Lösung ist" , residual)

def solve_sor(A, b, x0, eps=1e-8, max_iter=1000, min_red=1e-7, omega=0.3):
    """ Solves the linear system Ax = b via the successive over relaxation method.
    Parameters
    ---------
    A : scipy.sparse.csr_matrix system matrix of the linear system
    b : numpy.ndarray right-hand-side of the linear system

    x0 : numpy.ndarray initial guess of the solution
    params : dict, optional
        dictionary containing termination conditions

        eps : float tolerance for the norm of the residual in the infinity norm. If set less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int maximal number of iterations that the solver will perform. If set less or equal to 0 no constraint on the number of iterations is imposed.
        min_red : float minimal reduction of the residual in the infinity norm in every step. If set less or equal to 0 no constraint on the norm of the reduction of the residual is imposed.

    omega : float, optional relaxation parameter
    Returns
    ------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray)
        iterates of the algorithm. First entry is ‘x0‘.
    list (of float)
        infinity norm of the residuals of the iterates
    Raises
    -----
    ValueError
        If no termination condition is active, i.e., ‘eps=0‘ and ‘max_iter=0‘, etc.
    """
    if eps==0 or max_iter==0 or min_red==0:
        print("no termination condition is active")
        return
    termination=""
    residual = np.linalg.norm(A*x0-b)
    x_new=x0
    counter=0
    x=[x_new]
    residuals=[residual]
    while residual > eps:
        x_old=np.copy(x_new)
        for i in range(sp.csr_matrix(A).shape[0]):
            sigma = 0
            for j in range(sp.csr_matrix(A).shape[1]):
                if j < i:
                    sigma += (A[i,j] * x_new[j])
                if j > i:
                    sigma += (A[i,j] * x_old[j])
            x_new[i] = (1 - omega) * x_old[i] + ((omega / A[i,i]) * (b[i] - sigma))
        x.append(x_new)
        new_residual = np.linalg.norm(A*x_new-b)
        residuals.append(new_residual)
        #print(new_residual)
        if abs(new_residual-residual)<min_red:
            termination="min_red"
            break
        residual=new_residual
        counter+=1
        if counter== max_iter:
            termination="max_iter"
            break
        else:
            termination="eps"
    return termination,x,residuals

def solve_sor2(A, b, x0, eps=1e-8, max_iter=1500, min_red=1e-7, omega=0.6):
    """ Solves the linear system Ax = b via the successive over relaxation method.
    Parameters
    ---------
    A : scipy.sparse.csr_matrix system matrix of the linear system
    b : numpy.ndarray right-hand-side of the linear system

    x0 : numpy.ndarray initial guess of the solution
    params : dict, optional
        dictionary containing termination conditions

        eps : float tolerance for the norm of the residual in the infinity norm. If set less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int maximal number of iterations that the solver will perform. If set less or equal to 0 no constraint on the number of iterations is imposed.
        min_red : float minimal reduction of the residual in the infinity norm in every step. If set less or equal to 0 no constraint on the norm of the reduction of the residual is imposed.

    omega : float, optional relaxation parameter
    Returns
    ------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray)
        iterates of the algorithm. First entry is ‘x0‘.
    list (of float)
        infinity norm of the residuals of the iterates
    Raises
    -----
    ValueError
        If no termination condition is active, i.e., ‘eps=0‘ and ‘max_iter=0‘, etc.
    """
    if eps==0 or max_iter==0 or min_red==0:
        print("no termination condition is active")
        return
    termination=""
    residual = np.linalg.norm(A*x0-b)
    x_new=x0
    counter=0
    L,D,U=get_LDU(A)
    x=[x_new]
    residuals=[residual]
    while residual > eps:
        rhs=(omega*U+(1-omega)*D)*x_new
        rhs+=omega*b
        lhs=D-omega*L
        x_new=sp.linalg.spsolve_triangular(lhs,rhs)
        x.append(x_new)
        new_residual = np.linalg.norm(A*x_new-b)
        residuals.append(new_residual)
        if abs(new_residual-residual)<min_red:
            termination="min_red"
            break
        residual=new_residual
        counter+=1
        if counter== max_iter:
            termination="min_red"
            break
        else:
            termination="eps"
    return termination,x,residuals

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

def get_LDU(A):
    A=sp.csr_matrix(A)
    D=A.diagonal()
    D=sp.diags(D)
    U=sp.triu(A, k=1)
    L_t=sp.triu(A.transpose(), k=1)
    L=L_t.transpose()
    return L,D,U

"""
def main():
    A=np.matrix([[1,3,5],[3,7,2],[4,3,7]])
    A=sp.csr_matrix(A)
    b=np.array([4,3,5])
    #b=sp.csr_matrix(b)
    x0=np.array([1,1,2],dtype=float)
    #x0=sp.csr_matrix(x0)
    print(sp.linalg.spsolve(A,b))
    print(solve_sor(A,b,x0))
"""

if __name__ == "__main__":
    main()

