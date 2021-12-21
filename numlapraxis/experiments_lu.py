"""
    Serie 3
    Kurs: Praxisübung Numerische Lineare Algebra
    Programm: experiments_lu
    Authoren: Aron Ventura, Annika Thiele
    Datum: 20.12.2021
    Funktionen:
        graph_errors()
        u()
        f()
        graph()
        graph3D()
        cond_hilmat()
        vgl_cond()
        vgl_spar()
"""

#import sys
import matplotlib.pyplot as plt
import numpy as np
import block_matrix as bl
import linear_solvers as ls
import poisson_problem as pp
from mpl_toolkits import mplot3d

#sys.path.append("/Users/annikathiele/Desktop/numlapraxis")
 #pylint: disable=invalid-name
  #pylint: disable= pointless-string-statement
Kappa = 2
def main():
    """
    Die Main Funktion demonstriert die Funktionalität der Funktionen.
    Returns
    -------
    None
    """
    graph3D()

    #pp.graph_errors(f,u,25)
    """
    print("Wir stellen hiermit unsere Experimente vor.")
    print("Zunächst stellen wir graphisch den Fehler unserer berechneten Lösung,\
dar.")
    print("Wir betrachten hier bis zu 13824 Diskretisierungpunkte in\
ein bis drei Dimensionen.")
    print("Einen kurzen Moment bitte...")
    graph_error()
    print("Als nächstes haben wir die Kondition unserer Matrizen A^(d) mit \
der von dxd-Hilbertmatrizen verglichen.")
    print("Hier die Ergebnisse:")
    vgl_cond()
    print("Es war auch interessant zu sehen wie stark die LU Zerlegung die \
Sparsity beeinflusst hat.")
    print("Im Folgenden sehen Sie die Anzahl an Nicht-Null-Einträgen:")
    vgl_spar()
    print("Außerdem haben wir noch zwei Plots:")
    print("Beide sind in Abhängigkeit von N mit Werten bis N=595 mit \
einer Schrittweite von 85")
    print("1: Ein Plot zur Kondition von A")
    bl.graph_cond()
    print("2: Ein Plot zur Sparsity von A")
    bl.graph_sparsity()
    array=[10,10^2,10^3]
    for dimension in range (1,4):
        for n in range(2,100,10):
            A=bl.BlockMatrix(dimension, n)
            p,l,uu=A.get_lu()
            b = rhs(dimension, n, f)
            hat_u= ls.solve_lu(p,l,uu,b)
            e=compute_error(dimension,n, hat_u, u)
            print(dimension,n,e)
            """




 #pylint: disable=unused-variable
def u(x):
    """
    Diese Funktion berechnet u(x) für ein gegebenes Numpy.array x.
    Parameters
    -------
    x : numpy.array Stellee an der Funktionswert berechnet werden soll
    Returns
    -------
    float Funktionswert an der gegebenen Stelle
    """

    prod = 1
    for counter, value in enumerate(x):
        prod=prod*value* np.sin(Kappa*np.pi*value)
    return prod





def f(x):
    """
    Diese Funktion berechnet f(x) für ein gegebenes Numpy.array x.
    Parameters
    -------
    x : numpy.array Stellee an der Funktionswert berechnet werden soll
    Returns
    -------
    float Funktionswert an der gegebenen Stelle
    """
    laplace=0
    for count, value in enumerate(x):
        prod=1
        for c,v in enumerate(x):
            if c!=count:
                prod=prod*float(v)* np.sin(float(Kappa)*np.pi*float(v))
        prod_one = -prod*value*((Kappa*np.pi)**2)*np.sin(Kappa*np.pi*value)
        prod_two = prod*(1+Kappa*np.pi)*np.cos(Kappa*np.pi*value)
        laplace+=prod_one+prod_two
    return -laplace



def graph():
    """
    Diese Funktion erstellt einen Graphen, der die exakte und approximierte
    Lösung des Poissionproblems für n=4 und n=10 ind der Dimension d=1 darstellt.
    Returns
    -------
    None
    """
    y_list =[]
    for i in np.linspace(0,1,100):
        y_list.append(u([i]))
    A=bl.BlockMatrix(1,10)
    p,l,uu=A.get_lu()
    b = pp.rhs(1,10,f)
    hat_u= ls.solve_lu(p,l,uu,b)
    x_list_u = np.linspace(0, 1, num=9)


    B=bl.BlockMatrix(1,4)
    pq,ll,uuu=B.get_lu()
    bb = pp.rhs(1,4,f)
    hat_uu= ls.solve_lu(pq,ll,uuu,bb)
    x_listt_u = np.linspace(0, 1, num=3)

    plt.plot(np.linspace(0,1,100), y_list, 'b--',label= 'Exakte Lösung' )
    plt.plot(x_list_u, hat_u, 'b-',label= 'Approximierte Lösung für n=10' )
    plt.plot(x_listt_u, hat_uu, 'r-',label= 'Approximierte Lösung für n=4' )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def graph3D():
    """
    Diese Funktion erstellt einen Graphen, der die exakte und approximierte
    Lösung des Poissionproblems für n=4 und n=11 ind der Dimension d=2 darstellt.
    Returns
    -------
    None
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_list = np.linspace(0,1,100)
    y_list = np.linspace(0,1,100)

    X_list, Y_list = np.meshgrid(x_list,y_list)
    z_list = u([X_list,Y_list])

    A=bl.BlockMatrix(2,11)
    p,l,uu=A.get_lu()
    b = pp.rhs(2,11,f)
    hat_u= ls.solve_lu(p,l,uu,b)
    print(hat_u)
    x_list_u = np.linspace(0, 1, num=10)
    y_list_u = np.linspace(0, 1, num=10)
    X_list_u,Y_list_u = np.meshgrid(x_list_u,y_list_u)

    B=bl.BlockMatrix(2,4)
    pq,ll,uuu=B.get_lu()
    bb = pp.rhs(2,4,f)
    hat_uu= ls.solve_lu(pq,ll,uuu,bb)
    x_listt_u = np.linspace(0, 1, num=3)
    y_listt_u = np.linspace(0, 1, num=3)
    X_listt_u,Y_listt_u = np.meshgrid(x_listt_u,y_listt_u)

    ax.plot_wireframe(X_list, Y_list, z_list ,cmap='plasma',label= 'Exakte Lösung' )
    #ax.plot_wireframe(X_list_u, Y_list_u, hat_u, cmap='viridis',label= 'Approximierte Lösung für n=10' )
    #ax.plot_wireframe(X_listt_u, Y_listt_u, hat_uu, cmap=cm.coolwarm ,label= 'Approximierte Lösung für n=4' )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()

def cond_hilmat(n):
    """
    Liefert zu einem gegebenen n die Kondition der HIlbertmatrix der Größe nxn.
    Input
    ______
    n (int) Größe der zu betrachtenden Hilbertmatrix
    Returns
    -------
    float Kondition der Hilbertmatrix der Größe nxn
    """
    cond= lg.norm(lg.hilbert(n), np.inf)*lg.norm(lg.invhilbert(n), np.inf)
    return cond

def vgl_cond():
    """
    Gibt für n=1 bis n=6 die Konditionen der Hilbert und Blockmatritzen der
    jeweiligen Größe aus.
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
    Gibt für verschiedene Dimensionen und n die Sparsität der Blockmatrix und
    der LU Zerlegung aus.
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

if __name__ == "__main__":
    main()
