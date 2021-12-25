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
import scipy.linalg as lg
#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import linear_solvers as ls
import poisson_problem as pp
import block_matrix as bl

#sys.path.append("/Users/annikathiele/Desktop/numlapraxis")
 #pylint: disable=invalid-name
  #pylint: disable= pointless-string-statement

def main():
        """
    Die Main Funktion demonstriert die Funktionalität der Funktionen.
    Returns
    -------
    None
    """
    print("Wir stellen hiermit unsere Experimente vor.")
    print("Zunächst können Sie bestimmen mit welchem Kappa gerechnet werden soll")
    global Kappa
    while True:
        try:
            Kappa = input("Bitte wählen Sie eine ganze Zahl zwischen 1 und 100: ")
            Kappa = int(Kappa)
            if 1<=Kappa<=100:
                break
            print("Keine gültige Zahl, probiere es nochmals ...")
        except ValueError:
            print("Keine gültige Zahl, probiere es nochmals ...")
    print("Es wurde Kappa =", Kappa, "festgelegt.")

    while True:
        print()
        print("Sie können wählen, welche Funktion Sie gezeigt bekommen wollen.")
        print("Ihnen steht folgende Auswahl zur Verfügung:")
        print()
        print("1: Graph zum Fehler der Approximation")
        print("2: Vergleich zu Kondition von Hilbertmatrizen")
        print("3: 3-D Graph zur Approximation der Lösung des Poisson-Problems")
        print("4: Vergleich der Sparsity vor und nach LU-Zerlegung")
        print()
        print("Geben Sie zur Vorstellung einer Funktion die ihr zugewiesene Nummer ein")
        while True:
            try:
                zahl = input("Bitte wählen Sie eine ganze Zahl zwischen 1 und 4: ")
                zahl = int(zahl)
                if 1<=zahl<=4:
                    break
                print("Keine gültige Zahl, probiere es nochmals ...")
            except ValueError:
                print("Keine gültige Zahl, probiere es nochmals ...")
        print()
        print("Ihre Eingabe war erfolgreich!")
        Functions = {
            1: m_errors,
            2: m_cond,
            3: m_3d,
            4: m_sparsity,
        }
        Functions[zahl]()
        print()
        print("Möchten Sie eine weitere Funktion testen?")
        print('Für "ja" geben Sie bitte 0 ein')
        print("Jede andere Eingabe beendet das Programm")
        try:
            weiter = input("Weitermachen?: ")
            weiter = int(weiter)
            if weiter!=0:
                break
        except ValueError:
            break


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

def graph_errors(f,u,n):
    """
    Diese Funktion erstellt einen Graphen, der die Fehler der approximierten
    Lösung für das Poisson Problem in Abhängigkeit von den
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
    Nlist=[]
    Nlistt=[]
    Nlisttt=[]

    ylist=[]
    for x in range(2, (n-1)**3+2):
        A=bl.BlockMatrix(1, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(1, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(1,x, hat_u, u)
        ylist.append(e)
        Nlist.append(x-1)

    ylistt=[]
    for x in range(2,int((n-1)**(3/2)+2)):
        A=bl.BlockMatrix(2, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(2, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(2,x, hat_u, u)
        ylistt.append(e)
        Nlistt.append((x-1)**2)

    ylisttt=[]
    for x in range(2,n+1):
        A=bl.BlockMatrix(3, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(3, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(3,x, hat_u, u)
        ylisttt.append(e)
        Nlisttt.append((x-1)**3)

    """
    ylistt=[]
    for x in range(2,int((n)**(1/2)+1)):
        A=bl.BlockMatrix(2, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(2, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(2,x, hat_u, u)
        ylistt.append(e)
        Nlistt.append((x-1)**2)

    ylisttt=[]
    for x in range(2,int((n)**(1/3)+1)):
        A=bl.BlockMatrix(3, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(3, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(3,x, hat_u, u)
        ylisttt.append(e)
        Nlisttt.append((x-1)**3)
    """

    plt.plot(Nlist, ylist, 'b-',label= 'd=1' )
    plt.plot(Nlistt, ylistt, 'g-',label= 'd=2' )
    plt.plot(Nlisttt, ylisttt, 'r-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def graph_er():
    """
    Diese Funktion erstellt einen Graphen, welcher zum einen die Funktion u auf dem Intervall
    (0,1) und zum anderen die Approximierte Lösung für Ax=f+10^(-3)r.
    Returns
    -------
    None

    """

    zero=np.array([0])
    y_list =[]
    for i in np.linspace(0,1,300):
        y_list.append(u([i]))
    A=bl.BlockMatrix(1,200)
    p,l,uu=A.get_lu()
    b = pp.rhs_er(1,200,f)
    hat_u= ls.solve_lu(p,l,uu,b)
    hat_u=np.append(zero,hat_u)
    hat_u=np.append(hat_u,zero)
    x_list_u = np.linspace(0, 1, num=201)
    plt.plot(np.linspace(0,1,300), y_list, 'b--',label= 'Exakte Lösung' )
    plt.plot(x_list_u, hat_u, 'b-',label= 'Approximierte Lösung für n=200' )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



def graph():
    """
    Diese Funktion erstellt einen Graphen, der die exakte und approximierte
    Lösung des Poissionproblems für n=100 und n=10 ind der Dimension d=1 darstellt.
    Returns
    -------
    None
    """

    zero=np.array([0])
    y_list =[]
    for i in np.linspace(0,1,300):
        y_list.append(u([i]))
    A=bl.BlockMatrix(1,10)
    p,l,uu=A.get_lu()
    b = pp.rhs(1,10,f)
    hat_u= ls.solve_lu(p,l,uu,b)
    hat_u=np.append(zero,hat_u)
    hat_u=np.append(hat_u,zero)
    x_list_u = np.linspace(0, 1, num=11)




    B=bl.BlockMatrix(1,100)
    pq,ll,uuu=B.get_lu()
    bb = pp.rhs(1,100,f)
    hat_uu= ls.solve_lu(pq,ll,uuu,bb)
    hat_uu=np.append(zero,hat_uu)
    hat_uu=np.append(hat_uu,zero)
    x_listt_u = np.linspace(0, 1, num=101)




    plt.plot(np.linspace(0,1,300), y_list, 'b--',label= 'Exakte Lösung' )
    plt.plot(x_list_u, hat_u, 'r-',label= 'Approximierte Lösung für n=10' )
    plt.plot(x_listt_u, hat_uu, 'b-',label= 'Approximierte Lösung für n=100' )
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
    hat_u_2D=[]
    for i in range(10):
        array=[]
        for j in range(10):
            array.append(hat_u[10*i+j])
        hat_u_2D.append(array)
    hat_u_array= np.array(hat_u_2D)
    x_list_u = np.linspace(0, 1, num=10)
    y_list_u = np.linspace(0, 1, num=10)
    X_list_u,Y_list_u = np.meshgrid(x_list_u,y_list_u)

    B=bl.BlockMatrix(2,4)
    pq,ll,uuu=B.get_lu()
    bb = pp.rhs(2,4,f)
    hat_uu= ls.solve_lu(pq,ll,uuu,bb)
    hat_uu_2D=[]
    for i in range(3):
        array=[]
        for j in range(3):
            array.append(hat_uu[3*i+j])
        hat_uu_2D.append(array)
    hat_uu_array= np.array(hat_uu_2D)
    x_listt_u = np.linspace(0, 1, num=3)
    y_listt_u = np.linspace(0, 1, num=3)
    X_listt_u,Y_listt_u = np.meshgrid(x_listt_u,y_listt_u)

    ax.plot_wireframe(X_list, Y_list, z_list ,label= 'Exakte Lösung' )
    ax.plot_wireframe(X_list_u, Y_list_u, hat_u_array, color='green' ,\
label= 'Approximierte Lösung für n=11' )
    ax.plot_wireframe(X_listt_u, Y_listt_u, hat_uu_array, color='red',\
label= 'Approximierte Lösung für n=4' )
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

def m_errors():
    """
    Stellt die Funktion graph_errors() für n=8 vor
    Parameter
    ------
    None
    Returns
    -------
    None
    """
    print("Wir stellen graphisch den Fehler unserer berechneten Lösung, dar.")
    print("Dazu betrachten wir hier bis zu 13824 Diskretisierungpunkte in \
ein bis drei Dimensionen.")
    print("Einen kurzen Moment bitte...")
    graph_errors(f,u,8)

def m_cond():
    """
    Stellt die Funktion vgl_cond() vor
    Parameter
    ------
    None
    Returns
    -------
    None
    """
    print("Wir vergleichen die Kondition unserer Koeffizientenmatrizen mit \
der von Hilbertmatrizen.")
    print("Hier die Ergebnisse:")
    vgl_cond()

def m_3d():
    """
    Stellt die Funktion graph3D() vor
    Parameter
    ------
    None
    Returns
    -------
    None
    """
    print("Wir stellen in einem 3D-Plot die Approximation der Lösung des \
Poisson-Problems graphisch dar.")
    graph3D()

def m_sparsity():
    """
    Stellt die Funktion vgl_spar() vor
    Parameter
    ------
    None
    Returns
    -------
    None
    """
    print("Es war interessant zu sehen wie stark die LU Zerlegung die \
Sparsity beeinflusst hat.")
    print("Im Folgenden sehen Sie die Anzahl an Nicht-Null-Einträgen:")
    vgl_spar()
         
if __name__ == "__main__":
    main()
