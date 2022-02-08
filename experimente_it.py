"""
    Serie 5
    Kurs: Praxisübung Numerische Lineare Algebra
    Programm: experimente_it
    Authoren: Aron Ventura, Annika Thiele
    Datum: 06.02.2022
    Funktionen:
        main()
        get_n()
        graph_error()
        graph_eps()
        graph_omega()
        graph_error_lu()
        graph_time()
        graph_sparsity()
        time_solve()
        time_solve_sor_comp()
        time_solve_sor()
        u()
        f()
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import linear_solversfinal as ls
import poisson_problemfinal as pp
import block_matrixfinal as bl
#pylint: disable=invalid-name

def main():
    """
    Die Main Funktion demonstriert die Funktionalität der Funktionen.
    Returns
    -------
    None
    Exceptions
    -------
    ValueError
    """
    print("Wir stellen hiermit unsere Experimente vor.")
    while True:
        print()
        print("Sie können wählen, welchen Graphen Sie gezeigt bekommen wollen.")
        print("Ihnen steht folgende Auswahl zur Verfügung:")
        print()
        print("1: Plot zum Fehler des iterativen Verfahrens")
        print("2: Plot zu unterschiedlichen Epsilon als Abbruchkriterium")
        print("3: Plot zum Fehler in Abhängigkeit von Omega")
        print("4: Plot zum Fehler des SOR- und des LU-Verfahrens")
        print("5: Vergleich der Laufzeit des SOR- und des LU-Verfahrens")
        print("6: Vergleich der Sparsität der beiden Verfahren")

        print()
        print("Geben Sie zur Vorstellung einer Funktion die ihr zugewiesene Nummer ein")
        while True:
            try:
                zahl = input("Bitte wählen Sie eine ganze Zahl zwischen 1 und 6: ")
                zahl = int(zahl)
                if 1<=zahl<=6:
                    break
                print("Keine gültige Zahl, probiere es nochmals ...")
            except ValueError:
                print("Keine gültige Zahl, probiere es nochmals ...")
        print()
        print("Ihre Eingabe war erfolgreich!")
        Functions = {
            1: graph_error,
            2: graph_eps,
            3: graph_omega,
            4: graph_error_lu,
            5: graph_time,
            6: graph_sparsity,

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

def get_n(x):
    """
    Lässt Nutzer ein n wählen
    Parameter
    ------
    x : int
    Maximal wählbares n
    Returns
    -------
    n : int ausgewähltes n
    Exceptions
    -------
    ValueError
    Falls keine gültige Zahl eingegeben wurde
    """
    print("Wählen Sie als zunächst ein n")
    while True:
        try:
            print("Bitte wählen Sie eine ganze Zahl zwischen 3 und ",x)
            n = input("Eingabe: " )
            n = int(n)
            if 3<=n<=x:
                break
            print("Keine gültige Zahl, probiere es nochmals ...")
        except ValueError:
            print("Keine gültige Zahl, probiere es nochmals ...")
    return n



def graph_error():
    """
    Zunächst kann der Nutzer ein n zwischen 3 und 18 auswählen.
    Diese Funktion erstellt einen Graphen, der für d=1,2,3 den maximalen Fehler
    der Iterierten darstellt.
    Returns
    -------
    None
    """
    n=get_n(18)
    ylist=[]
    x1=(n-1)**3+1
    A=bl.BlockMatrix(1, x1)
    A=A.get_sparse()
    b = pp.rhs(1, x1, f)
    x0=np.ones(b.size)
    xx=ls.solve_sor(A,b,x0)[1]
    print("Abruchkriterium: ",ls.solve_sor(A,b,x0)[0])
    for hat_u in xx:
        e=pp.compute_error(1,x1, hat_u, u)
        ylist.append(e)
    xlist=range(len(xx))

    ylistt=[]
    x2=int((n-1)**(3/2)+1)
    A=bl.BlockMatrix(2,x2 )
    A=A.get_sparse()
    b = pp.rhs(2, x2, f)
    x0=np.ones(b.size)
    xx= ls.solve_sor(A,b,x0)[1]
    print("Abruchkriterium: ",ls.solve_sor(A,b,x0)[0])
    for hat_u in xx:
        e=pp.compute_error(2,x2, hat_u, u)
        ylistt.append(e)
    xlistt=range(len(xx))

    ylisttt=[]
    A=bl.BlockMatrix(3, n)
    A=A.get_sparse()
    b = pp.rhs(3, n, f)
    x0=np.ones(b.size)
    xx= ls.solve_sor(A,b,x0)[1]
    print("Abruchkriterium: ",ls.solve_sor(A,b,x0)[0])

    for hat_u in xx:
        e=pp.compute_error(3,n, hat_u, u)
        ylisttt.append(e)
    xlisttt=range(len(xx))

    plt.figure()
    plt.ylabel("Fehler")
    plt.xlabel("Iteration")
    #plt.plot(xlist, xlist, 'r--',label="n")
    plt.plot(xlist, ylist, 'b-',label= 'd=1' )
    plt.plot(xlistt, ylistt, 'y-',label= 'd=2' )
    plt.plot(xlisttt, ylisttt, 'm-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error")
    plt.legend()
    plt.show()

def graph_eps():
    """
    Zunächst kann der Nutzer ein n zwischen 3 und 47 auswählen.
    Diese Funktion erstellt einen Graphen, der für d=2 und eps=h^k, k=-2,0,2,4,6
    das konvergenzverhalten darstellt.
    Returns
    -------
    None
    """
    n=get_n(47)
    plt.figure()
    xlist =[]
    for i in range(2,n,5):
        xlist.append(i)
    ylist=[]
    for k in range(-2,8,2):
        yy=[]
        for x in xlist:
            A=bl.BlockMatrix(2, x)
            A=A.get_sparse()
            b = pp.rhs(2, x, f)
            x0=np.ones(b.size)
            epsi=(1/n)**k
            params=dict(eps=epsi, max_iter=5000, min_red=1e-7, omega=0.6)
            hat_u=ls.solve_sor(A,b,x0,params)[1].pop()
            e=pp.compute_error(2,x, hat_u, u)
            yy.append(e)

        ylist.append(yy)

    xlist=np.array(xlist)
    xlistt=np.ones(len(xlist))*0.03
    plt.plot(xlist, xlistt, 'k--', label='f(x)=0.03')
    plt.plot(xlist, ylist[0], 'b-',label= 'eps=h^-2' )
    plt.plot(xlist, ylist[1], 'r-',label= 'eps=h^0' )
    plt.plot(xlist, ylist[2], 'y-',label= 'eps=h^2' )
    plt.plot(xlist, ylist[3], 'g-',label= 'eps=h^4' )
    plt.plot(xlist, ylist[4], 'm-',label= 'eps=h^6' )
    plt.ylabel("Fehler")
    plt.xlabel("n")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Epsilon")
    plt.legend()
    plt.show()

def graph_omega():
    """
    Zunächst kann der Nutzer ein n zwischen 3 und 15 auswählen.
    Diese Funktion erstellt einen Graphen, der den Fehler in Abhängigkeit von
    Omega darstellt. Wir betrachten d=2.
    Returns
    -------
    None
    """
    n=get_n(15)
    plt.figure()
    xlist =[]
    for i in range(2,21):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(2, n)
        A=A.get_sparse()
        b = pp.rhs(2, n, f)
        x0=np.ones(b.size)
        ome=1.8/20*x
        params=dict(eps=1e-8, max_iter=5000, min_red=1e-7, omega=ome)
        xx=ls.solve_sor(A,b,x0,params)
        hat_u=xx[1].pop()
        print("Abrruchkriterium: ",xx[0])
        e=pp.compute_error(2,n, hat_u, u)
        ylist.append(e)

    xlist=np.array(xlist)*(1.8/20)

    plt.plot(xlist, ylist, 'b-',label= 'omega' )
    plt.ylabel("Fehler")
    plt.xlabel("Omega")
    #plt.xscale("log")
    plt.yscale("log")
    plt.title("Omega")
    plt.legend()
    plt.show()

def graph_error_lu():#pylint: disable=too-many-locals
    """
    Zunächst kann der Nutzer ein n zwischen 3 und 40 auswählen.
    Diese Funktion erstellt einen Graphen, der für d=1 das Konvergenzverhalten
    des LU und SOR-Verfahrens vergleicht. Das SOR-Verfahren läuft
    einmal mit max_iter=1000 und einmal mit max_iter=4000
    Returns
    -------
    None
    """
    n=get_n(40)
    xlist =[]
    for i in range(2,n):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(1, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(1,x, hat_u, u)
        ylist.append(e)

    ylistt=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        A=A.get_sparse()
        b = pp.rhs(1, x, f)
        x0=np.ones(b.size)
        params=dict(eps=1e-6, max_iter=3000, min_red=1e-7, omega=0.1)
        hat_u= ls.solve_sor(A,b,x0,params)[1].pop()
        e=pp.compute_error(1,x, hat_u, u)
        ylistt.append(e)

    ylisttt=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        A=A.get_sparse()
        b = pp.rhs(1, x, f)
        x0=np.ones(b.size)
        params=dict(eps=1e-6, max_iter=10000, min_red=1e-7, omega=0.1)
        hat_u= ls.solve_sor(A,b,x0,params)[1].pop()
        e=pp.compute_error(1,x, hat_u, u)
        ylisttt.append(e)

    plt.figure()
    plt.ylabel("Fehler")
    plt.xlabel("n")
    #plt.plot(xlist, xlist, 'r--',label="n")
    plt.plot(xlist, ylist, 'b-',label= 'LU' )
    plt.plot(xlist, ylistt, 'y--',label= 'SOR max_iter=3000' )
    plt.plot(xlist, ylisttt, 'r--',label= 'SOR max_iter=10000' )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Fehler LU/SOR")
    plt.legend()
    plt.show()

def graph_time():
    """
    Zunächst kann der Nutzer ein n zwischen 3 und 10 auswählen.
    Diese Funktion erstellt einen Graphen, der die Laufzeit des
    LU- und des SOR-Verfahrens komponentenweise und mit
    solve-triangular vergleicht.
    Returns
    -------
    None
    """
    x=get_n(10)
    xlist=[]
    ylist=[]
    ylistt=[]
    ylistt2=[]
    for n in range(2,x):
        print(n)
        xlist.append(n)
        tim=0
        timee=0
        timee2=0
        for _ in range(15):
            tim+=time_solve(n,1)
            timee+=time_solve_sor_comp(n,1)
            timee2+=time_solve_sor(n,1)
        ylist.append(tim/15)
        ylistt.append(timee/15)
        ylistt2.append(timee2/15)
    xlistt=np.array(xlist)
    xlistt=xlistt/100
    plt.figure()
    plt.ylabel("Zeit in ms")
    plt.xlabel("n")
    plt.plot(xlist, ylist, 'b-',label= 'LU Laufzeit' )
    plt.plot(xlist, ylistt2, 'r-',label= 'SOR mit solve_triangular() Laufzeit' )
    plt.plot(xlist, ylistt, 'r--',label= 'SOR komponentenweise Laufzeit' )
    plt.plot(xlist, xlistt, 'g--', label='f(x)=x/100')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Laufzeit in Abhängigkeit von n")
    plt.legend()
    plt.show()

def graph_sparsity():# pylint: disable-msg=too-many-locals

    """ Erstellt ein Plot, welcher für Dimension 1,2,3 die Sparsität der
    Blockmatirx in Abhängigkeit von
    der relativen Anzahl an inneren Diskretisierungspunkten abbildet.
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

    zlist_one = []
    zlist_two = []
    zlist_three = []
    zlist = [zlist_one , zlist_two , zlist_three]

    rangelist = []
    for counter in range (1, 8):
        rangelist.append(counter*(85))
    for dimension in range (1,4):
        for i in rangelist:
            n=int(i**(float(1)/float(dimension))+1)
            nlist[dimension-1].append(n)
            Nlist[dimension-1].append((n-1)**(dimension))
        for n in nlist[dimension-1]:
            A=bl.BlockMatrix(dimension, n)
            zlist[dimension-1].append(A.eval_sparsity()[1])
            ylist[dimension-1].append(A.eval_sparsity_lu()[1])

    plt.figure()
    Nlist_one=np.array(Nlist_one)
    Nlist_onee=1/Nlist_one
    print(Nlist_onee)
    Nlist_oneee=1/(Nlist_one**(1/2))
    ylist_one=np.array(ylist_one)*1.03
    plt.plot(Nlist_one, ylist_one, 'c-',label= 'd=1, LU' )
    plt.plot(Nlist_two, ylist_two, 'c--',label= 'd=2, LU' )
    plt.plot(Nlist_three, ylist_three, 'c-.',label= 'd=3 LU' )
    plt.plot(Nlist_one, Nlist_oneee, 'dimgrey', label='f(x)=1/sqrt(x)')
    plt.plot(Nlist_one, zlist_one, 'r-',label= 'd=1 SOR' )
    plt.plot(Nlist_two, zlist_two, 'r--',label= 'd=2 SOR' )
    plt.plot(Nlist_three, zlist_three, 'r-.', label = 'd=3 SOR')
    plt.plot(Nlist_one, Nlist_onee, 'k-', label='f(x)=1/x')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("relative Anzahl Nicht-Null-Einträge")
    plt.title("Sparsity")
    plt.legend()
    plt.show()



def time_solve(n,d):
    """ Diese Funktion gibt die Zeit an, die das Programm zum berechnen der Lösung benötigt.
    Returns
    -------
    None
    """
    start=time.time()
    A=bl.BlockMatrix(d,n)
    p,l,uu=A.get_lu()
    b=pp.rhs(d,n,f)
    ls.solve_lu(p,l,uu,b)
    end=time.time()
    return end-start

def time_solve_sor_comp(n,d):
    """ Diese Funktion gibt die Zeit an, die das Programm zum berechnen der Lösung benötigt.
    Returns
    -------
    None
    """
    start=time.time()
    A=bl.BlockMatrix(d,n)
    A= A.get_sparse()
    b=pp.rhs(d,n,f)
    ls.solve_sor_comp(A,b,b)
    end=time.time()
    return end-start

def time_solve_sor(n,d):
    """ Diese Funktion gibt die Zeit an, die das Programm zum berechnen der Lösung benötigt.
    Returns
    -------
    None
    """
    start=time.time()
    A=bl.BlockMatrix(d,n)
    A= A.get_sparse()
    b=pp.rhs(d,n,f)
    ls.solve_sor(A,b,b)
    end=time.time()
    return end-start

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
    for _, value in enumerate(x):
        prod=prod*value* np.sin(np.pi*value)
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
                prod=prod*float(v)* np.sin(np.pi*float(v))
        prod_one = -prod*value*((np.pi)**2)*np.sin(np.pi*value)
        prod_two = prod*(1+np.pi)*np.cos(np.pi*value)
        laplace+=prod_one+prod_two
    return -laplace

def graph_error2():
    """
    Diese Funktion erstellt einen Graphen, den Fehler in
    Abhängigkeit von n darstellt. n kann vom Nutzer ausgewählt werden.
    Returns
    -------
    None
    """
    n=get_n(15)
    xlist =[]
    for i in range(2,n):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        A=A.get_sparse()
        b = pp.rhs(1, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)[1].pop()
        e=pp.compute_error(1,x, hat_u, u)
        ylist.append(e)

    ylistt=[]
    for x in xlist:
        A=bl.BlockMatrix(2, x)
        A=A.get_sparse()
        b = pp.rhs(2, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)[1].pop()
        e=pp.compute_error(2,x, hat_u, u)
        ylistt.append(e)

    ylisttt=[]
    for x in xlist:
        A=bl.BlockMatrix(3, x)
        A=A.get_sparse()
        b = pp.rhs(3, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)[1].pop()
        e=pp.compute_error(3,x, hat_u, u)
        ylisttt.append(e)
    print("fertig")

    plt.figure()
    plt.ylabel("Fehler")
    plt.xlabel("n")
    #plt.plot(xlist, xlist, 'r--',label="n")
    plt.plot(xlist, ylist, 'b-',label= 'd=1' )
    plt.plot(xlist, ylistt, 'y-',label= 'd=2' )
    plt.plot(xlist, ylisttt, 'm-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error")
    plt.legend()
    plt.show()

def graph():# pylint: disable-msg=too-many-locals
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
    A=bl.BlockMatrix(1, 50)
    A=A.get_sparse()
    b = pp.rhs(1, 50, f)

   # hat_u=sp.linalg.spsolve(A,b)

    x0=np.ones(b.size)
    hat_u= ls.solve_sor(A,b,x0)[1].pop()
    print(hat_u)

    hat_u=np.append(zero,hat_u)
    hat_u=np.append(hat_u,zero)
    x_list_u = np.linspace(0, 1, num=51)

    B=bl.BlockMatrix(1,50)
    pq,ll,uuu=B.get_lu()
    bb = pp.rhs(1,50,f)
    hat_uu= ls.solve_lu(pq,ll,uuu,bb)
    print(hat_uu)
    hat_uu=np.append(zero,hat_uu)
    hat_uu=np.append(hat_uu,zero)
    x_listt_u = np.linspace(0, 1, num=51)

    plt.plot(np.linspace(0,1,300), y_list, 'b--',label= 'Exakte Lösung' )
    plt.plot(x_list_u, hat_u, 'r-',label= 'Approximierte Lösung für n=10' )
    plt.plot(x_listt_u, hat_uu, 'b-',label= 'Approximierte Lösung für n=100' )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def graph2():# pylint: disable-msg=too-many-locals
    """
    Diese Funktion erstellt einen Graphen, der die exakte und approximierte
    Lösung des Poissionproblems für n=100 und n=10 ind der Dimension d=1 darstellt.
    Returns
    -------
    None
    """

    zero=np.array([0])
    y_list =[]
    for i in np.linspace(0,1,100):
        y_list.append(u([i]))
    A=bl.BlockMatrix(1, 30)
    A=A.get_sparse()
    b = pp.rhs(1, 30, f)
    x0=np.ones(b.size)
    params=dict(eps=1e-8, max_iter=5000, min_red=1e-7, omega=0.3)
    hat_u= ls.solve_sor(A,b,x0,params)[1].pop()

    hat_u=np.append(zero,hat_u)
    hat_u=np.append(hat_u,zero)
    x_list_u = np.linspace(0, 1, num=31)

    A=bl.BlockMatrix(1, 50)
    p,l,uu=A.get_lu()
    b = pp.rhs(1, 50, f)
    hat_uu= ls.solve_lu(p,l,uu,b)
    print(hat_uu)
    hat_uu=np.append(zero,hat_uu)
    hat_uu=np.append(hat_uu,zero)
    x_listt_u = np.linspace(0, 1, num=51)

    plt.plot(np.linspace(0,1,100), y_list, 'b--',label= 'Exakte Lösung' )
    plt.plot(x_list_u, hat_u, 'r-',label= 'Approximierte Lösung für n=10' )
    plt.plot(x_listt_u, hat_uu, 'b-',label= 'Approximierte Lösung für n=100' )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
