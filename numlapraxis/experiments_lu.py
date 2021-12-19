import sys
sys.path.append("/Users/annikathiele/Desktop/numlapraxis")
import block_matrix as bl
import linear_solvers as ls
import poisson_problem as pp
import numpy as np
import matplotlib.pyplot as plt

Kappa = 1




def main():


    graph_errors()
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
    bl.vgl_cond()
    print("Es war auch interessant zu sehen wie stark die LU Zerlegung die \
Sparsity beeinflusst hat.")
    print("Im Folgenden sehen Sie die Anzahl an Nicht-Null-Einträgen:")
    bl.vgl_spar()
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
def graph_errors():

    """
    nlist =[]

    ylist=[]
    Nlist=[]
    for x in range(2,200):
        A=bl.BlockMatrix(1, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(1, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(1,x, hat_u, u)
        ylist.append(e)
        Nlist.append(x-1)

    ylistt=[]
    Nlistt=[]
    for x in range(3,200):
        A=bl.BlockMatrix(2, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(2, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(2,x, hat_u, u)
        ylistt.append(e)
        Nlistt.append((x-1)**2)
    """
    ylisttt=[]
    Nlisttt=[]
    for x in range(3,18):
        A=bl.BlockMatrix(3, x)
        p,l,uu=A.get_lu()
        b = pp.rhs(3, x, f)
        hat_u= ls.solve_lu(p,l,uu,b)
        e=pp.compute_error(3,x, hat_u, u)
        ylisttt.append(e)
        Nlisttt.append((x-1)**3)

    plt.figure()
    #plt.plot(Nlist, ylist, 'b-',label= 'd=1' )
    #plt.plot(Nlistt, ylistt, 'b-',label= 'd=2' )
    plt.plot(Nlisttt, ylisttt, 'b-',label= 'd=3)' )

    plt.title("Error")
    plt.legend()
    plt.show()

def u(x):
    prod = 1
    for i in range(len(x)):
        prod=prod*x[i]* np.sin(Kappa*np.pi*x[i])
    return prod

def f(x):
    sum=0
    for i in range(len(x)):
        prod=1
        for k in range(len(x)):
            if k!=i:
                prod=prod*float(x[k])* np.sin(float(Kappa)*np.pi*float(x[k]))
        prod_one = prod*np.sin(Kappa*np.pi*x[i])
        prod_two = prod*x[i]*np.cos(Kappa*np.pi*x[i])*Kappa*np.pi
        sum+=prod_one+prod_two
    return sum
"""
def graph():
    x_list_u = np.linspace(-100, 100, num=50)
    y_list_u =[]
    for i in x_list_u:
        y_list_u.append(u(i,2))
    plt.plot(x_list_u, y_list_u, 'b-',label= 'u(x)' )
    plt.legend()
    plt.show()
"""
if __name__ == "__main__":
    main()
