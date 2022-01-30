import block_matrix as bl
import linear_solvers as ls
import poisson_problem as pp
import numpy as np
import matplotlib.pyplot as plt

def main():
    graph_error()

def graph_eps():
    plt.figure()
    xlist =[]
    for i in range(2,11):
        xlist.append(i)
    ylist=[]
    for k in range(-2,8,2):
        yy=[]
        for x in xlist:
            A=bl.BlockMatrix(2, x)
            A=A.get_sparse()
            b = pp.rhs(2, x, f)
            x0=np.ones(b.size)
            epsi=(1/x)**k
            hat_u= ls.solve_sor(A,b,x0,eps=epsi)
            e=pp.compute_error(1,x, hat_u, u)
            yy.append(e)
        ylist.append(yy)

    plt.plot(xlist, ylist[0], 'b-',label= 'eps=-2' )
    plt.plot(xlist, ylist[0], 'b-',label= 'eps=0' )
    plt.plot(xlist, ylist[0], 'b-',label= 'eps=2' )
    plt.plot(xlist, ylist[0], 'b-',label= 'eps=4' )
    plt.plot(xlist, ylist[0], 'b-',label= 'eps=6' )

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Epsilon")
    plt.legend()
    plt.show()

def graph_omega():
    plt.figure()
    xlist =[]
    for i in range(2,11):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(2, x)
        A=A.get_sparse()
        b = pp.rhs(2, x, f)
        x0=np.ones(b.size)
        ome=2/10*x
        hat_u= ls.solve_sor(A,b,x0,omega=ome)
        e=pp.compute_error(1,x, hat_u, u)
        yy.append(e)
    ylist.append(yy)

    plt.plot(xlist, ylist, 'b-',label= 'omega' )

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Omega")
    plt.legend()
    plt.show()

def graph_error():
    xlist =[]
    for i in range(2,11):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        A=A.get_sparse()
        b = pp.rhs(1, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)
        e=pp.compute_error(1,x, hat_u, u)
        ylist.append(e)
    print("fertig")

    ylistt=[]
    for x in xlist:
        A=bl.BlockMatrix(2, x)
        A=A.get_sparse()
        b = pp.rhs(2, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)
        e=pp.compute_error(2,x, hat_u, u)
        ylistt.append(e)
    print("fertig")
    ylisttt=[]
    for x in xlist:
        A=bl.BlockMatrix(3, x)
        A=A.get_sparse()
        b = pp.rhs(3, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)
        e=pp.compute_error(3,x, hat_u, u)
        ylisttt.append(e)
    print("fertig")

    plt.figure()
    plt.plot(xlist, ylist, 'b-',label= 'd=1' )
    plt.plot(xlist, ylistt, 'b-',label= 'd=2' )
    plt.plot(xlist, ylisttt, 'b-',label= 'd=3' )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error")
    plt.legend()
    plt.show()

def u(x, d):
    prod = 1
    for i in range(d):
        prod=prod*x[i]* np.sin(np.pi*x[i])
    return prod

def f(x,d):
    sum=0
    for i in range(d):
        prod=1
        for k in range(d):
            if k!=i:
                prod=prod*float(x[k])* np.sin(np.pi*float(x[k]))
        prod_one = prod*np.sin(np.pi*x[i])
        prod_two = prod*x[i]*np.cos(np.pi*x[i])*np.pi
        sum+=prod_one+prod_two
    return sum

if __name__ == "__main__":
    main()