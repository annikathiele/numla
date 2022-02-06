import block_matrixfinal as bl
import linear_solversfinal as ls
import poisson_problemfinal as pp
import numpy as np
import matplotlib.pyplot as plt

def main():
    graph_omega()

def graph_eps():
    plt.figure()
    xlist =[]
    for i in range(2,15):
        xlist.append(i)
    ylist=[]
    #ylist2=[]
    for k in range(-2,8,2):
        print(k)
        yy=[]
        #yy2=[]
        for x in xlist:
            A=bl.BlockMatrix(2, x)
            A=A.get_sparse()
            b = pp.rhs(2, x, f)
            x0=np.ones(b.size)
            epsi=(1/x)**k
            hat_u=ls.solve_sor2(A,b,x0,eps=epsi)[1].pop()
            e=pp.compute_error(2,x, hat_u, u)
            yy.append(e)

            #hat_u2=ls.solve_sor2(A,b,x0,eps=epsi)[1].pop()
            #e2=pp.compute_error(2,x, hat_u, u)
            #yy2.append(e2)
        ylist.append(yy)
        #ylist2.append(yy2)

    plt.plot(xlist, ylist[0], 'b-',label= 'eps=-2' )
    plt.plot(xlist, ylist[1], 'r-',label= 'eps=0' )
    plt.plot(xlist, ylist[2], 'y-',label= 'eps=2' )
    plt.plot(xlist, ylist[3], 'g-',label= 'eps=4' )
    plt.plot(xlist, ylist[4], 'm-',label= 'eps=6' )
    #plt.plot(xlist, ylist2[3], 'g--',label= '2,eps=4' )
    #plt.plot(xlist, ylist2[4], 'm--',label= '2,eps=6' )
    plt.ylabel("Fehler")
    plt.xlabel("n")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Epsilon")
    plt.legend()
    plt.show()

def graph_omega2():
    plt.figure()
    xlist =[]
    for i in range(19):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(2, 6)
        A=A.get_sparse()
        b = pp.rhs(2, 6, f)
        x0=np.ones(b.size)
        ome=1.9/14*x+0.001
        hat_u=ls.solve_sor(A,b,x0,omega=ome)[1].pop()
        e=pp.compute_error(2,6, hat_u, u)
        ylist.append(e)

    xlist=np.array(xlist)*(1.9/14)

    plt.plot(xlist, ylist, 'b-',label= 'omega' )
    plt.ylabel("Fehler")
    plt.xlabel("Omega")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Omega")
    plt.legend()
    plt.show()

def graph_omega():
    plt.figure()
    xlist =[]
    for i in range(2,16):
        xlist.append(i)
    ylist=[]
    for x in xlist:
        A=bl.BlockMatrix(2, 7)
        A=A.get_sparse()
        b = pp.rhs(2, 7, f)
        x0=np.ones(b.size)
        ome=1.8/16*x
        hat_u=ls.solve_sor2(A,b,x0,omega=ome)[1].pop()
        e=pp.compute_error(2,7, hat_u, u)
        ylist.append(e)

    xlist=np.array(xlist)*(1.8/16)

    plt.plot(xlist, ylist, 'b-',label= 'omega' )

    #plt.xscale("log")
    plt.yscale("log")
    plt.title("Omega")
    plt.legend()
    plt.show()

def graph_error():
    xlist =[]
    for i in range(2,11):
        xlist.append(i)
    ylist=[]
    ylist2=[]
    for x in xlist:
        A=bl.BlockMatrix(1, x)
        A=A.get_sparse()
        b = pp.rhs(1, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)[1].pop()
        e=pp.compute_error(1,x, hat_u, u)
        ylist.append(e)

        hat_u2= ls.solve_sor2(A,b,x0)[1].pop()
        e2=pp.compute_error(1,x, hat_u2, u)
        ylist2.append(e2)

    ylistt=[]
    ylistt2=[]
    for x in xlist:
        A=bl.BlockMatrix(2, x)
        A=A.get_sparse()
        b = pp.rhs(2, x, f)
        x0=np.ones(b.size)
        hat_u= ls.solve_sor(A,b,x0)[1].pop()
        e=pp.compute_error(2,x, hat_u, u)
        ylistt.append(e)

        hat_u2= ls.solve_sor2(A,b,x0)[1].pop()
        e2=pp.compute_error(2,x, hat_u2, u)
        ylistt2.append(e2)
    """
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
    """
    plt.figure()
    plt.ylabel("Fehler")
    plt.xlabel("n")
    #plt.plot(xlist, xlist, 'r--',label="n")
    plt.plot(xlist, ylist, 'b-',label= '1:d=1' )
    plt.plot(xlist, ylist, 'b--',label= '2:d=1' )
    plt.plot(xlist, ylistt, 'y-',label= '1:d=2' )
    plt.plot(xlist, ylistt2, 'y--',label= '2:d=2' )
    #plt.plot(xlist, ylisttt, 'm-',label= 'd=3' )
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
