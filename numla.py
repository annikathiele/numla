import math
import matplotlib.pyplot as plt

def main():
    def f(x):
        y = math.sin(x)/x
        return y
    def d_f(x):
        y = math.cos(x)/x - float(math.sin(x))/float((x**2))
        return y
    def dd_f(x):
        y= -(math.sin(x)/x)+(2*math.sin(x)/x**3)-(2*math.cos(x)/x**2)
        return y

    g_1 = FiniteDifference(0.5, f, d_f, dd_f)
    print(g_1.compute_errors(1, 10, 500))
    g_1.graph(1, 10, 500)
        array = []
    for counter in range(20):
        array.append(10**(-counter))

    #array = [1,0.1,0.01,0.001,0.0001,0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    graph_error(math.pi, 3*math.pi, 1000, array, f, d_f, dd_f)

class FiniteDifference:
    """ Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.
    Parameters
    ----------
    h : float
    Step size of the approximation.
    f : callable
    Function to approximate the derivatives of. The calling signature is
    ‘f(x)‘. Here ‘x‘ is a scalar or array_like of ‘numpy‘. The return
    value is of the same type as ‘x‘.
    d_f : callable, optional
    The analytic first derivative of ‘f‘ with the same signature.
    dd_f : callable, optional
    The analytic second derivative of ‘f‘ with the same signature..
    Attributes
    ----------
    h : float
    Step size of the approximation.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f

    def compute_dh_f(self):
        """Calculates the approximation for the first derivative of the f with step size h.
        Parameters
        ----------
        -
        Return
        ------
        callable
        Calculates the approximation of the first derivative for a given x.
        """
        def my_func_dh(x):
            y = (self.f(x+self.h)-self.f(x))/self.h
            return y

        return my_func_dh

    def compute_ddh_f(self):
        """Calculates the approximation for the second derivative of f with step size h.
        Parameters
        ----------
        -
        Return
        ------
        callable
        Calculates the approximation of the first derivative for a given x.
        """
        def my_func_ddh(x):
            y = (self.f(x+self.h)-2*self.f(x)+self.f(x-self.h))/self.h**2
            return y

        return my_func_ddh


    def compute_errors(self, a, b, p): # pylint: disable=invalid-name
        """ Calculates an approximation to the errors between an approximation
        and the exact derivative for first and second order derivatives in the
        infinity norm.
        Parameters
        ----------
        a, b : float
        Start and end point of the interval.
        p : int
        Number of intervals used in the approximation of the infinity norm.
        Returns
        -------
        float
        Errors of the approximation of the first derivative.
        float
        Errors of the approximation of the second derivative.
        Raises
        ------
        ValueError
        If no analytic derivative was provided by the user.
        """
        e_h = 0
        ee_h = 0
        for i in range(p+1):
            x_i = a + i*abs(b-a) / p
            candidate_e_h = abs(self.d_f(x_i)- self.compute_dh_f()(x_i))
            if candidate_e_h > e_h:
                e_h = candidate_e_h

            candidate_ee_h = abs(self.dd_f(x_i)- self.compute_ddh_f()(x_i))
            if candidate_ee_h > ee_h:
                ee_h = candidate_ee_h

        return e_h, ee_h

        def graph(self, a, b, p): #pylint: disable=invalid-name
            """
            Graphs the functions f, the first and second order finite difference
            approximation of f and if given, the analytic derivative of f.
            Parameters
            ----------
            a, b : float
            Start and end point of the interval.
            p : int
            Number of intervals used in the approximation of the infinity norm.
            Returns
            ----------
            None
                    
            """
            xlist = []
            ylist_f = []
            ylist_df = []
            ylist_ddf = []
            if self.d_f != None:
                ylist_d_f = []
            if self.dd_f != None:
                ylist_dd_f = []
            for counter in range(p+1):
                xpoint = a + (counter*abs(b-a))/p
                xlist.append(xpoint)
                ylist_f.append(self.f(xpoint))
                ylist_df.append(self.compute_dh_f()(xpoint))
                ylist_ddf.append(self.compute_ddh_f()(xpoint))
                if self.d_f != None:
                    ylist_d_f.append(self.d_f(xpoint))
                if self.dd_f != None:
                    ylist_dd_f.append(self.dd_f(xpoint))
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(xlist, ylist_f, color='tab:blue',label= 'f(x)' )
            ax.plot(xlist, ylist_df, color = 'tab:red',label= 'df(x)' )
            ax.plot(xlist, ylist_ddf, color = 'tab:green',label= 'ddf(x)' )
            if self.d_f != None:
                ax.plot(xlist, ylist_d_f, color = 'tab:orange',label= 'd_f(x)' )
            if self.dd_f != None:
                ax.plot(xlist, ylist_dd_f, color = 'tab:pink',label= 'dd_f(x)' )
            plt.title("Finite Differences")
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
def graph_error(a, b, p, h, f, d_f, dd_f): #pylint: disable=invalid-name
    e_h = []
    ee_h = []
    quad = []
    cubic = []
    for number in h:
        g_1 = FiniteDifference(number, f, d_f, dd_f)
        errors = g_1.compute_errors(a,b,p)
        e_h.append(errors[0])
        ee_h.append(errors[1])
        quad.append(number * number)
        cubic.append(number*number*number)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(h,e_h, color = 'tab:blue', label = 'e_f(h)1')
    ax.plot(h,ee_h, color = 'tab:red', label = 'e_f(h)2')
    ax.plot(h,h,color = 'tab:green', label = 'f(h)=h')
    ax.plot(h,quad,color = 'tab:orange', label = 'f(h)=h^2')
    ax.plot(h,cubic,color = 'tab:pink', label = 'f(h)=h^3')
    plt.title("Errors")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('h')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()
