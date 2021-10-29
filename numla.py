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



    def compute_dh_f(self):
       """Calculates the approximation for the first derivative of the f with step size h.
       Parameters
       ----------

       Return
       ------
       callable
           Calculates the approximation of the first derivative for a given x.
       """

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