__author__ = "Pierre Navaro"
__email__ = "navaro@math.cnrs.fr"

class Est:
    """
     type: chosen predefined type of model error covatiance ('fixed', 'adaptive')
     form: chosen esimated matrix form ('full', 'diag', 'constant')
     base: for fixed base of model covariance (for 'constant' matrix form only)
     decision: chosen if Q is estimated or not ('True', 'False')
    """

    def __init__(self, value=None, type=None, form=None, base=None, decision=None):
        self.value = value
        self.type = type
        self.form = form
        self.base = base
        self.decision = decision