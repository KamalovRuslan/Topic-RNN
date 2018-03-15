import torch
from .optimizer import Optimizer, required


class TRnnOpt(Optimizer):
    r"""Implements Exponential gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        super(TRnnOpt, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TRnnOpt, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            pass

        # Exponential gradient descent step for each value in TokTop dict
        # Gradient descent for proir Diriclet distribution in TokTop dict

        return loss
