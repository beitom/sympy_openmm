"""Symbolic, non-distributional OpenMM step/delta/select functions."""
from __future__ import annotations

import sympy as sp
from sympy.core.function import ArgumentIndexError


class step(sp.Function):
    """0 for x < 0, 1 for x >= 0, including step(0) == 1."""
    nargs = 1

    @classmethod
    def eval(cls, x):
        if x.is_negative is True:
            return sp.S.Zero
        if x.is_nonnegative is True:
            return sp.S.One

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        return sp.S.Zero

    def _eval_is_real(self):
        return True


class delta(sp.Function):
    """Discrete zero-test: 1 exactly at x == 0, 0 otherwise. Not DiracDelta."""
    nargs = 1

    @classmethod
    def eval(cls, x):
        if x.is_zero is True:
            return sp.S.One
        if x.is_zero is False:
            return sp.S.Zero

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        return sp.S.Zero

    def _eval_is_real(self):
        return True


class select(sp.Function):
    """select(condition, yes, no): choose yes when condition != 0, else no."""
    nargs = 3

    @classmethod
    def eval(cls, condition, yes, no):
        if condition.is_zero is True:
            return no
        if condition.is_zero is False:
            return yes
        if yes == no:
            return yes

    def _eval_derivative(self, symbol):
        # Preserve selection instead of emitting mask*derivative products.
        # 0 * an undefined derivative of an inactive branch is not safe.
        condition, yes, no = self.args
        return select(condition, sp.diff(yes, symbol), sp.diff(no, symbol))

    def fdiff(self, argindex=1):
        condition = self.args[0]
        if argindex == 1:
            return sp.S.Zero
        if argindex == 2:
            return select(condition, sp.S.One, sp.S.Zero)
        if argindex == 3:
            return select(condition, sp.S.Zero, sp.S.One)
        raise ArgumentIndexError(self, argindex)
