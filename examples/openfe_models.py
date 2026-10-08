"""Independent symbolic examples matching seven pinned OpenFE expressions.

Source: https://github.com/OpenFreeEnergy/openfe/blob/18208178d084a28ecbd93ce48ebea75d6b1b733e/src/openfe/protocols/openmm_rfe/_rfe_utils/relative.py
Energy-only examples; not complete force-field or hybrid topology builders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import sympy as sp

from sympy_openmm import CompiledExpression, compile_expression, delta, select, step

COULOMB = sp.Float("138.935458")


@dataclass(frozen=True)
class ForceModel:
    name: str
    force_class: str
    energy: sp.Expr
    definitions: Mapping[sp.Symbol, sp.Expr]
    coordinate: str
    globals: tuple[str, ...]
    per_parameters: tuple[str, ...]

    def compile(self, *, cse=True) -> CompiledExpression:
        if self.force_class == "CustomNonbondedForce":
            names = {p + i for p in self.per_parameters for i in ("1", "2")}
        else:
            names = set(self.per_parameters)
        return compile_expression(
            self.energy, definitions=self.definitions, cse=cse,
            allowed_variables={self.coordinate, *self.globals, *names},
        )

    def expanded(self) -> sp.Expr:
        result = self.energy
        for _ in range(len(self.definitions) + 1):
            updated = result.xreplace(self.definitions)
            if result == updated:
                return result
            result = updated
        raise ValueError("Definition expansion did not converge")


class Symbols:
    def __init__(self):
        self.symbols = {}

    def __call__(self, name):
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name, real=True)
        return self.symbols[name]


def bond() -> ForceModel:
    s = Symbols()
    lam = s("lambda_bonds")
    definitions = {
        s("K"): (1-lam)*s("K1") + lam*s("K2"),
        s("length"): (1-lam)*s("length1") + lam*s("length2"),
    }
    return ForceModel("bond", "CustomBondForce", s("K")*(s("r")-s("length"))**2/2,
                      definitions, "r", ("lambda_bonds",), ("length1", "K1", "length2", "K2"))


def angle() -> ForceModel:
    s = Symbols()
    lam = s("lambda_angles")
    definitions = {
        s("K"): (1-lam)*s("K_1") + lam*s("K_2"),
        s("theta0"): (1-lam)*s("theta0_1") + lam*s("theta0_2"),
    }
    return ForceModel("angle", "CustomAngleForce", s("K")*(s("theta")-s("theta0"))**2/2,
                      definitions, "theta", ("lambda_angles",), ("theta0_1", "K_1", "theta0_2", "K_2"))


def torsion() -> ForceModel:
    s = Symbols()
    lam = s("lambda_torsions")
    definitions = {
        s("U1"): s("K1")*(1+sp.cos(s("periodicity1")*s("theta")-s("phase1"))),
        s("U2"): s("K2")*(1+sp.cos(s("periodicity2")*s("theta")-s("phase2"))),
    }
    return ForceModel("torsion", "CustomTorsionForce", (1-lam)*s("U1")+lam*s("U2"),
                      definitions, "theta", ("lambda_torsions",),
                      ("periodicity1", "phase1", "K1", "periodicity2", "phase2", "K2"))


def _sterics(s, *, v2, exceptions):
    r, sigma, epsilon = s("r"), s("sigma"), s("epsilon")
    lam = s("lambda_sterics")
    insert, delete = s("lambda_sterics_insert"), s("lambda_sterics_delete")
    new, old = s("new_interaction"), s("old_interaction")
    alpha_lam = new*(1-insert)+old*delete
    defs = {
        epsilon: (1-lam)*s("epsilonA") + lam*s("epsilonB"),
        sigma: (1-lam)*s("sigmaA") + lam*s("sigmaB"),
    }
    if exceptions:
        defs.update({
            lam: new*insert+old*delete,
            new: delta(1-s("unique_new")),
            old: delta(1-s("unique_old")),
        })
    else:
        defs.update({
            lam: s("core_interaction")*s("lambda_sterics_core")+new*insert+old*delete,
            s("core_interaction"): delta(sum(
                s(n) for n in ("unique_old1","unique_old2","unique_new1","unique_new2")
            )),
            new: sp.Max(s("unique_new1"), s("unique_new2")),
            old: sp.Max(s("unique_old1"), s("unique_old2")),
            s("epsilonA"): sp.sqrt(s("epsilonA1")*s("epsilonA2")),
            s("epsilonB"): sp.sqrt(s("epsilonB1")*s("epsilonB2")),
            s("sigmaA"): (s("sigmaA1")+s("sigmaA2"))/2,
            s("sigmaB"): (s("sigmaB1")+s("sigmaB2"))/2,
        })
    if v2:
        rc, x, force = s("r_LJ"), s("x"), s("Force")
        defs.update({
            s("lambda_sterics_deprecated"): alpha_lam,
            rc: s("softcore_alpha")*(sp.Rational(26,7)*sigma**6*s("lambda_sterics_deprecated"))**sp.Rational(1,6),
            x: (sigma/r)**6,
            force: -4*epsilon*(-12*sigma**12/rc**13+6*sigma**6/rc**7),
            s("U_sterics_cut"): 4*epsilon*(sigma/rc)**6*((sigma/rc)**6-1),
            s("U_sterics_quad"): force*((r-rc)**2/2-(r-rc))+s("U_sterics_cut"),
            s("U_sterics"): select(step(r-rc), 4*epsilon*x*(x-1), s("U_sterics_quad")),
        })
    else:
        reff, x = s("reff_sterics"), s("x")
        defs.update({
            s("lambda_alpha"): alpha_lam,
            reff: sigma*(s("softcore_alpha")*s("lambda_alpha")+(r/sigma)**6)**sp.Rational(1,6),
            x: (sigma/reff)**6,
            s("U_sterics"): 4*epsilon*x*(x-1),
        })
    return defs


def nonbonded(*, v2: bool) -> ForceModel:
    s = Symbols()
    return ForceModel(
        "sterics_v2" if v2 else "sterics_v1", "CustomNonbondedForce",
        s("U_sterics"), _sterics(s, v2=v2, exceptions=False), "r",
        ("softcore_alpha","lambda_sterics_core","lambda_sterics_insert","lambda_sterics_delete"),
        ("sigmaA","epsilonA","sigmaB","epsilonB","unique_old","unique_new"),
    )


def exceptions(*, v2: bool) -> ForceModel:
    s = Symbols()
    defs = _sterics(s, v2=v2, exceptions=True)
    defs[s("U_electrostatics")] = (
        (s("lambda_electrostatics_insert")*s("unique_new")+
         s("unique_old")*(1-s("lambda_electrostatics_delete")))
        * s("ONE_4PI_EPS0")*s("chargeProd")/s("r")
    )
    defs[s("ONE_4PI_EPS0")] = COULOMB
    return ForceModel(
        "exceptions_v2" if v2 else "exceptions_v1", "CustomBondForce",
        s("U_sterics")+s("U_electrostatics"), defs, "r",
        ("softcore_alpha","lambda_electrostatics_insert","lambda_electrostatics_delete",
         "lambda_sterics_insert","lambda_sterics_delete"),
        ("chargeProd","sigmaA","epsilonA","sigmaB","epsilonB","unique_old","unique_new"),
    )


def all_models() -> list[ForceModel]:
    return [bond(), angle(), torsion(), nonbonded(v2=False), nonbonded(v2=True),
            exceptions(v2=False), exceptions(v2=True)]
