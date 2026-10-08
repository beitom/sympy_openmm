"""Compiler and OpenFE examples require SymPy, not OpenMM."""
import math
import sys

import pytest
import sympy as sp

from sympy_openmm import (
    CompilationError, FunctionSpec, compile_expression, openmm_code,
    delta, select, step,
)
from examples.openfe_models import all_models


def interpret(expression, values):
    """A small *test-only* scalar evaluator for generated Lepton syntax."""
    funcs = {k: getattr(math, k) for k in (
        'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos',
        'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'erf', 'erfc', 'floor',
        'ceil',
    )}
    funcs.update(dict(abs=abs, min=min, max=max,
                      step=lambda x: 1 if x >= 0 else 0,
                      delta=lambda x: 1 if x == 0 else 0,
                      select=lambda s, a, b: a if s != 0 else b))
    context = {**funcs, **values}
    chunks = [c.strip() for c in expression.split(';') if c.strip()]
    for part in reversed(chunks[1:]):
        name, definition = part.split('=', 1)
        context[name.strip()] = eval(definition.replace('^', '**'), {'__builtins__': {}}, context)
    return eval(chunks[0].replace('^', '**'), {'__builtins__': {}}, context)


VALUES = dict(
    r=0.45, theta=1.2, lambda_bonds=0.3, lambda_angles=0.5,
    lambda_torsions=0.4, K1=220, K2=500, K_1=330, K_2=550,
    length1=0.14, length2=0.17, theta0_1=1.1, theta0_2=1.4,
    periodicity1=2, periodicity2=3, phase1=0.2, phase2=1.1,
    softcore_alpha=0.5, lambda_sterics_core=0.5,
    lambda_sterics_insert=0.35, lambda_sterics_delete=0.4,
    lambda_electrostatics_insert=0.55,
    lambda_electrostatics_delete=0.3,
    sigmaA=0.3, sigmaB=0.32, epsilonA=0.6, epsilonB=0.8,
    chargeProd=-0.2, unique_new=1, unique_old=0,
    unique_new1=1, unique_new2=0, unique_old1=0, unique_old2=0,
    sigmaA1=0.31, sigmaA2=0.29, sigmaB1=0.33, sigmaB2=0.31,
    epsilonA1=0.6, epsilonA2=0.6, epsilonB1=0.8, epsilonB2=0.8,
)


@pytest.mark.parametrize('cse', [False, True])
@pytest.mark.parametrize('model', all_models(), ids=lambda m: m.name)
def test_openfe_sympy_expressions_agree_with_emitted_strings(model, cse):
    result = model.compile(cse=cse)
    obtained = interpret(result.code, VALUES)
    expanded = model.expanded()
    expected = float(expanded.subs({s: VALUES[s.name] for s in expanded.free_symbols}).evalf())
    assert math.isfinite(obtained)
    assert obtained == pytest.approx(expected, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize('x_value', [-2, -0.5, 0, 0.5, 2])
def test_heaviside_boundary_and_openmm_step(x_value):
    x = sp.Symbol('x', real=True)
    assert interpret(openmm_code(sp.Heaviside(x)), {'x': x_value}) == (0.5 if x_value == 0 else int(x_value > 0))
    assert interpret(openmm_code(step(x)), {'x': x_value}) == int(x_value >= 0)
    assert interpret(openmm_code(delta(x)), {'x': x_value}) == int(x_value == 0)


@pytest.mark.parametrize('x_value', [-1.1, 0.1, 1.5, 3.2])
def test_piecewise_boundaries(x_value):
    x = sp.Symbol('x', real=True)
    expr = sp.Piecewise((1, x < 0), (x**2, x < 2), (3, True))
    assert interpret(openmm_code(expr), {'x': x_value}) == pytest.approx(float(expr.subs(x, x_value)))


def test_named_definitions_are_sorted_in_openmm_order():
    x, a, b, c = sp.symbols('x a b c', real=True)
    result = compile_expression(a**2 + b**2, definitions={a: c + 1, b: c - 1, c: x**2}, cse=True)
    assert result.variables == ('x',)
    assert interpret(result.code, {'x': 1.5}) == pytest.approx((1.5**2 + 1)**2 + (1.5**2 - 1)**2)


def test_function_compilation_callable_and_lambda():
    def bond(r, k, r0):
        return k * (r-r0)**2 / 2
    for func in [bond, sp.Lambda(sp.symbols('r k r0'), sp.Symbol('k') * (sp.Symbol('r')-sp.Symbol('r0'))**2 / 2)]:
        compiled = compile_expression(func)
        assert interpret(compiled.code, {'r': 0.3, 'k': 800, 'r0': 0.2}) == pytest.approx(4.0)


def test_reserved_functions_and_symbols_fail():
    x = sp.Symbol('x')
    with pytest.raises(CompilationError):
        openmm_code(sp.Function('not_supported')(x))
    with pytest.raises(CompilationError):
        openmm_code(sp.Symbol('invalid-name'))
    with pytest.raises(CompilationError):
        openmm_code(x, allowed_variables={'y'})


def test_custom_tables_need_registration():
    x = sp.Symbol('x')
    table = sp.Function('my_lookup')
    compiled = compile_expression(table(x), functions={table: FunctionSpec('my_table', 1)})
    assert compiled.code == 'my_table(x)'
    assert compiled.required_functions == ('my_table',)


def test_no_openmm_imported_as_part_of_core_package():
    assert 'openmm' not in sys.modules
