"""SymPy to OpenMM compiler — generated single-file distribution.

MIT License; Copyright (c) 2026 sympy-openmm-compiler contributors.
This file is equivalent to the package API and requires SymPy, not OpenMM.
See the full project README and tests for scope, conventions and validation status.
"""
from __future__ import annotations
# MIT License
# 
# Copyright (c) 2026 sympy-openmm-compiler contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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


from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import inspect
import math
import re
from typing import Any

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, BooleanTrue, BooleanFalse



class CompilationError(ValueError):
    """The input cannot be faithfully represented in the supported target subset."""


@dataclass(frozen=True)
class FunctionSpec:
    """Explicit opt-in for a force-specific or tabulated function.

    Parameters
    ----------
    name: OpenMM function name, e.g. 'distance' or 'my_table'.
    arity: Exact number of arguments, or a tuple of allowed arities.
    particle_arguments: If True, arguments must be bare symbolic particle/group
        tokens (e.g. p1, p2 or g1, g2), not expressions or scalar parameters.
        CSE will never extract these calls or rewrite their token arguments.
    """
    name: str
    arity: int | tuple[int, ...]
    particle_arguments: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not _IDENTIFIER.fullmatch(self.name):
            raise CompilationError(f"Invalid function name: {self.name!r}")
        arities = (self.arity,) if isinstance(self.arity, int) else self.arity
        if not isinstance(arities, tuple) or not arities or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in arities
        ):
            raise CompilationError("Function arity must be a positive integer or tuple of them")

    def accepts(self, nargs: int) -> bool:
        return nargs == self.arity if isinstance(self.arity, int) else nargs in self.arity


@dataclass(frozen=True)
class CompiledExpression:
    """Compilation output and namespace information (no OpenMM dependency)."""
    code: str
    energy: str
    intermediates: tuple[tuple[str, str], ...]
    variables: tuple[str, ...]
    particle_tokens: tuple[str, ...]
    symbol_names: tuple[tuple[sp.Symbol, str], ...]
    required_functions: tuple[str, ...]

    def __str__(self) -> str:
        return self.code


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UNARY = {
    sp.exp: "exp", sp.log: "log", sp.sin: "sin", sp.cos: "cos",
    sp.sec: "sec", sp.csc: "csc", sp.tan: "tan", sp.cot: "cot",
    sp.asin: "asin", sp.acos: "acos", sp.atan: "atan", sp.sinh: "sinh",
    sp.cosh: "cosh", sp.tanh: "tanh", sp.erf: "erf", sp.erfc: "erfc",
    sp.Abs: "abs", sp.floor: "floor", sp.ceiling: "ceil", step: "step", delta: "delta",
}
_RESERVED = set(_UNARY.values()) | {
    "sqrt", "atan2", "min", "max", "select", "nan", "inf", "Infinity",
}
ExpressionInput = sp.Basic | int | float | Callable[..., Any]


def _scalar(value: Any) -> sp.Basic:
    if isinstance(value, (str, bytes)):
        raise CompilationError("Pass a SymPy expression, not source text; strings are not parsed")
    try:
        result = sp.sympify(value)
    except (TypeError, ValueError, sp.SympifyError) as exc:
        raise CompilationError(f"Not a scalar symbolic expression: {type(value).__name__}") from exc
    if not isinstance(result, (sp.Expr, Relational, BooleanTrue, BooleanFalse, And, Or, Not)):
        raise CompilationError(f"Expected a scalar expression, got {type(result).__name__}")
    if result.is_commutative is False:
        raise CompilationError("Noncommutative expressions cannot be compiled to scalar arithmetic")
    return result


def _trace(source: ExpressionInput, arguments: Mapping[str, sp.Symbol] | None) -> sp.Basic:
    if isinstance(source, sp.Lambda):
        if arguments is not None:
            raise CompilationError("arguments applies to Python callables, not SymPy Lambda")
        return _scalar(source.expr)
    if isinstance(source, sp.Basic) or not callable(source):
        if arguments is not None:
            raise CompilationError("arguments applies only to Python callables")
        return _scalar(source)
    try:
        signature = inspect.signature(source)
    except (TypeError, ValueError) as exc:
        raise CompilationError("Callable needs an inspectable signature; otherwise pass its expression") from exc
    supplied = dict(arguments or {})
    unknown = set(supplied) - set(signature.parameters)
    if unknown:
        raise CompilationError(f"Unknown callable arguments: {sorted(unknown)}")
    pos, kw = [], {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise CompilationError("Variadic callables are unsupported; bind a fixed signature first")
        symbol = supplied.get(name, sp.Symbol(name, real=True))
        if not isinstance(symbol, sp.Symbol):
            raise CompilationError(f"Argument {name!r} must map to a SymPy Symbol")
        if parameter.kind == parameter.KEYWORD_ONLY:
            kw[name] = symbol
        else:
            pos.append(symbol)
    try:
        # This is symbolic tracing, NOT a sandbox. Only use trusted callables.
        # Every declared argument is symbolic, even one with a Python default.
        return _scalar(source(*pos, **kw))
    except CompilationError:
        raise
    except Exception as exc:
        raise CompilationError(
            "Symbolic tracing failed. Use SymPy math/Piecewise instead of math, "
            "NumPy, or Python 'if' on symbolic values."
        ) from exc


def _dependency_order(definitions: Mapping[sp.Symbol, sp.Basic]) -> list[sp.Symbol]:
    """Dependency-first DFS, with explicit cycle errors."""
    states: dict[sp.Symbol, int] = {}
    stack: list[sp.Symbol] = []
    ordered: list[sp.Symbol] = []

    def visit(symbol):
        state = states.get(symbol, 0)
        if state == 2:
            return
        if state == 1:
            cycle = stack[stack.index(symbol):] + [symbol]
            raise CompilationError("Cyclic definitions: " + " -> ".join(map(str, cycle)))
        states[symbol] = 1
        stack.append(symbol)
        for dependency in sorted(definitions[symbol].free_symbols & definitions.keys(), key=sp.default_sort_key):
            visit(dependency)
        stack.pop()
        states[symbol] = 2
        ordered.append(symbol)

    for symbol in sorted(definitions, key=sp.default_sort_key):
        visit(symbol)
    return ordered


def _live_definitions(root: sp.Basic, definitions: Mapping[sp.Symbol, sp.Basic]) -> dict[sp.Symbol, sp.Basic]:
    live: set[sp.Symbol] = set()
    pending = list(root.free_symbols & definitions.keys())
    while pending:
        symbol = pending.pop()
        if symbol in live:
            continue
        live.add(symbol)
        pending.extend(definitions[symbol].free_symbols & definitions.keys())
    return {symbol: value for symbol, value in definitions.items() if symbol in live}


class _Printer:
    def __init__(self, names: Mapping[sp.Symbol, str], functions: Mapping[Any, FunctionSpec]):
        self.names = names
        self.functions = functions
        self.used_functions: set[str] = set()
        self.tokens: set[sp.Symbol] = set()

    def call(self, name: str, args: Iterable[str]) -> str:
        self.used_functions.add(name)
        return name + "(" + ", ".join(args) + ")"

    @staticmethod
    def number(expr) -> str:
        try:
            numeric = float(expr)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CompilationError(f"Cannot represent {expr!s} as a real double") from exc
        if not math.isfinite(numeric):
            raise CompilationError(f"Non-finite or overflowing numeric constant: {expr!s}")
        if numeric == 0 and expr.is_zero is False:
            raise CompilationError(f"Numeric constant underflows double precision: {expr!s}")
        if isinstance(expr, sp.Integer):
            return str(expr.p)
        if isinstance(expr, sp.Rational):
            # Keep small rational constants exact until Lepton evaluates them.
            # Avoid overflowing the numerator/denominator of a finite ratio.
            try:
                bounded = math.isfinite(float(expr.p)) and math.isfinite(float(expr.q))
            except OverflowError:
                bounded = False
            return f"({expr.p}/{expr.q})" if bounded else repr(numeric)
        return repr(numeric)

    def print(self, expr: sp.Basic) -> str:
        if isinstance(expr, sp.Symbol):
            return self.names[expr]
        if expr.is_real is False:
            raise CompilationError(f"Known nonreal expression cannot be compiled: {expr!s}")
        if expr == sp.S.true:
            return "1"
        if expr == sp.S.false:
            return "0"
        if isinstance(expr, (sp.Number, sp.NumberSymbol)):
            return self.number(expr)
        if expr is sp.I or expr in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            raise CompilationError(f"OpenMM requires real finite constants, not {expr!s}")
        if isinstance(expr, sp.Add):
            return "(" + " + ".join(self.print(x) for x in expr.args) + ")"
        if isinstance(expr, sp.Mul):
            return "(" + " * ".join(self.print(x) for x in expr.args) + ")"
        if isinstance(expr, sp.Pow):
            base, exponent = expr.args
            b = self.print(base)
            if exponent == sp.S.Half:
                return self.call("sqrt", [b])
            if exponent == -sp.S.Half:
                return "(1/" + self.call("sqrt", [b]) + ")"
            if exponent == -1:
                return f"(1/({b}))"
            return f"(({b})^({self.print(exponent)}))"
        if isinstance(expr, sp.Piecewise):
            if expr.args[-1].cond is not sp.S.true:
                raise CompilationError("Piecewise requires a final (value, True) branch; there is no NaN default")
            result = self.print(expr.args[-1].expr)
            for value, condition in reversed(expr.args[:-1]):
                result = self.call("select", [self.print(condition), self.print(value), result])
            return result
        if isinstance(expr, Relational):
            left, right = self.print(expr.lhs), self.print(expr.rhs)
            diff = f"({left} - {right})"
            if isinstance(expr, sp.Equality):
                return self.call("delta", [diff])
            if isinstance(expr, sp.Unequality):
                return "(1 - " + self.call("delta", [diff]) + ")"
            if isinstance(expr, sp.StrictLessThan):
                return "(1 - " + self.call("step", [diff]) + ")"
            if isinstance(expr, sp.GreaterThan):
                return self.call("step", [diff])
            reverse = f"({right} - {left})"
            if isinstance(expr, sp.LessThan):
                return self.call("step", [reverse])
            if isinstance(expr, sp.StrictGreaterThan):
                return "(1 - " + self.call("step", [reverse]) + ")"
        if isinstance(expr, And):
            return "(" + " * ".join(self.print(arg) for arg in expr.args) + ")"
        if isinstance(expr, Or):
            return "(1 - (" + " * ".join(f"(1 - {self.print(arg)})" for arg in expr.args) + "))"
        if isinstance(expr, Not):
            return f"(1 - {self.print(expr.args[0])})"
        if isinstance(expr, sp.Heaviside):
            x, h0 = expr.args
            if h0 == 1:
                return self.call("step", [self.print(x)])
            # SymPy defaults to H(0)=1/2, unlike OpenMM step(0)=1.
            xcode = self.print(x)
            return self.call("select", [self.call("delta", [xcode]), self.print(h0), self.call("step", [xcode])])
        if isinstance(expr, sp.KroneckerDelta):
            if len(expr.args) != 2:
                raise CompilationError("Range-restricted KroneckerDelta is unsupported")
            return self.call("delta", [f"({self.print(expr.args[0])} - {self.print(expr.args[1])})"])
        if expr.func is sp.sign:
            x = self.print(expr.args[0])
            return self.call("select", [self.call("delta", [x]), "0", f"(2*{self.call('step', [x])} - 1)"])
        if isinstance(expr, (sp.Min, sp.Max)):
            name = "min" if isinstance(expr, sp.Min) else "max"
            args = [self.print(arg) for arg in expr.args]
            result = args[-1]
            for arg in reversed(args[:-1]):
                result = self.call(name, [arg, result])
            return result
        if expr.func in _UNARY and len(expr.args) == 1:
            return self.call(_UNARY[expr.func], [self.print(expr.args[0])])
        if expr.func is sp.atan2 and len(expr.args) == 2:
            return self.call("atan2", map(self.print, expr.args))
        if expr.func is select:
            return self.call("select", map(self.print, expr.args))
        if expr.func is sp.Mod:
            a, b = map(self.print, expr.args)
            return f"({a} - {b}*{self.call('floor', [f'({a}/{b})'])})"
        if expr.func in self.functions:
            spec = self.functions[expr.func]
            if not spec.accepts(len(expr.args)):
                raise CompilationError(f"{spec.name} expects arity {spec.arity}, got {len(expr.args)}")
            if spec.particle_arguments:
                if not all(isinstance(arg, sp.Symbol) for arg in expr.args):
                    raise CompilationError(f"{spec.name} requires bare particle/group tokens")
                self.tokens.update(expr.args)
            return self.call(spec.name, map(self.print, expr.args))
        if isinstance(expr, sp.DiracDelta):
            raise CompilationError("DiracDelta is a distribution, not OpenMM delta; use delta or KroneckerDelta")
        raise CompilationError(
            f"Unsupported SymPy node {expr.func.__name__}: {expr!s}. "
            "Rewrite it using supported primitives or explicitly register a FunctionSpec."
        )


def compile_expression(
    source: ExpressionInput,
    *,
    definitions: Mapping[sp.Symbol, Any] | None = None,
    cse: bool = True,
    symbol_names: Mapping[sp.Symbol, str] | None = None,
    arguments: Mapping[str, sp.Symbol] | None = None,
    functions: Mapping[Any, FunctionSpec] | None = None,
    allowed_variables: Iterable[str] | None = None,
    temp_prefix: str = "tmp",
) -> CompiledExpression:
    """Compile an expression, SymPy Lambda, or symbolically traceable callable.

    definitions maps symbolic intermediate names to their expressions, in any
    order. Dead definitions are dropped, cycles rejected, and live intermediates
    emitted in OpenMM's use-before-definition order. Automatic CSE is optional.

    symbol_names renames Symbols explicitly; ambiguous names are otherwise errors.
    arguments optionally supplies real Symbol objects for Python callable arguments.
    Every callable argument is traced as symbolic (defaults do not bind constants).
    functions explicitly opts in to registered force-specific/tabulated functions.
    allowed_variables can validate the final scalar namespace; particle tokens are
    tracked separately. It is NOT a parameter-registration or dimensional check.

    Only real scalar mathematics is supported. User assumptions and any SymPy
    simplifications that occurred before compilation remain the user's responsibility.
    """
    root = _trace(source, arguments)
    allowed = None
    if allowed_variables is not None:
        if isinstance(allowed_variables, str):
            raise CompilationError("allowed_variables must be an iterable of names, not a single string")
        allowed = set(allowed_variables)
        if any(not isinstance(name, str) for name in allowed):
            raise CompilationError("allowed_variables entries must be strings")
    defs = dict(definitions or {})
    if any(not isinstance(key, sp.Symbol) for key in defs):
        raise CompilationError("Definition keys must be SymPy Symbols")
    defs = {key: _scalar(value) for key, value in defs.items()}
    _dependency_order(defs)  # Validate even dead definition cycles.
    defs = _live_definitions(root, defs)
    registry = dict(functions or {})
    for func, spec in registry.items():
        if not isinstance(spec, FunctionSpec):
            raise CompilationError("functions values must be FunctionSpec instances")
        if not isinstance(func, sp.core.function.FunctionClass):
            raise CompilationError("functions keys must be symbolic function classes, e.g. sympy.Function('table')")
        if func in _UNARY or func in {
            select, sp.atan2, sp.sign, sp.Heaviside, sp.KroneckerDelta, sp.Mod,
        }:
            raise CompilationError(f"Cannot override builtin symbolic function {func.__name__!r}")
        if spec.name in _RESERVED:
            raise CompilationError(f"Cannot redefine builtin function {spec.name!r}")
    if len({spec.name for spec in registry.values()}) != len(registry):
        raise CompilationError("Multiple symbolic functions map to the same target function name")
    if not isinstance(temp_prefix, str) or not _IDENTIFIER.fullmatch(temp_prefix):
        raise CompilationError(f"Invalid temporary prefix: {temp_prefix!r}")
    expressions = [root, *defs.values()]
    symbols = set(defs)
    for expr in expressions:
        symbols.update(expr.free_symbols)
    if any(not isinstance(symbol, sp.Symbol) for symbol in symbols):
        raise CompilationError("Only scalar Symbols are supported, not Indexed or matrix-valued references")
    overrides = dict(symbol_names or {})
    if any(not isinstance(key, sp.Symbol) for key in overrides):
        raise CompilationError("symbol_names keys must be SymPy Symbols")
    # Permit unused override entries, e.g. after CSE/dead-code elimination.
    names = {symbol: overrides.get(symbol, symbol.name) for symbol in symbols}
    occupied: dict[str, sp.Symbol] = {}
    reserved = _RESERVED | {spec.name for spec in registry.values()}
    for symbol, name in names.items():
        if symbol.is_commutative is False:
            raise CompilationError(f"Noncommutative symbol: {symbol!s}")
        if symbol.is_real is False:
            raise CompilationError(f"Explicitly nonreal symbol: {symbol!s}")
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or name in reserved:
            raise CompilationError(f"Invalid or reserved OpenMM identifier {name!r}; use symbol_names to rename it")
        if name in occupied and occupied[name] != symbol:
            raise CompilationError(f"Distinct Symbols map to {name!r}; supply distinct symbol_names")
        occupied[name] = symbol
    # Validate BEFORE CSE so unsupported nodes cannot be hidden by rewrites.
    preflight = _Printer(names, registry)
    for expr in expressions:
        preflight.print(expr)
    tokens = preflight.tokens
    if tokens & defs.keys():
        raise CompilationError("Particle/group tokens cannot also be defined as scalar intermediates")
    # A token must not double as a scalar parameter elsewhere in an expression.
    token_call_heads = {head for head, spec in registry.items() if spec.particle_arguments}
    def scalar_symbols(expr):
        if expr.func in token_call_heads:
            return set()
        if isinstance(expr, sp.Symbol):
            return {expr}
        return set().union(*(scalar_symbols(arg) for arg in expr.args)) if expr.args else set()
    if tokens & set().union(*(scalar_symbols(expr) for expr in expressions)):
        raise CompilationError("Particle/group tokens cannot also be used in scalar arithmetic")

    if cse:
        reserved_names = set(occupied) | reserved
        # Protect source-symbol identity as well as the emitted names. Otherwise
        # a renamed Symbol('tmp0') could collide with a newly generated tmp0.
        reserved_names.update(symbol.name for symbol in symbols)
        if allowed is not None:
            reserved_names.update(allowed)
        # Respect unused explicit renamings too when generating temporary names.
        reserved_names.update(value for value in overrides.values() if isinstance(value, str))
        def temporaries():
            index = 0
            while True:
                name = f"{temp_prefix}{index}"
                index += 1
                if name not in reserved_names:
                    reserved_names.add(name)
                    symbol = sp.Symbol(name)
                    names[symbol] = name
                    yield symbol
        keys = list(defs)
        replacements, reduced = sp.cse(
            [root, *(defs[key] for key in keys)],
            symbols=temporaries(), order="canonical", optimizations=None,
            # Geometry token arguments must remain literal tokens, not CSE temps.
            ignore=tuple(tokens),
        )
        root = reduced[0]
        defs = dict(zip(keys, reduced[1:]))
        defs.update(replacements)
        defs = _live_definitions(root, defs)
    order = _dependency_order(defs)
    # OpenMM's order is the opposite of normal imperative assignments.
    printer = _Printer(names, registry)
    energy = printer.print(root)
    intermediates = tuple((names[key], printer.print(defs[key])) for key in reversed(order))
    free = set(root.free_symbols)
    for value in defs.values():
        free.update(value.free_symbols)
    free.difference_update(defs)
    scalar_free = free - printer.tokens
    variables = tuple(sorted(names[symbol] for symbol in scalar_free))
    if allowed is not None:
        unknown = set(variables) - allowed
        if unknown:
            raise CompilationError(f"Undeclared scalar variables: {sorted(unknown)}")
    code = "; ".join([energy, *(f"{name} = {value}" for name, value in intermediates)])
    used = free | defs.keys()
    return CompiledExpression(
        code=code, energy=energy, intermediates=intermediates, variables=variables,
        particle_tokens=tuple(sorted(names[symbol] for symbol in printer.tokens)),
        symbol_names=tuple(sorted(((symbol, names[symbol]) for symbol in used), key=lambda pair: pair[1])),
        required_functions=tuple(sorted(printer.used_functions & {spec.name for spec in registry.values()})),
    )


def openmm_code(source: ExpressionInput, **kwargs: Any) -> str:
    """String-only convenience wrapper around compile_expression."""
    return compile_expression(source, **kwargs).code

__version__ = "0.1.0"
__all__ = ["CompilationError", "CompiledExpression", "FunctionSpec", "compile_expression",
           "openmm_code", "delta", "select", "step"]