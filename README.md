# SymPy → OpenMM custom-force expression compiler

Convert scalar SymPy expressions, SymPy Lambdas, and symbolically traceable Python callables into OpenMM/Lepton energy strings.

**The compiler requires only SymPy.** OpenMM is not a runtime dependency, optional extra, or dependency group. Producing a string does not construct or import OpenMM.

## Development with uv

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q -ra
```

`uv sync` resolves the runtime dependency (`sympy`) and dev-group dependencies (`pytest`, `numpy`), **not OpenMM**. To install this project into another uv project from a clone:

```bash
uv add --editable /path/to/sympy_openmm
```

Native OpenMM testing is **optional** and explicitly opted into at invocation (without adding it to `pyproject.toml`):

```bash
REQUIRE_OPENMM=1 uv run --with 'openmm>=8.2' pytest -m openmm -q
```

The native suite uses the Reference platform by default. Set `OPENMM_TEST_PLATFORM=CPU` or `CUDA` to run on another available backend. The default CI job never installs OpenMM.

## Quick start

```python
import sympy as sp
from sympy_openmm import compile_expression, openmm_code

def bond_energy(r, k, length):
    return k*(r-length)**2/2

compiled = compile_expression(bond_energy, cse=True)
print(compiled.code)
print(compiled.variables)

x, a, b = sp.symbols("x a b")
print(openmm_code(a*(x-b)**2/2))
```

Then in an **application already using OpenMM**, pass the generated energy string to `openmm.CustomBondForce` or another compatible force, and register parameters with OpenMM yourself. The compiler does not infer physical units, topology, exclusions, switching, PME, or parameter scopes.

### Named definitions and CSE

```python
from sympy_openmm import compile_expression
import sympy as sp

x, k, length = sp.symbols("x k length")
result = compile_expression(
    k*(x-length)**2/2,
    definitions={k: x+1, length: x/2},
    cse=True,
)
print(result.code)
```

Definitions are dependency-ordered into OpenMM's reverse assignment convention. Cycles, name collisions, invalid identifiers, and unsupported constructs fail explicitly with `CompilationError`.

The printer supports common arithmetic and OpenMM functions, `Piecewise`, comparisons, and boolean conditions. Boundaries are preserved: SymPy `Heaviside(0)=1/2` differs from OpenMM `step(0)=1`, so the compiler handles this case explicitly. An OpenMM `select` is not guaranteed to short-circuit singular inactive branches on all platforms.

### Pinned OpenFE examples

`examples/openfe_models.py` defines seven scalar energy models, adapted from [OpenFE `relative.py`, commit `18208178`](https://github.com/OpenFreeEnergy/openfe/blob/18208178d084a28ecbd93ce48ebea75d6b1b733e/src/openfe/protocols/openmm_rfe/_rfe_utils/relative.py#L866): interpolated bonds, angles, torsions, nonbonded sterics v1/v2, and exception potentials v1/v2.

```python
from examples.openfe_models import all_models
for model in all_models():
    print(model.name, model.compile(cse=True).code)
```

These are *energy equations only*, not full hybrid topologies or FEP jobs.

## Validation status

The repository's lightweight, OpenMM-free tests compare all seven generated potentials against their independent SymPy evaluations, including CSE on/off, and exercise expression lowering, symbol checks and boundary behavior. An optional native Context smoke test must be run separately.

An earlier prototype had 159 passing tests and 7,000 energy/24,000 derivative numerical comparisons (see `validation_report.json`), but that does not demonstrate native OpenMM execution or replace validation on your desired backend. Production adoption requires native gradient, edge-case, GPU and protocol testing.

The source compiler also has a portable single-file form in `standalone/sympy_openmm.py`.

## License and documentation

MIT. See `THIRD_PARTY_NOTICES.md` for OpenFreeEnergy attribution.

- [OpenMM custom expressions and order of intermediates](https://docs.openmm.org/latest/userguide/theory/03_custom_forces.html)
- [SymPy CSE](https://docs.sympy.org/latest/modules/rewriting.html)
