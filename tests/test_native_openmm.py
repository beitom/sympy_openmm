"""Opt-in native OpenMM smoke test. OpenMM is NOT a project dependency."""
import os
import pytest
import sympy as sp

try:
    import openmm
    import openmm.unit as unit
except ImportError:
    if os.environ.get("REQUIRE_OPENMM") == "1":
        raise
    pytest.skip("OpenMM is optional and not installed", allow_module_level=True)

from sympy_openmm import openmm_code


@pytest.mark.openmm
def test_harmonic_bond_native_context():
    r, k, r0 = sp.symbols("r k r0", real=True)
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.CustomBondForce(openmm_code(k*(r-r0)**2/2))
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    force.addBond(0, 1, [1000.0, 0.15])
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    platform = openmm.Platform.getPlatformByName(os.environ.get("OPENMM_TEST_PLATFORM", "Reference"))
    context = openmm.Context(system, integrator, platform)
    try:
        context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(0.2, 0, 0)]*unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        assert energy == pytest.approx(1.25, rel=1e-7)
        assert forces[0][0] == pytest.approx(50.0, rel=1e-7)
        assert forces[1][0] == pytest.approx(-50.0, rel=1e-7)
    finally:
        del context
        del integrator
