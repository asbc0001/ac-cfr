"""Selection helpers for configured Deep CFR implementations."""

from ac_cfr.common.config import DeepCFRImplementationId
from ac_cfr.solvers.deep_cfr import DeepCFR
from ac_cfr.solvers.naive_deep_cfr import NaiveDeepCFR

_SOLVER_TYPES: dict[DeepCFRImplementationId, type[NaiveDeepCFR]] = {
    DeepCFRImplementationId.REFERENCE: NaiveDeepCFR,
    DeepCFRImplementationId.OPTIMISED: DeepCFR,
}


def deep_cfr_solver_type(
    implementation: DeepCFRImplementationId,
) -> type[NaiveDeepCFR]:
    """Return the solver class selected by a validated implementation ID."""
    if not isinstance(implementation, DeepCFRImplementationId):
        raise TypeError("implementation must be a DeepCFRImplementationId")
    return _SOLVER_TYPES[implementation]


def deep_cfr_implementation(solver: NaiveDeepCFR) -> DeepCFRImplementationId:
    """Return the exact implementation ID for a supported solver instance."""
    for implementation, solver_type in _SOLVER_TYPES.items():
        if type(solver) is solver_type:
            return implementation
    raise TypeError("solver must be a supported Deep CFR implementation")
