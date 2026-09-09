r"""Robust RBE safe-load analysis with finite geometry uncertainty scenarios.

The selected ``load_dofs`` define the common two-dimensional visible load
``u``. Geometry uncertainty is represented by a finite set of scenario
problems with scenario-specific equilibrium, friction, compression, baseline
load, and visible-load projection matrices:

.. math::

    U_G = \{u \mid \forall s,\ \exists x_s:
    A_s x_s = -p_s,\ H_s x_s \leq h_s,\ P_s x_s = u\}.

This implementation enumerates geometry scenarios and therefore remains a
linear program for the linearized RBE friction cone. Continuous nonlinear
geometry uncertainty is intentionally out of scope.

Generated ``point_offset_bounds`` scenarios sample independent local in-plane
offsets for every interface point with zero normal offset. Generated
``normal_tilt_bounds`` scenarios sample independent frame tilts per interface.
"""

import copy
import itertools
import time
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from compas.geometry import Rotation
from compas_assembly.datastructures import Assembly
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from .rbe_robust import RobustForceResult
from .rbe_robust import _directions
from .rbe_robust import _inner_polygon
from .rbe_robust import _LoadSetAnalysis
from .rbe_robust import _outer_polygon
from .rbe_robust import _prepare_problem
from .rbe_robust import _validate_load_dofs
from .rbe_robust import _validate_tolerance
from .rbe_robust import _xyz_array


@dataclass
class GeometryScenarioProblem:
    """Sparse matrices defining one geometry-uncertainty scenario."""

    equilibrium: csr_matrix
    inequalities: csr_matrix
    inequality_rhs: np.ndarray
    baseline_load: np.ndarray
    load_projection: csr_matrix
    variable_count: int
    name: str = ""


def rbe_uncertainty_geometry_sample(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    interface_scale_factors: Optional[Sequence[float]] = None,
    point_offset_vectors: Optional[Sequence[Sequence[float]]] = None,
    normal_tilt_vectors: Optional[Sequence[Sequence[float]]] = None,
    point_offset_bounds: Optional[Sequence[float]] = None,
    point_offset_sample_count: int = 60,
    point_offset_seed: Optional[int] = 0,
    normal_tilt_bounds: Optional[Sequence[float]] = None,
    normal_tilt_sample_count: int = 60,
    normal_tilt_seed: Optional[int] = 1,
    contact_failure_scenarios: Optional[Sequence[Any]] = None,
    geometry_scenarios: Optional[Sequence[GeometryScenarioProblem]] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate the geometry-robust safe load range by radial sampling."""
    start_time = time.time()
    load_dofs, scenarios = _prepare_geometry_scenarios(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        interface_scale_factors=interface_scale_factors,
        point_offset_vectors=point_offset_vectors,
        point_offset_bounds=point_offset_bounds,
        point_offset_sample_count=point_offset_sample_count,
        point_offset_seed=point_offset_seed,
        normal_tilt_vectors=normal_tilt_vectors,
        normal_tilt_bounds=normal_tilt_bounds,
        normal_tilt_sample_count=normal_tilt_sample_count,
        normal_tilt_seed=normal_tilt_seed,
        contact_failure_scenarios=contact_failure_scenarios,
        geometry_scenarios=geometry_scenarios,
    )
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set_geometry(scenarios, options)

    statuses = []
    radial_limits = []
    boundary_points = []
    for direction in directions:
        status, limit = _solve_radial_geometry(scenarios, direction, analysis.feasible_center, options)
        statuses.append(status)
        radial_limits.append(limit)
        if limit is not None:
            boundary_points.append((analysis.feasible_center + limit * direction).tolist())

    inner_polygon = _inner_polygon(boundary_points, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="sample",
        load_dofs=load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=analysis.origin_feasible,
        is_bounded=analysis.is_bounded,
        feasible_center=analysis.feasible_center.tolist(),
        radial_limits=radial_limits,
        boundary_points=boundary_points,
        inner_polygon=inner_polygon,
        polygon=inner_polygon,
    )
    _report_geometry(result, len(scenarios), start_time, verbose, timer)
    return result


def rbe_uncertainty_geometry_support_primal(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    interface_scale_factors: Optional[Sequence[float]] = None,
    point_offset_vectors: Optional[Sequence[Sequence[float]]] = None,
    normal_tilt_vectors: Optional[Sequence[Sequence[float]]] = None,
    point_offset_bounds: Optional[Sequence[float]] = None,
    point_offset_sample_count: int = 60,
    point_offset_seed: Optional[int] = 0,
    normal_tilt_bounds: Optional[Sequence[float]] = None,
    normal_tilt_sample_count: int = 60,
    normal_tilt_seed: Optional[int] = 1,
    contact_failure_scenarios: Optional[Sequence[Any]] = None,
    geometry_scenarios: Optional[Sequence[GeometryScenarioProblem]] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate the geometry-robust safe load range with primal supports."""
    start_time = time.time()
    load_dofs, scenarios = _prepare_geometry_scenarios(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        interface_scale_factors=interface_scale_factors,
        point_offset_vectors=point_offset_vectors,
        point_offset_bounds=point_offset_bounds,
        point_offset_sample_count=point_offset_sample_count,
        point_offset_seed=point_offset_seed,
        normal_tilt_vectors=normal_tilt_vectors,
        normal_tilt_bounds=normal_tilt_bounds,
        normal_tilt_sample_count=normal_tilt_sample_count,
        normal_tilt_seed=normal_tilt_seed,
        contact_failure_scenarios=contact_failure_scenarios,
        geometry_scenarios=geometry_scenarios,
    )
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set_geometry(scenarios, options)

    statuses = []
    support_values = []
    support_points = []
    halfspaces = []
    for direction in directions:
        status, support, point = _solve_primal_support_geometry(scenarios, direction, options)
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])
        if point is not None:
            support_points.append(point)

    inner_polygon = _inner_polygon(support_points, tolerance) if analysis.is_bounded else []
    outer_polygon = _outer_polygon(halfspaces, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="support",
        load_dofs=load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=analysis.origin_feasible,
        is_bounded=analysis.is_bounded,
        feasible_center=analysis.feasible_center.tolist(),
        support_formulation="primal",
        support_values=support_values,
        support_points=support_points,
        halfspaces=halfspaces,
        inner_polygon=inner_polygon,
        outer_polygon=outer_polygon,
        polygon=outer_polygon,
    )
    _report_geometry(result, len(scenarios), start_time, verbose, timer)
    return result


def rbe_uncertainty_geometry_support_dual(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    interface_scale_factors: Optional[Sequence[float]] = None,
    point_offset_vectors: Optional[Sequence[Sequence[float]]] = None,
    normal_tilt_vectors: Optional[Sequence[Sequence[float]]] = None,
    point_offset_bounds: Optional[Sequence[float]] = None,
    point_offset_sample_count: int = 60,
    point_offset_seed: Optional[int] = 0,
    normal_tilt_bounds: Optional[Sequence[float]] = None,
    normal_tilt_sample_count: int = 60,
    normal_tilt_seed: Optional[int] = 1,
    contact_failure_scenarios: Optional[Sequence[Any]] = None,
    geometry_scenarios: Optional[Sequence[GeometryScenarioProblem]] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate the geometry-robust safe load range with dual supports."""
    start_time = time.time()
    load_dofs, scenarios = _prepare_geometry_scenarios(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        interface_scale_factors=interface_scale_factors,
        point_offset_vectors=point_offset_vectors,
        point_offset_bounds=point_offset_bounds,
        point_offset_sample_count=point_offset_sample_count,
        point_offset_seed=point_offset_seed,
        normal_tilt_vectors=normal_tilt_vectors,
        normal_tilt_bounds=normal_tilt_bounds,
        normal_tilt_sample_count=normal_tilt_sample_count,
        normal_tilt_seed=normal_tilt_seed,
        contact_failure_scenarios=contact_failure_scenarios,
        geometry_scenarios=geometry_scenarios,
    )
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set_geometry(scenarios, options)

    statuses = []
    support_values = []
    halfspaces = []
    for direction in directions:
        status, support = _solve_dual_support_geometry(scenarios, direction, options)
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])

    outer_polygon = _outer_polygon(halfspaces, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="support",
        load_dofs=load_dofs,
        directions=[direction.tolist() for direction in directions],
        statuses=statuses,
        origin_feasible=analysis.origin_feasible,
        is_bounded=analysis.is_bounded,
        feasible_center=analysis.feasible_center.tolist(),
        support_formulation="dual",
        support_values=support_values,
        halfspaces=halfspaces,
        outer_polygon=outer_polygon,
        polygon=outer_polygon,
    )
    _report_geometry(result, len(scenarios), start_time, verbose, timer)
    return result


def rbe_uncertainty_geometry_support(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    interface_scale_factors: Optional[Sequence[float]] = None,
    point_offset_vectors: Optional[Sequence[Sequence[float]]] = None,
    normal_tilt_vectors: Optional[Sequence[Sequence[float]]] = None,
    point_offset_bounds: Optional[Sequence[float]] = None,
    point_offset_sample_count: int = 60,
    point_offset_seed: Optional[int] = 0,
    normal_tilt_bounds: Optional[Sequence[float]] = None,
    normal_tilt_sample_count: int = 60,
    normal_tilt_seed: Optional[int] = 1,
    contact_failure_scenarios: Optional[Sequence[Any]] = None,
    geometry_scenarios: Optional[Sequence[GeometryScenarioProblem]] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Alias for :func:`rbe_uncertainty_geometry_support_dual`."""
    return rbe_uncertainty_geometry_support_dual(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        interface_scale_factors=interface_scale_factors,
        point_offset_vectors=point_offset_vectors,
        point_offset_bounds=point_offset_bounds,
        point_offset_sample_count=point_offset_sample_count,
        point_offset_seed=point_offset_seed,
        normal_tilt_vectors=normal_tilt_vectors,
        normal_tilt_bounds=normal_tilt_bounds,
        normal_tilt_sample_count=normal_tilt_sample_count,
        normal_tilt_seed=normal_tilt_seed,
        contact_failure_scenarios=contact_failure_scenarios,
        geometry_scenarios=geometry_scenarios,
        num_directions=num_directions,
        tolerance=tolerance,
        solver_options=solver_options,
        verbose=verbose,
        timer=timer,
    )


def rbe_uncertainty_geometry(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    interface_scale_factors: Optional[Sequence[float]] = None,
    point_offset_vectors: Optional[Sequence[Sequence[float]]] = None,
    normal_tilt_vectors: Optional[Sequence[Sequence[float]]] = None,
    point_offset_bounds: Optional[Sequence[float]] = None,
    point_offset_sample_count: int = 60,
    point_offset_seed: Optional[int] = 0,
    normal_tilt_bounds: Optional[Sequence[float]] = None,
    normal_tilt_sample_count: int = 60,
    normal_tilt_seed: Optional[int] = 1,
    contact_failure_scenarios: Optional[Sequence[Any]] = None,
    geometry_scenarios: Optional[Sequence[GeometryScenarioProblem]] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Alias for :func:`rbe_uncertainty_geometry_support_dual`."""
    return rbe_uncertainty_geometry_support_dual(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        interface_scale_factors=interface_scale_factors,
        point_offset_vectors=point_offset_vectors,
        point_offset_bounds=point_offset_bounds,
        point_offset_sample_count=point_offset_sample_count,
        point_offset_seed=point_offset_seed,
        normal_tilt_vectors=normal_tilt_vectors,
        normal_tilt_bounds=normal_tilt_bounds,
        normal_tilt_sample_count=normal_tilt_sample_count,
        normal_tilt_seed=normal_tilt_seed,
        contact_failure_scenarios=contact_failure_scenarios,
        geometry_scenarios=geometry_scenarios,
        num_directions=num_directions,
        tolerance=tolerance,
        solver_options=solver_options,
        verbose=verbose,
        timer=timer,
    )


def _prepare_geometry_scenarios(
    assembly,
    load_dofs,
    mu,
    density,
    external_forces,
    load_application_points,
    application_force_bound,
    interface_scale_factors,
    point_offset_vectors,
    normal_tilt_vectors,
    contact_failure_scenarios,
    geometry_scenarios,
    point_offset_bounds=None,
    point_offset_sample_count=60,
    point_offset_seed=0,
    normal_tilt_bounds=None,
    normal_tilt_sample_count=60,
    normal_tilt_seed=1,
):
    load_dofs = _validate_load_dofs(assembly, load_dofs)
    if geometry_scenarios is not None:
        if any(
            value is not None
            for value in (
                interface_scale_factors,
                point_offset_vectors,
                point_offset_bounds,
                normal_tilt_vectors,
                normal_tilt_bounds,
                contact_failure_scenarios,
            )
        ):
            raise ValueError("geometry_scenarios cannot be combined with generated geometry uncertainty inputs.")
        return load_dofs, _validate_geometry_scenarios(geometry_scenarios)

    scale_factors = _validate_scale_factors(interface_scale_factors)
    point_offset_scenarios = _point_offset_scenarios(
        assembly,
        point_offset_vectors,
        point_offset_bounds,
        point_offset_sample_count,
        point_offset_seed,
    )
    normal_tilt_scenarios = _normal_tilt_scenarios(
        assembly,
        normal_tilt_vectors,
        normal_tilt_bounds,
        normal_tilt_sample_count,
        normal_tilt_seed,
    )
    failures = _validate_failure_scenarios(contact_failure_scenarios)

    scenarios = []
    for scale, point_offset_scenario, normal_tilt_scenario, failure in itertools.product(
        scale_factors,
        point_offset_scenarios,
        normal_tilt_scenarios,
        failures,
    ):
        point_offsets, point_offset_name = point_offset_scenario
        normal_tilts, normal_tilt_name = normal_tilt_scenario
        scenario_assembly = copy.deepcopy(assembly)
        _apply_interface_geometry(
            scenario_assembly,
            scale=scale,
            point_offsets=point_offsets,
            normal_tilts=normal_tilts,
        )
        problem = _prepare_problem(
            scenario_assembly,
            load_dofs,
            mu,
            density,
            external_forces,
            load_application_points,
            application_force_bound,
        )
        scenario = GeometryScenarioProblem(
            equilibrium=problem.equilibrium.tocsr(),
            inequalities=problem.inequalities.tocsr(),
            inequality_rhs=np.asarray(problem.inequality_rhs, dtype=float).ravel(),
            baseline_load=np.asarray(problem.baseline_load, dtype=float).ravel(),
            load_projection=problem.load_projection.tocsr(),
            variable_count=problem.equilibrium.shape[1],
            name=_scenario_name(scale, point_offset_name, normal_tilt_name, failure),
        )
        if failure:
            scenario = _apply_contact_failures(scenario, scenario_assembly, failure)
        scenarios.append(scenario)

    return load_dofs, _validate_geometry_scenarios(scenarios)


def _validate_geometry_scenarios(geometry_scenarios):
    try:
        scenarios = list(geometry_scenarios)
    except TypeError as error:
        raise ValueError("geometry_scenarios must be a sequence of GeometryScenarioProblem objects.") from error
    if not scenarios:
        raise ValueError("geometry_scenarios must contain at least one scenario.")

    normalized = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, GeometryScenarioProblem):
            raise ValueError("geometry_scenarios[{}] must be a GeometryScenarioProblem.".format(index))
        equilibrium = csr_matrix(scenario.equilibrium)
        inequalities = csr_matrix(scenario.inequalities)
        load_projection = csr_matrix(scenario.load_projection)
        inequality_rhs = np.asarray(scenario.inequality_rhs, dtype=float).ravel()
        baseline_load = np.asarray(scenario.baseline_load, dtype=float).ravel()
        variable_count = int(scenario.variable_count)

        if variable_count <= 0:
            raise ValueError("geometry_scenarios[{}].variable_count must be positive.".format(index))
        if equilibrium.shape[1] != variable_count:
            raise ValueError(
                "geometry_scenarios[{}].equilibrium column count must match variable_count.".format(index)
            )
        if inequalities.shape[1] != variable_count:
            raise ValueError(
                "geometry_scenarios[{}].inequalities column count must match variable_count.".format(index)
            )
        if load_projection.shape != (2, variable_count):
            raise ValueError(
                "geometry_scenarios[{}].load_projection must have shape (2, variable_count).".format(index)
            )
        if baseline_load.shape[0] != equilibrium.shape[0]:
            raise ValueError("geometry_scenarios[{}].baseline_load length must match equilibrium rows.".format(index))
        if inequality_rhs.shape[0] != inequalities.shape[0]:
            raise ValueError("geometry_scenarios[{}].inequality_rhs length must match inequality rows.".format(index))

        normalized.append(
            GeometryScenarioProblem(
                equilibrium=equilibrium.tocsr(),
                inequalities=inequalities.tocsr(),
                inequality_rhs=inequality_rhs,
                baseline_load=baseline_load,
                load_projection=load_projection.tocsr(),
                variable_count=variable_count,
                name=scenario.name or "scenario {}".format(index),
            )
        )
    return normalized


def _validate_scale_factors(interface_scale_factors):
    if interface_scale_factors is None:
        return [1.0]
    try:
        values = list(interface_scale_factors)
    except TypeError as error:
        raise ValueError("interface_scale_factors must be a sequence of positive finite values.") from error
    if not values:
        raise ValueError("interface_scale_factors must contain at least one value.")
    factors = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("interface_scale_factors must contain positive finite values.")
        factor = float(value)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("interface_scale_factors must contain positive finite values.")
        factors.append(factor)
    return factors


def _validate_vectors(vectors, name, dimension, default):
    if vectors is None:
        return [np.asarray(default, dtype=float)]
    try:
        raw_vectors = list(vectors)
    except TypeError as error:
        raise ValueError("{} must be a sequence of finite vectors.".format(name)) from error
    if not raw_vectors:
        raise ValueError("{} must contain at least one vector.".format(name))

    normalized = []
    for index, vector in enumerate(raw_vectors):
        candidate = np.asarray(vector, dtype=float)
        if candidate.shape != (dimension,) or not np.all(np.isfinite(candidate)):
            raise ValueError("{}[{}] must be a finite {}D vector.".format(name, index, dimension))
        normalized.append(candidate)
    return normalized


def _point_offset_scenarios(assembly, point_offset_vectors, point_offset_bounds, sample_count, seed):
    if point_offset_vectors is not None:
        raise ValueError(
            "point_offset_vectors are no longer supported; use point_offset_bounds or geometry_scenarios."
        )
    if point_offset_bounds is None:
        return [(None, "point offsets nominal")]

    bounds = _validate_bound_vector(point_offset_bounds, "point_offset_bounds", 2)
    if np.allclose(bounds, 0.0):
        return [(None, "point offsets nominal")]
    count = _validate_sample_count(sample_count, "point_offset_sample_count")
    rng = np.random.default_rng(_validate_seed(seed, "point_offset_seed"))
    interfaces = _interface_point_counts(assembly)

    scenarios = [(None, "point offsets nominal")]
    for sample_index in range(1, count):
        offsets = {}
        for edge, interface_index, point_count in interfaces:
            offsets[(edge, interface_index)] = rng.uniform(-bounds, bounds, size=(point_count, 2))
        scenarios.append((offsets, "point offsets sample {}".format(sample_index)))
    return scenarios


def _normal_tilt_scenarios(assembly, normal_tilt_vectors, normal_tilt_bounds, sample_count, seed):
    if normal_tilt_vectors is not None and normal_tilt_bounds is not None:
        raise ValueError("normal_tilt_vectors cannot be combined with normal_tilt_bounds.")
    if normal_tilt_vectors is not None:
        vectors = _validate_vectors(normal_tilt_vectors, "normal_tilt_vectors", 2, [0.0, 0.0])
        return [(vector, "normal tilt {}".format(vector.tolist())) for vector in vectors]
    if normal_tilt_bounds is None:
        return [(None, "normal tilt nominal")]

    bounds = _validate_bound_vector(normal_tilt_bounds, "normal_tilt_bounds", 2)
    if np.allclose(bounds, 0.0):
        return [(None, "normal tilt nominal")]
    count = _validate_sample_count(sample_count, "normal_tilt_sample_count")
    rng = np.random.default_rng(_validate_seed(seed, "normal_tilt_seed"))
    interfaces = _interface_keys(assembly)

    scenarios = [(None, "normal tilt nominal")]
    for sample_index in range(1, count):
        tilts = {}
        for edge, interface_index in interfaces:
            tilts[(edge, interface_index)] = rng.uniform(-bounds, bounds, size=2)
        scenarios.append((tilts, "normal tilt sample {}".format(sample_index)))
    return scenarios


def _validate_bound_vector(bounds, name, dimension):
    if isinstance(bounds, bool):
        raise ValueError("{} must be a finite nonnegative scalar or {}D bound vector.".format(name, dimension))
    if np.isscalar(bounds):
        values = np.full(dimension, float(bounds), dtype=float)
    else:
        try:
            values = np.asarray(bounds, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "{} must be a finite nonnegative scalar or {}D bound vector.".format(name, dimension)
            ) from error
    if values.shape != (dimension,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("{} must be a finite nonnegative scalar or {}D bound vector.".format(name, dimension))
    return values


def _validate_sample_count(sample_count, name):
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise ValueError("{} must be a positive integer.".format(name))
    count = int(sample_count)
    if count <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return count


def _validate_seed(seed, name):
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("{} must be an integer or None.".format(name))
    return int(seed)


def _interface_point_counts(assembly):
    counts = []
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        for interface_index, interface in enumerate(interfaces):
            counts.append((edge, interface_index, len(interface.points)))
    return counts


def _interface_keys(assembly):
    keys = []
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        for interface_index, _ in enumerate(interfaces):
            keys.append((edge, interface_index))
    return keys


def _validate_failure_scenarios(contact_failure_scenarios):
    if contact_failure_scenarios is None:
        return [None]
    try:
        scenarios = list(contact_failure_scenarios)
    except TypeError as error:
        raise ValueError("contact_failure_scenarios must be a sequence.") from error
    if not scenarios:
        raise ValueError("contact_failure_scenarios must contain at least one scenario.")
    return scenarios


def _apply_interface_geometry(assembly, scale, point_offsets, normal_tilts):
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        for interface_index, interface in enumerate(interfaces):
            frame = interface.frame
            points = [_xyz_array(point, "interface point") for point in interface.points]
            centroid = np.mean(np.asarray(points), axis=0)
            local_offsets = _point_offsets_for_interface(point_offsets, edge, interface_index, len(points))
            xaxis = np.asarray(frame.xaxis, dtype=float)
            yaxis = np.asarray(frame.yaxis, dtype=float)
            perturbed_points = []
            for point, local_offset in zip(points, local_offsets):
                offset = float(local_offset[0]) * xaxis + float(local_offset[1]) * yaxis
                perturbed_points.append((centroid + scale * (point - centroid) + offset).tolist())
            interface.points = perturbed_points
            interface._frame = _tilted_frame(frame, _normal_tilt_for_interface(normal_tilts, edge, interface_index))


def _point_offsets_for_interface(point_offsets, edge, interface_index, point_count):
    if point_offsets is None:
        return np.zeros((point_count, 2))
    offsets = np.asarray(point_offsets[(edge, interface_index)], dtype=float)
    if offsets.shape != (point_count, 2):
        raise ValueError("Generated point-offset scenario has an invalid shape.")
    return offsets


def _normal_tilt_for_interface(normal_tilts, edge, interface_index):
    if normal_tilts is None:
        return np.zeros(2)
    if isinstance(normal_tilts, dict):
        return np.asarray(normal_tilts[(edge, interface_index)], dtype=float)
    return np.asarray(normal_tilts, dtype=float)


def _tilted_frame(frame, normal_tilt):
    theta_u = float(normal_tilt[0])
    theta_v = float(normal_tilt[1])
    tilted = frame.copy()
    if theta_u:
        tilted = tilted.transformed(Rotation.from_axis_and_angle(tilted.xaxis, theta_u, point=tilted.point))
    if theta_v:
        tilted = tilted.transformed(Rotation.from_axis_and_angle(tilted.yaxis, theta_v, point=tilted.point))
    return tilted


def _apply_contact_failures(scenario, assembly, failure):
    failed_contacts = _failed_contact_indices(assembly, failure)
    if not failed_contacts:
        return scenario

    rows = []
    columns = []
    data = []
    row = 0
    for contact_index in sorted(failed_contacts):
        for local_component in range(3):
            column = contact_index * 3 + local_component
            rows.extend([row, row + 1])
            columns.extend([column, column])
            data.extend([1.0, -1.0])
            row += 2

    zero_rows = csr_matrix((data, (rows, columns)), shape=(row, scenario.variable_count))
    return GeometryScenarioProblem(
        equilibrium=scenario.equilibrium,
        inequalities=vstack([scenario.inequalities, zero_rows], format="csr"),
        inequality_rhs=np.concatenate([scenario.inequality_rhs, np.zeros(row)]),
        baseline_load=scenario.baseline_load,
        load_projection=scenario.load_projection,
        variable_count=scenario.variable_count,
        name=scenario.name,
    )


def _failed_contact_indices(assembly, failure):
    failure = _normalize_failure(failure)
    contact_map = _contact_index_map(assembly)
    interface_failures = {
        _normalize_interface_selector(selector, assembly) for selector in failure.get("interfaces", [])
    }
    point_failures = {_normalize_point_selector(selector, assembly) for selector in failure.get("points", [])}

    failed_contacts = set()
    matched_interface_failures = set()
    matched_point_failures = set()
    for edge, interface_index, point_index, contact_index in contact_map:
        interface_key = (edge, interface_index)
        point_key = (edge, interface_index, point_index)
        if interface_key in interface_failures:
            matched_interface_failures.add(interface_key)
            failed_contacts.add(contact_index)
        if point_key in point_failures:
            matched_point_failures.add(point_key)
            failed_contacts.add(contact_index)
    if matched_interface_failures != interface_failures:
        missing = sorted(interface_failures - matched_interface_failures, key=repr)[0]
        raise ValueError("Unknown failed interface selector: {!r}.".format(missing))
    if matched_point_failures != point_failures:
        missing = sorted(point_failures - matched_point_failures, key=repr)[0]
        raise ValueError("Unknown failed point selector: {!r}.".format(missing))
    return failed_contacts


def _normalize_failure(failure):
    if failure is None:
        return {}
    if isinstance(failure, dict):
        normalized = {}
        normalized["interfaces"] = list(failure.get("interfaces", []))
        normalized["points"] = list(failure.get("points", failure.get("contact_points", [])))
        return normalized
    raise ValueError("Each contact failure scenario must be a dict or None.")


def _contact_index_map(assembly):
    mapping = []
    contact_index = 0
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        normalized_edge = _normalize_edge(edge, assembly)
        for interface_index, interface in enumerate(interfaces):
            for point_index, _ in enumerate(interface.points):
                mapping.append((normalized_edge, interface_index, point_index, contact_index))
                contact_index += 1
    return mapping


def _normalize_interface_selector(selector, assembly):
    if isinstance(selector, dict):
        edge = selector.get("edge")
        interface_index = selector.get("interface", selector.get("interface_index"))
    elif isinstance(selector, (list, tuple)) and len(selector) == 3:
        edge = (selector[0], selector[1])
        interface_index = selector[2]
    else:
        raise ValueError("Interface failure selectors must be (u, v, interface_index) or dicts.")
    return _normalize_edge(edge, assembly), _validate_nonnegative_int(interface_index, "interface index")


def _normalize_point_selector(selector, assembly):
    if isinstance(selector, dict):
        edge = selector.get("edge")
        interface_index = selector.get("interface", selector.get("interface_index"))
        point_index = selector.get("point", selector.get("point_index"))
    elif isinstance(selector, (list, tuple)) and len(selector) == 4:
        edge = (selector[0], selector[1])
        interface_index = selector[2]
        point_index = selector[3]
    else:
        raise ValueError("Point failure selectors must be (u, v, interface_index, point_index) or dicts.")
    return (
        _normalize_edge(edge, assembly),
        _validate_nonnegative_int(interface_index, "interface index"),
        _validate_nonnegative_int(point_index, "point index"),
    )


def _normalize_edge(edge, assembly):
    if not isinstance(edge, (list, tuple)) or len(edge) != 2:
        raise ValueError("Contact failure edge selectors must contain two node keys.")
    first, second = edge
    if assembly.graph.has_edge((first, second)):
        return first, second
    if assembly.graph.has_edge((second, first)):
        return second, first
    raise ValueError("Unknown contact failure edge: {!r}.".format(edge))


def _validate_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("{} must be a nonnegative integer.".format(name))
    return int(value)


def _scenario_name(scale, point_offset_name, normal_tilt_name, failure):
    return "scale={}, offset={}, tilt={}, failure={}".format(
        "{:.6g}".format(scale),
        point_offset_name,
        normal_tilt_name,
        "yes" if failure else "no",
    )


def _analyze_load_set_geometry(scenarios, options):
    origin_feasible = _is_origin_feasible_geometry(scenarios, options)
    feasibility_witness = _solve_load_set_feasibility_geometry(scenarios, options)
    cardinal_directions = (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    )
    cardinal_results = [
        _solve_primal_support_geometry(scenarios, direction, options) for direction in cardinal_directions
    ]
    is_bounded = all(status == "optimal" for status, _, _ in cardinal_results)

    if origin_feasible:
        feasible_center = np.zeros(2)
    elif is_bounded:
        feasible_center = np.mean(
            np.asarray([point for _, _, point in cardinal_results if point is not None]),
            axis=0,
        )
    else:
        feasible_center = feasibility_witness

    return _LoadSetAnalysis(
        origin_feasible=origin_feasible,
        is_bounded=is_bounded,
        feasible_center=np.asarray(feasible_center, dtype=float),
    )


def _is_origin_feasible_geometry(scenarios, options):
    equality_rows = []
    equality_rhs = []
    for scenario_index, scenario in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(scenarios, scenario_index, scenario.equilibrium, 0))
        equality_rhs.append(-scenario.baseline_load)
        equality_rows.append(_scenario_matrix_row(scenarios, scenario_index, scenario.load_projection, 0))
        equality_rhs.append(np.zeros(2))

    result = linprog(
        np.zeros(_total_variable_count(scenarios)),
        A_ub=_scenario_inequality_matrix(scenarios, 0),
        b_ub=np.concatenate([scenario.inequality_rhs for scenario in scenarios]),
        A_eq=vstack(equality_rows, format="csr"),
        b_eq=np.concatenate(equality_rhs),
        bounds=[(None, None)] * _total_variable_count(scenarios),
        method="highs",
        options=options,
    )
    if result.status == 0:
        return True
    if result.status == 2:
        return False
    raise RuntimeError("Geometry origin feasibility solve failed: {}.".format(result.message))


def _solve_load_set_feasibility_geometry(scenarios, options):
    total_variables = _total_variable_count(scenarios) + 2
    equality_rows = []
    equality_rhs = []
    for scenario_index, scenario in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(scenarios, scenario_index, scenario.equilibrium, 2))
        equality_rhs.append(-scenario.baseline_load)
        equality_rows.append(
            hstack(
                _scenario_blocks(scenarios, scenario_index, scenario.load_projection) + [-csr_matrix(np.eye(2))],
                format="csr",
            )
        )
        equality_rhs.append(np.zeros(2))

    result = linprog(
        np.zeros(total_variables),
        A_ub=_scenario_inequality_matrix(scenarios, 2),
        b_ub=np.concatenate([scenario.inequality_rhs for scenario in scenarios]),
        A_eq=vstack(equality_rows, format="csr"),
        b_eq=np.concatenate(equality_rhs),
        bounds=[(None, None)] * total_variables,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return np.asarray(result.x[-2:], dtype=float)
    if result.status == 2:
        raise ValueError("The safe load set is empty for compression-only RBE.")
    raise RuntimeError("Geometry safe load-set feasibility solve failed: {}.".format(result.message))


def _solve_primal_support_geometry(scenarios, direction, options):
    total_variables = _total_variable_count(scenarios) + 2
    objective = np.zeros(total_variables)
    objective[-2:] = -np.asarray(direction, dtype=float)

    equality_rows = []
    equality_rhs = []
    for scenario_index, scenario in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(scenarios, scenario_index, scenario.equilibrium, 2))
        equality_rhs.append(-scenario.baseline_load)
        equality_rows.append(
            hstack(
                _scenario_blocks(scenarios, scenario_index, scenario.load_projection) + [-csr_matrix(np.eye(2))],
                format="csr",
            )
        )
        equality_rhs.append(np.zeros(2))

    result = linprog(
        objective,
        A_ub=_scenario_inequality_matrix(scenarios, 2),
        b_ub=np.concatenate([scenario.inequality_rhs for scenario in scenarios]),
        A_eq=vstack(equality_rows, format="csr"),
        b_eq=np.concatenate(equality_rhs),
        bounds=[(None, None)] * total_variables,
        method="highs",
        options=options,
    )
    if result.status == 0:
        point = np.asarray(result.x[-2:], dtype=float)
        support = float(np.asarray(direction, dtype=float).dot(point))
        return "optimal", support, point.tolist()
    if result.status == 3:
        return "unbounded", None, None
    raise RuntimeError("Geometry primal support solve failed: {}.".format(result.message))


def _solve_radial_geometry(scenarios, direction, center, options):
    total_variables = _total_variable_count(scenarios) + 1
    objective = np.zeros(total_variables)
    objective[-1] = -1.0
    direction_column = csr_matrix(-np.asarray(direction, dtype=float).reshape((2, 1)))

    equality_rows = []
    equality_rhs = []
    for scenario_index, scenario in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(scenarios, scenario_index, scenario.equilibrium, 1))
        equality_rhs.append(-scenario.baseline_load)
        equality_rows.append(
            hstack(
                _scenario_blocks(scenarios, scenario_index, scenario.load_projection) + [direction_column],
                format="csr",
            )
        )
        equality_rhs.append(np.asarray(center, dtype=float))

    result = linprog(
        objective,
        A_ub=_scenario_inequality_matrix(scenarios, 1),
        b_ub=np.concatenate([scenario.inequality_rhs for scenario in scenarios]),
        A_eq=vstack(equality_rows, format="csr"),
        b_eq=np.concatenate(equality_rhs),
        bounds=[(None, None)] * _total_variable_count(scenarios) + [(0, None)],
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", max(0.0, float(result.x[-1]))
    if result.status == 3:
        return "unbounded", None
    raise RuntimeError("Geometry radial load solve failed: {}.".format(result.message))


def _solve_dual_support_geometry(scenarios, direction, options):
    objective = []
    bounds = []
    dual_rows = []
    dual_rhs = []
    block_sizes = []
    variable_counts = []
    scenario_blocks = []

    for scenario in scenarios:
        equality_rows = scenario.equilibrium.shape[0]
        inequality_rows = scenario.inequalities.shape[0]
        block_size = equality_rows + inequality_rows + 2
        block_sizes.append(block_size)
        variable_counts.append(scenario.variable_count)
        objective.extend(np.concatenate([-scenario.baseline_load, scenario.inequality_rhs, np.zeros(2)]))
        bounds.extend([(None, None)] * equality_rows)
        bounds.extend([(0, None)] * inequality_rows)
        bounds.extend([(None, None)] * 2)

        scenario_block = hstack(
            [
                scenario.equilibrium.transpose(),
                scenario.inequalities.transpose(),
                -scenario.load_projection.transpose(),
            ],
            format="csr",
        )
        scenario_blocks.append(scenario_block)

    for scenario_index, scenario_block in enumerate(scenario_blocks):
        dual_rows.append(_dual_scenario_row(scenario_block, scenario_index, block_sizes, variable_counts))
        dual_rhs.append(np.zeros(variable_counts[scenario_index]))

    eta_rows = []
    for scenario, block_size in zip(scenarios, block_sizes):
        equality_rows = scenario.equilibrium.shape[0]
        inequality_rows = scenario.inequalities.shape[0]
        eta_rows.append(
            hstack(
                [
                    csr_matrix((2, equality_rows)),
                    csr_matrix((2, inequality_rows)),
                    csr_matrix(np.eye(2)),
                ],
                format="csr",
            )
        )
    dual_rows.append(hstack(eta_rows, format="csr"))
    dual_rhs.append(np.asarray(direction, dtype=float))

    result = linprog(
        np.asarray(objective, dtype=float),
        A_eq=vstack(dual_rows, format="csr"),
        b_eq=np.concatenate(dual_rhs),
        bounds=bounds,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", float(result.fun)
    if result.status == 2:
        return "unbounded", None
    raise RuntimeError("Geometry dual support solve failed: {}.".format(result.message))


def _total_variable_count(scenarios):
    return sum(scenario.variable_count for scenario in scenarios)


def _scenario_blocks(scenarios, scenario_index, matrix):
    return [
        matrix if index == scenario_index else csr_matrix((matrix.shape[0], scenario.variable_count))
        for index, scenario in enumerate(scenarios)
    ]


def _scenario_matrix_row(scenarios, scenario_index, matrix, trailing_columns):
    return hstack(
        _scenario_blocks(scenarios, scenario_index, matrix) + [csr_matrix((matrix.shape[0], trailing_columns))],
        format="csr",
    )


def _scenario_inequality_matrix(scenarios, trailing_columns):
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        rows.append(_scenario_matrix_row(scenarios, scenario_index, scenario.inequalities, trailing_columns))
    return vstack(rows, format="csr")


def _dual_scenario_row(matrix, scenario_index, block_sizes, row_counts):
    return hstack(
        [
            matrix if index == scenario_index else csr_matrix((row_counts[scenario_index], block_size))
            for index, block_size in enumerate(block_sizes)
        ],
        format="csr",
    )


def _report_geometry(result, scenario_count, start_time, verbose, timer):
    if verbose:
        optimal_count = result.statuses.count("optimal")
        print(
            "geometry uncertainty RBE {}: {} scenarios; {} of {} directions bounded; load set bounded={}".format(
                result.method,
                scenario_count,
                optimal_count,
                len(result.statuses),
                result.is_bounded,
            )
        )
    if timer:
        print("--- geometry uncertainty RBE time: {} seconds ---".format(time.time() - start_time))
