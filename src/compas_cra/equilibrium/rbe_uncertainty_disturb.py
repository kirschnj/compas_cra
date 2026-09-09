r"""Robust RBE safe-load analysis with polyhedral external load uncertainty.

This module separates the visible two-dimensional safe load from disturbance
loads. The selected ``load_dofs`` define the plotted load ``u``. Disturbance
uncertainty is introduced independently through ``uncertainty_load_dofs`` or a
direct ``uncertainty_basis`` matrix ``W``. Foundation tilt uncertainty is
represented as scenario-dependent gravity direction changes while keeping
geometry, contact normals, friction cones, and structural matrices fixed. Tilt
safe regions are reported in the world XZ load frame by default; each scenario
maps the common world visible load into the local tilted structure frame.

For uncertainty vertices ``xi_s``, the robust safe set is

.. math::

    U_R = \{u \mid \forall s,\ \exists x_s:
    A x_s = -(p_s + W \xi_s),\ H x_s \leq h,\ P x_s = T_s u\}.

The implementation enumerates polytope vertices and therefore remains a linear
program for the linearized RBE friction cone.
"""

import math
import time
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from compas_assembly.datastructures import Assembly
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from .cra_helper import free_nodes
from .rbe_robust import _COMPONENT_INDEX
from .rbe_robust import RobustForceResult
from .rbe_robust import _directions
from .rbe_robust import _inner_polygon
from .rbe_robust import _LoadSetAnalysis
from .rbe_robust import _outer_polygon
from .rbe_robust import _prepare_problem
from .rbe_robust import _validate_tolerance


@dataclass(frozen=True)
class _UncertainScenario:
    """One load/disturbance scenario with a visible-load frame transform."""

    load: np.ndarray
    load_transform: np.ndarray


def rbe_uncertainty_disturb_sample(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    uncertainty_vertices: Optional[Sequence[Sequence[float]]] = None,
    uncertainty_load_dofs: Optional[Sequence[Tuple[Any, str]]] = None,
    uncertainty_basis=None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
    tilt_angle_bounds: Optional[Sequence[float]] = None,
    tilt_angle_count: int = 2,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_load_frame: str = "world",
) -> RobustForceResult:
    """Approximate the disturbance-robust safe load range by radial sampling."""
    start_time = time.time()
    problem = _prepare_problem(
        assembly,
        load_dofs,
        mu,
        density,
        external_forces,
        load_application_points,
        application_force_bound,
    )
    scenarios = _scenario_records(
        problem,
        assembly=assembly,
        density=density,
        external_forces=external_forces,
        uncertainty_load_dofs=uncertainty_load_dofs,
        uncertainty_vertices=uncertainty_vertices,
        uncertainty_basis=uncertainty_basis,
        tilt_angle_bounds=tilt_angle_bounds,
        tilt_angle_count=tilt_angle_count,
        tilt_angles=tilt_angles,
        tilt_load_frame=tilt_load_frame,
    )
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set_uncertain(problem, scenarios, options)

    statuses = []
    radial_limits = []
    boundary_points = []
    for direction in directions:
        status, limit = _solve_radial_uncertain(problem, direction, analysis.feasible_center, scenarios, options)
        statuses.append(status)
        radial_limits.append(limit)
        if limit is not None:
            boundary_points.append((analysis.feasible_center + limit * direction).tolist())

    inner_polygon = _inner_polygon(boundary_points, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="sample",
        load_dofs=problem.load_dofs,
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
    _report_uncertain(result, start_time, verbose, timer)
    return result


def rbe_uncertainty_disturb_support_primal(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    uncertainty_vertices: Optional[Sequence[Sequence[float]]] = None,
    uncertainty_load_dofs: Optional[Sequence[Tuple[Any, str]]] = None,
    uncertainty_basis=None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
    tilt_angle_bounds: Optional[Sequence[float]] = None,
    tilt_angle_count: int = 2,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_load_frame: str = "world",
) -> RobustForceResult:
    """Approximate the disturbance-robust safe load range with primal supports."""
    start_time = time.time()
    problem = _prepare_problem(
        assembly,
        load_dofs,
        mu,
        density,
        external_forces,
        load_application_points,
        application_force_bound,
    )
    scenarios = _scenario_records(
        problem,
        assembly=assembly,
        density=density,
        external_forces=external_forces,
        uncertainty_load_dofs=uncertainty_load_dofs,
        uncertainty_vertices=uncertainty_vertices,
        uncertainty_basis=uncertainty_basis,
        tilt_angle_bounds=tilt_angle_bounds,
        tilt_angle_count=tilt_angle_count,
        tilt_angles=tilt_angles,
        tilt_load_frame=tilt_load_frame,
    )
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set_uncertain(problem, scenarios, options)

    statuses = []
    support_values = []
    support_points = []
    halfspaces = []
    for direction in directions:
        status, support, point = _solve_primal_support_uncertain(problem, direction, scenarios, options)
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
        load_dofs=problem.load_dofs,
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
    _report_uncertain(result, start_time, verbose, timer)
    return result


def rbe_uncertainty_disturb_support_dual(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    uncertainty_vertices: Optional[Sequence[Sequence[float]]] = None,
    uncertainty_load_dofs: Optional[Sequence[Tuple[Any, str]]] = None,
    uncertainty_basis=None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
    tilt_angle_bounds: Optional[Sequence[float]] = None,
    tilt_angle_count: int = 2,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_load_frame: str = "world",
) -> RobustForceResult:
    """Approximate the disturbance-robust safe load range with dual supports."""
    start_time = time.time()
    problem = _prepare_problem(
        assembly,
        load_dofs,
        mu,
        density,
        external_forces,
        load_application_points,
        application_force_bound,
    )
    scenarios = _scenario_records(
        problem,
        assembly=assembly,
        density=density,
        external_forces=external_forces,
        uncertainty_load_dofs=uncertainty_load_dofs,
        uncertainty_vertices=uncertainty_vertices,
        uncertainty_basis=uncertainty_basis,
        tilt_angle_bounds=tilt_angle_bounds,
        tilt_angle_count=tilt_angle_count,
        tilt_angles=tilt_angles,
        tilt_load_frame=tilt_load_frame,
    )
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set_uncertain(problem, scenarios, options)

    statuses = []
    support_values = []
    halfspaces = []
    for direction in directions:
        status, support = _solve_dual_support_uncertain(problem, direction, scenarios, options)
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])

    outer_polygon = _outer_polygon(halfspaces, tolerance) if analysis.is_bounded else []
    result = RobustForceResult(
        method="support",
        load_dofs=problem.load_dofs,
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
    _report_uncertain(result, start_time, verbose, timer)
    return result


def rbe_uncertainty_disturb_support(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    uncertainty_vertices: Optional[Sequence[Sequence[float]]] = None,
    uncertainty_load_dofs: Optional[Sequence[Tuple[Any, str]]] = None,
    uncertainty_basis=None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
    tilt_angle_bounds: Optional[Sequence[float]] = None,
    tilt_angle_count: int = 2,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_load_frame: str = "world",
) -> RobustForceResult:
    """Alias for :func:`rbe_uncertainty_disturb_support_dual`."""
    return rbe_uncertainty_disturb_support_dual(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        uncertainty_vertices=uncertainty_vertices,
        uncertainty_load_dofs=uncertainty_load_dofs,
        uncertainty_basis=uncertainty_basis,
        num_directions=num_directions,
        tolerance=tolerance,
        solver_options=solver_options,
        verbose=verbose,
        timer=timer,
        tilt_angle_bounds=tilt_angle_bounds,
        tilt_angle_count=tilt_angle_count,
        tilt_angles=tilt_angles,
        tilt_load_frame=tilt_load_frame,
    )


def rbe_uncertainty_disturb(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    uncertainty_vertices: Optional[Sequence[Sequence[float]]] = None,
    uncertainty_load_dofs: Optional[Sequence[Tuple[Any, str]]] = None,
    uncertainty_basis=None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
    tilt_angle_bounds: Optional[Sequence[float]] = None,
    tilt_angle_count: int = 2,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_load_frame: str = "world",
) -> RobustForceResult:
    """Alias for :func:`rbe_uncertainty_disturb_support_dual`."""
    return rbe_uncertainty_disturb_support_dual(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        uncertainty_vertices=uncertainty_vertices,
        uncertainty_load_dofs=uncertainty_load_dofs,
        uncertainty_basis=uncertainty_basis,
        num_directions=num_directions,
        tolerance=tolerance,
        solver_options=solver_options,
        verbose=verbose,
        timer=timer,
        tilt_angle_bounds=tilt_angle_bounds,
        tilt_angle_count=tilt_angle_count,
        tilt_angles=tilt_angles,
        tilt_load_frame=tilt_load_frame,
    )


def _load_basis_nd(assembly, load_dofs, row_count):
    """Build a free-body wrench basis for any number of load DOFs."""
    try:
        load_dofs = list(load_dofs)
    except TypeError as error:
        raise ValueError("uncertainty_load_dofs must be a sequence of (node, component) pairs.") from error
    if not load_dofs:
        raise ValueError("uncertainty_load_dofs must contain at least one degree of freedom.")

    node_keys = list(assembly.graph.nodes())
    node_index = {node: index for index, node in enumerate(node_keys)}
    free = free_nodes(assembly)
    rows = []
    columns = []
    for column, load_dof in enumerate(load_dofs):
        if not isinstance(load_dof, (list, tuple)) or len(load_dof) != 2:
            raise ValueError("Each uncertainty load degree of freedom must be a (node, component) pair.")
        node, component = load_dof
        if node not in node_index:
            raise ValueError("Unknown uncertainty load node: {!r}.".format(node))
        if assembly.graph.node_attribute(node, "is_support"):
            raise ValueError("Uncertainty load degree of freedom node {!r} is a support.".format(node))
        if not isinstance(component, str) or component.lower() not in _COMPONENT_INDEX:
            raise ValueError("Unknown uncertainty load component: {!r}.".format(component))
        free_position = free.index(node_index[node])
        rows.append(free_position * 6 + _COMPONENT_INDEX[component.lower()])
        columns.append(column)
    return csr_matrix((np.ones(len(load_dofs)), (rows, columns)), shape=(row_count, len(load_dofs)))


def _uncertainty_shifts(
    problem,
    assembly=None,
    uncertainty_load_dofs=None,
    uncertainty_vertices=None,
    uncertainty_basis=None,
):
    """Return one baseline-load shift per uncertainty vertex."""
    row_count = problem.equilibrium.shape[0]
    if uncertainty_vertices is None:
        return [np.zeros(row_count)]

    vertices = np.asarray(uncertainty_vertices, dtype=float)
    if vertices.ndim != 2:
        raise ValueError("uncertainty_vertices must be a 2D array.")
    if vertices.shape[0] == 0:
        raise ValueError("uncertainty_vertices must contain at least one vertex.")

    if uncertainty_basis is None:
        if uncertainty_load_dofs is None:
            raise ValueError("Provide uncertainty_basis or uncertainty_load_dofs.")
        if assembly is None:
            raise ValueError("assembly is required when using uncertainty_load_dofs.")
        basis = _load_basis_nd(assembly, uncertainty_load_dofs, row_count)
    else:
        basis = csr_matrix(uncertainty_basis)

    if basis.shape[0] != row_count:
        raise ValueError("uncertainty_basis row count must match equilibrium rows.")
    if basis.shape[1] != vertices.shape[1]:
        raise ValueError("uncertainty vertex dimension must match uncertainty_basis columns.")
    return [np.asarray(basis.dot(vertex)).ravel() for vertex in vertices]


def _scenario_records(
    problem,
    assembly,
    density,
    external_forces,
    uncertainty_load_dofs,
    uncertainty_vertices,
    uncertainty_basis,
    tilt_angle_bounds,
    tilt_angle_count,
    tilt_angles=None,
    tilt_load_frame="world",
):
    """Return scenario load vectors and visible-load transforms."""
    tilt_load_frame = _validate_tilt_load_frame(tilt_load_frame)
    shifts = _uncertainty_shifts(
        problem,
        assembly=assembly,
        uncertainty_load_dofs=uncertainty_load_dofs,
        uncertainty_vertices=uncertainty_vertices,
        uncertainty_basis=uncertainty_basis,
    )
    tilt_scenarios = _tilt_scenarios(
        problem,
        assembly=assembly,
        density=density,
        external_forces=external_forces,
        tilt_angle_bounds=tilt_angle_bounds,
        tilt_angle_count=tilt_angle_count,
        tilt_angles=tilt_angles,
        tilt_load_frame=tilt_load_frame,
    )
    return [
        _UncertainScenario(load=baseline + shift, load_transform=load_transform)
        for baseline, load_transform in tilt_scenarios
        for shift in shifts
    ]


def _scenario_loads(
    problem,
    assembly,
    density,
    external_forces,
    uncertainty_load_dofs,
    uncertainty_vertices,
    uncertainty_basis,
    tilt_angle_bounds,
    tilt_angle_count,
    tilt_angles=None,
    tilt_load_frame="world",
):
    """Return full baseline load vectors for disturbance and tilt scenarios."""
    return [
        scenario.load
        for scenario in _scenario_records(
            problem,
            assembly=assembly,
            density=density,
            external_forces=external_forces,
            uncertainty_load_dofs=uncertainty_load_dofs,
            uncertainty_vertices=uncertainty_vertices,
            uncertainty_basis=uncertainty_basis,
            tilt_angle_bounds=tilt_angle_bounds,
            tilt_angle_count=tilt_angle_count,
            tilt_angles=tilt_angles,
            tilt_load_frame=tilt_load_frame,
        )
    ]


def _tilt_scenarios(
    problem,
    assembly,
    density,
    external_forces,
    tilt_angle_bounds,
    tilt_angle_count,
    tilt_angles,
    tilt_load_frame,
):
    """Return tilted baseline loads and world-to-local visible-load transforms."""
    if tilt_angles is not None and tilt_angle_bounds is not None:
        raise ValueError("Provide tilt_angles or tilt_angle_bounds, not both.")
    identity = np.eye(2)
    if tilt_angles is None and tilt_angle_bounds is None:
        return [(problem.baseline_load, identity)]

    angles = (
        _explicit_tilt_angles(tilt_angles)
        if tilt_angles is not None
        else _tilt_angles(
            tilt_angle_bounds,
            tilt_angle_count,
        )
    )
    return [
        (
            _tilted_gravity_load(assembly, density, external_forces, angle, problem.equilibrium.shape[0]),
            identity if tilt_load_frame == "structure" else _tilt_load_transform(angle, problem.load_dofs),
        )
        for angle in angles
    ]


def _validate_tilt_load_frame(tilt_load_frame):
    """Return a normalized tilt load-frame option."""
    if not isinstance(tilt_load_frame, str):
        raise ValueError("tilt_load_frame must be 'world' or 'structure'.")
    normalized = tilt_load_frame.lower()
    if normalized not in ("world", "structure"):
        raise ValueError("tilt_load_frame must be 'world' or 'structure'.")
    return normalized


def _explicit_tilt_angles(tilt_angles):
    """Return finite explicit tilt angles."""
    try:
        angles = np.asarray(list(tilt_angles), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("tilt_angles must be a one-dimensional sequence of finite angles.") from error
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("tilt_angles must be a one-dimensional sequence of finite angles.")
    return angles


def _tilt_load_transform(theta, load_dofs):
    """Map world ``Fx/Fz`` load components into the tilted structure frame."""
    nodes = [node for node, _ in load_dofs]
    components = [component for _, component in load_dofs]
    if nodes[0] != nodes[1] or components != ["fx", "fz"]:
        raise ValueError(
            "world-frame tilt load transformation requires same-node load_dofs [(node, 'fx'), (node, 'fz')]."
        )
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=float)


def _tilted_gravity_loads(
    assembly,
    density,
    external_forces,
    tilt_angle_bounds,
    tilt_angle_count,
    row_count,
):
    """Return baseline loads for uniformly sampled XZ tilt-angle scenarios."""
    angles = _tilt_angles(tilt_angle_bounds, tilt_angle_count)
    return [_tilted_gravity_load(assembly, density, external_forces, angle, row_count) for angle in angles]


def _tilt_angles(tilt_angle_bounds, tilt_angle_count):
    """Return uniformly sampled tilt angles from finite increasing bounds."""
    try:
        lower, upper = tilt_angle_bounds
    except (TypeError, ValueError) as error:
        raise ValueError("tilt_angle_bounds must contain exactly two values.") from error
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("tilt_angle_bounds must contain two finite increasing values.")
    if isinstance(tilt_angle_count, bool) or not isinstance(tilt_angle_count, (int, np.integer)):
        raise ValueError("tilt_angle_count must be an integer greater than or equal to 2.")
    if tilt_angle_count < 2:
        raise ValueError("tilt_angle_count must be an integer greater than or equal to 2.")
    return np.linspace(lower, upper, int(tilt_angle_count))


def _tilted_gravity_vector(theta):
    """Return the unit gravity vector for one XZ foundation tilt angle."""
    return np.asarray([-math.sin(theta), 0.0, -math.cos(theta)], dtype=float)


def _tilted_gravity_load(assembly, density, external_forces, theta, row_count):
    """Return one free-body baseline load vector for an XZ tilt angle."""
    external_forces = {} if external_forces is None else external_forces
    node_keys = list(assembly.graph.nodes())
    free = free_nodes(assembly)
    gravity = _tilted_gravity_vector(theta)

    wrenches = []
    for node in node_keys:
        block = assembly.node_block(node)
        block_density = block.attributes["density"] if "density" in block.attributes else density
        wrench = np.zeros(6, dtype=float)
        wrench[:3] = block.volume() * block_density * gravity
        if node in external_forces:
            external_wrench = np.asarray(external_forces[node], dtype=float)
            if external_wrench.shape != (6,):
                raise ValueError("external_forces[{!r}] must contain six wrench components.".format(node))
            wrench += external_wrench
        wrenches.append(wrench)

    load = np.asarray(wrenches, dtype=float)[free, :].reshape((-1,), order="C")
    if load.shape[0] != row_count:
        raise ValueError("Tilted gravity load row count must match equilibrium rows.")
    return load


def _analyze_load_set_uncertain(problem, scenarios, options):
    """Find a robust feasible load center and determine global boundedness."""
    origin_feasible = _is_origin_feasible_uncertain(problem, scenarios, options)
    feasibility_witness = _solve_load_set_feasibility_uncertain(problem, scenarios, options)
    cardinal_directions = (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    )
    cardinal_results = [
        _solve_primal_support_uncertain(problem, direction, scenarios, options) for direction in cardinal_directions
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


def _is_origin_feasible_uncertain(problem, scenarios, options):
    """Return whether zero visible load is feasible for every disturbance vertex."""
    variable_count = problem.equilibrium.shape[1]
    scenario_count = len(scenarios)
    equality_rows = []
    equality_rhs = []
    for scenario, scenario_record in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(problem.equilibrium, scenario, scenario_count, variable_count, 0))
        equality_rhs.append(-scenario_record.load)
        equality_rows.append(
            _scenario_matrix_row(problem.load_projection, scenario, scenario_count, variable_count, 0)
        )
        equality_rhs.append(np.zeros(2))

    result = linprog(
        np.zeros(scenario_count * variable_count),
        A_ub=_scenario_inequality_matrix(problem, scenario_count, 0),
        b_ub=np.tile(problem.inequality_rhs, scenario_count),
        A_eq=vstack(equality_rows, format="csr"),
        b_eq=np.concatenate(equality_rhs),
        bounds=[(None, None)] * (scenario_count * variable_count),
        method="highs",
        options=options,
    )
    if result.status == 0:
        return True
    if result.status == 2:
        return False
    raise RuntimeError("Uncertain origin feasibility solve failed: {}.".format(result.message))


def _solve_load_set_feasibility_uncertain(problem, scenarios, options):
    """Return one common feasible visible load for every disturbance vertex."""
    variable_count = problem.equilibrium.shape[1]
    scenario_count = len(scenarios)
    total_variables = scenario_count * variable_count + 2
    equality_rows = []
    equality_rhs = []
    for scenario, scenario_record in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(problem.equilibrium, scenario, scenario_count, variable_count, 2))
        equality_rhs.append(-scenario_record.load)
        equality_rows.append(
            hstack(
                _scenario_blocks(problem.load_projection, scenario, scenario_count, variable_count)
                + [-csr_matrix(scenario_record.load_transform)],
                format="csr",
            )
        )
        equality_rhs.append(np.zeros(2))

    result = linprog(
        np.zeros(total_variables),
        A_ub=_scenario_inequality_matrix(problem, scenario_count, 2),
        b_ub=np.tile(problem.inequality_rhs, scenario_count),
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
    raise RuntimeError("Uncertain safe load-set feasibility solve failed: {}.".format(result.message))


def _solve_primal_support_uncertain(problem, direction, scenarios, options):
    """Solve the robust primal support LP for one direction."""
    variable_count = problem.equilibrium.shape[1]
    scenario_count = len(scenarios)
    total_variables = scenario_count * variable_count + 2
    objective = np.zeros(total_variables)
    objective[-2:] = -np.asarray(direction, dtype=float)

    equality_rows = []
    equality_rhs = []
    for scenario, scenario_record in enumerate(scenarios):
        equality_rows.append(_scenario_matrix_row(problem.equilibrium, scenario, scenario_count, variable_count, 2))
        equality_rhs.append(-scenario_record.load)
        equality_rows.append(
            hstack(
                _scenario_blocks(problem.load_projection, scenario, scenario_count, variable_count)
                + [-csr_matrix(scenario_record.load_transform)],
                format="csr",
            )
        )
        equality_rhs.append(np.zeros(2))

    result = linprog(
        objective,
        A_ub=_scenario_inequality_matrix(problem, scenario_count, 2),
        b_ub=np.tile(problem.inequality_rhs, scenario_count),
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
    raise RuntimeError("Uncertain primal support solve failed: {}.".format(result.message))


def _solve_radial_uncertain(problem, direction, center, scenarios, options):
    """Solve the robust radial LP for one direction from a robust feasible center."""
    variable_count = problem.equilibrium.shape[1]
    scenario_count = len(scenarios)
    total_variables = scenario_count * variable_count + 1
    objective = np.zeros(total_variables)
    objective[-1] = -1.0
    direction = np.asarray(direction, dtype=float)
    center = np.asarray(center, dtype=float)

    equality_rows = []
    equality_rhs = []
    for scenario, scenario_record in enumerate(scenarios):
        direction_column = csr_matrix(-scenario_record.load_transform.dot(direction).reshape((2, 1)))
        equality_rows.append(_scenario_matrix_row(problem.equilibrium, scenario, scenario_count, variable_count, 1))
        equality_rhs.append(-scenario_record.load)
        equality_rows.append(
            hstack(
                _scenario_blocks(problem.load_projection, scenario, scenario_count, variable_count)
                + [direction_column],
                format="csr",
            )
        )
        equality_rhs.append(scenario_record.load_transform.dot(center))

    result = linprog(
        objective,
        A_ub=_scenario_inequality_matrix(problem, scenario_count, 1),
        b_ub=np.tile(problem.inequality_rhs, scenario_count),
        A_eq=vstack(equality_rows, format="csr"),
        b_eq=np.concatenate(equality_rhs),
        bounds=[(None, None)] * (scenario_count * variable_count) + [(0, None)],
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", max(0.0, float(result.x[-1]))
    if result.status == 3:
        return "unbounded", None
    raise RuntimeError("Uncertain radial load solve failed: {}.".format(result.message))


def _solve_dual_support_uncertain(problem, direction, scenarios, options):
    """Solve the robust dual support LP for one direction."""
    equality_rows = problem.equilibrium.shape[0]
    inequality_rows = problem.inequalities.shape[0]
    variable_count = problem.equilibrium.shape[1]
    scenario_count = len(scenarios)
    block_size = equality_rows + inequality_rows + 2

    objective = np.concatenate(
        [np.concatenate([-scenario.load, problem.inequality_rhs, np.zeros(2)]) for scenario in scenarios]
    )

    dual_rows = []
    dual_rhs = []
    scenario_block = hstack(
        [problem.equilibrium.transpose(), problem.inequalities.transpose(), -problem.load_projection.transpose()],
        format="csr",
    )
    for scenario in range(scenario_count):
        dual_rows.append(_dual_scenario_row(scenario_block, scenario, scenario_count, variable_count, block_size))
        dual_rhs.append(np.zeros(variable_count))

    eta_selectors = [
        hstack(
            [
                csr_matrix((2, equality_rows)),
                csr_matrix((2, inequality_rows)),
                csr_matrix(scenario.load_transform.transpose()),
            ],
            format="csr",
        )
        for scenario in scenarios
    ]
    dual_rows.append(hstack(eta_selectors, format="csr"))
    dual_rhs.append(np.asarray(direction, dtype=float))

    bounds = []
    for _ in range(scenario_count):
        bounds.extend([(None, None)] * equality_rows)
        bounds.extend([(0, None)] * inequality_rows)
        bounds.extend([(None, None)] * 2)

    result = linprog(
        objective,
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
    raise RuntimeError("Uncertain dual support solve failed: {}.".format(result.message))


def _scenario_blocks(matrix, scenario, scenario_count, variable_count):
    """Return sparse blocks with ``matrix`` inserted at one scenario column block."""
    return [
        matrix if index == scenario else csr_matrix((matrix.shape[0], variable_count))
        for index in range(scenario_count)
    ]


def _scenario_matrix_row(matrix, scenario, scenario_count, variable_count, trailing_columns):
    """Return a sparse row for one scenario block plus optional trailing columns."""
    return hstack(
        _scenario_blocks(matrix, scenario, scenario_count, variable_count)
        + [csr_matrix((matrix.shape[0], trailing_columns))],
        format="csr",
    )


def _scenario_inequality_matrix(problem, scenario_count, trailing_columns):
    """Return block-scenario inequalities ``H x_s <= h``."""
    rows = []
    for scenario in range(scenario_count):
        rows.append(
            _scenario_matrix_row(
                problem.inequalities,
                scenario,
                scenario_count,
                problem.equilibrium.shape[1],
                trailing_columns,
            )
        )
    return vstack(rows, format="csr")


def _dual_scenario_row(matrix, scenario, scenario_count, row_count, block_size):
    """Return a dual equality row with one scenario block active."""
    return hstack(
        [matrix if index == scenario else csr_matrix((row_count, block_size)) for index in range(scenario_count)],
        format="csr",
    )


def _report_uncertain(result, start_time, verbose, timer):
    """Print an optional compact uncertainty analysis report."""
    if verbose:
        optimal_count = result.statuses.count("optimal")
        print(
            "uncertainty RBE {}: {} of {} directions bounded; load set bounded={}".format(
                result.method,
                optimal_count,
                len(result.statuses),
                result.is_bounded,
            )
        )
    if timer:
        print("--- uncertainty RBE time: {} seconds ---".format(time.time() - start_time))
