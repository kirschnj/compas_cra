r"""Robust rigid-block equilibrium force-range analysis.

For ``B`` free blocks and ``V`` interface vertices, the safe load set is

.. math::

    U = \{u \in \mathbb{R}^2 \mid \exists x:
    A x = -p_0,\ H x \leq h,\ u = P x\}.

Here, ``A`` is the equilibrium matrix, ``H`` stores friction, compression, and
optional load-application bounds, ``p0`` is the baseline wrench vector, and
``P`` projects decision variables to the two visible load components. In the
legacy center-applied case, ``x`` contains contact-force components followed by
the two visible load increments. When load-application points are provided,
``x`` instead contains contact-force components followed by hidden point-force
components whose sum and generated moments define the visible load.
"""

import math
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from compas_assembly.datastructures import Assembly
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from .cra_helper import equilibrium_setup
from .cra_helper import external_force_setup
from .cra_helper import free_nodes
from .cra_helper import friction_setup
from .cra_helper import num_vertices

_COMPONENT_INDEX = {
    "fx": 0,
    "fy": 1,
    "fz": 2,
    "mx": 3,
    "my": 4,
    "mz": 5,
}
_FORCE_COMPONENTS = {"fx", "fy", "fz"}


@dataclass
class RobustForceResult:
    """Result of a two-dimensional robust RBE force-range analysis.

    Attributes
    ----------
    method : str
        ``"sample"`` for radial sampling or ``"support"`` for the dual
        support-function method.
    load_dofs : tuple
        The two normalized ``(node, component)`` load degrees of freedom.
    directions : list[list[float]]
        Unit directions used by the analysis.
    statuses : list[str]
        Solver status for each direction. Values are ``"optimal"`` or
        ``"unbounded"``.
    origin_feasible : bool
        Whether the baseline load, corresponding to zero load increment, is
        feasible.
    is_bounded : bool
        Whether the complete two-dimensional feasible load set is bounded.
    feasible_center : list[float]
        Feasible load point used as the center of radial sampling. This is the
        origin when the origin is feasible.
    support_formulation : str, optional
        ``"primal"`` or ``"dual"`` for support analyses. ``None`` for radial
        sampling.
    radial_limits : list[float | None]
        Maximum distances from ``feasible_center``. Populated by radial
        sampling.
    boundary_points : list[list[float]]
        Finite radial boundary points. Populated by radial sampling.
    support_values : list[float | None]
        Support-function values. Populated by primal and dual support methods.
    support_points : list[list[float]]
        Feasible load points returned by the primal support method.
    halfspaces : list[list[float]]
        Finite supporting halfspaces stored as ``[d0, d1, h]``, representing
        ``d0 * b0 + d1 * b1 <= h``.
    inner_polygon : list[list[float]]
        Counterclockwise inner approximation from radial or primal support
        points.
    outer_polygon : list[list[float]]
        Counterclockwise outer approximation from supporting halfspaces.
    polygon : list[list[float]]
        Backward-compatible polygon alias. This is ``inner_polygon`` for radial
        sampling and ``outer_polygon`` for both support formulations.
    """

    method: str
    load_dofs: Tuple[Tuple[Any, str], Tuple[Any, str]]
    directions: List[List[float]]
    statuses: List[str]
    origin_feasible: bool
    is_bounded: bool
    feasible_center: List[float] = field(default_factory=list)
    support_formulation: Optional[str] = None
    radial_limits: List[Optional[float]] = field(default_factory=list)
    boundary_points: List[List[float]] = field(default_factory=list)
    support_values: List[Optional[float]] = field(default_factory=list)
    support_points: List[List[float]] = field(default_factory=list)
    halfspaces: List[List[float]] = field(default_factory=list)
    inner_polygon: List[List[float]] = field(default_factory=list)
    outer_polygon: List[List[float]] = field(default_factory=list)
    polygon: List[List[float]] = field(default_factory=list)


@dataclass
class _RobustProblem:
    """Sparse matrices defining equilibrium, constraints, and visible load projection."""

    equilibrium: csr_matrix
    inequalities: csr_matrix
    inequality_rhs: np.ndarray
    baseline_load: np.ndarray
    load_projection: csr_matrix
    load_dofs: Tuple[Tuple[Any, str], Tuple[Any, str]]


@dataclass
class _LoadSetAnalysis:
    """Feasibility, boundedness, and center shared by all robust methods."""

    origin_feasible: bool
    is_bounded: bool
    feasible_center: np.ndarray


def rbe_robust_sample(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate a two-dimensional safe load range by radial sampling.

    The two selected load components are increments about the baseline load
    defined by gravity and ``external_forces``. Rays start at a feasible load
    point, which is not necessarily the origin. Contact forces use the
    compression-only RBE formulation ``[fn, fu, fv]``.

    Parameters
    ----------
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        Rigid-block assembly.
    load_dofs : sequence[tuple[node, str]]
        Exactly two unique load degrees of freedom. Components are ``"fx"``,
        ``"fy"``, ``"fz"``, ``"mx"``, ``"my"``, or ``"mz"``.
    mu : float, optional
        Friction coefficient.
    density : float, optional
        Default block density.
    external_forces : dict, optional
        Known baseline wrenches keyed by assembly node.
    load_application_points : dict, optional
        Candidate global load-application points keyed by the loaded node. When
        omitted, the two visible load components are applied directly at the
        block center.
    application_force_bound : float, optional
        Symmetric bound for each hidden load-application force component. Only
        used when ``load_application_points`` is provided.
    num_directions : int, optional
        Number of uniformly spaced radial directions.
    tolerance : float, optional
        Geometric tolerance used for polygon construction.
    solver_options : dict, optional
        Options passed to SciPy HiGHS.
    verbose : bool, optional
        Print a result summary.
    timer : bool, optional
        Print total analysis time.

    Returns
    -------
    :class:`RobustForceResult`
        Radial limits, finite boundary points, and an inner polygon.

    Raises
    ------
    ValueError
        If the input is invalid or the complete safe load set is empty.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
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
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set(problem, options)

    statuses = []
    radial_limits = []
    boundary_points = []
    for direction in directions:
        status, limit = _solve_radial(problem, direction, analysis.feasible_center, options)
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
    _report(result, start_time, verbose, timer)
    return result


def rbe_robust_support_primal(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    r"""Approximate a two-dimensional safe load range with primal supports.

    For each unit direction ``d``, this function solves

    .. math::

        \max_{f,u}\ d^T u
        \quad\mathrm{s.t.}\quad
        A f + T u = -p_0,\ H f \leq 0.

    Each solve returns a feasible support point for an inner convex hull and a
    support value for an outer halfspace intersection.

    Parameters
    ----------
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        Rigid-block assembly.
    load_dofs : sequence[tuple[node, str]]
        Exactly two unique load degrees of freedom. Components are ``"fx"``,
        ``"fy"``, ``"fz"``, ``"mx"``, ``"my"``, or ``"mz"``.
    mu : float, optional
        Friction coefficient.
    density : float, optional
        Default block density.
    external_forces : dict, optional
        Known baseline wrenches keyed by assembly node.
    load_application_points : dict, optional
        Candidate global load-application points keyed by the loaded node. When
        omitted, the two visible load components are applied directly at the
        block center.
    application_force_bound : float, optional
        Symmetric bound for each hidden load-application force component. Only
        used when ``load_application_points`` is provided.
    num_directions : int, optional
        Number of uniformly spaced support directions.
    tolerance : float, optional
        Geometric tolerance used for polygon construction.
    solver_options : dict, optional
        Options passed to SciPy HiGHS.
    verbose : bool, optional
        Print a result summary.
    timer : bool, optional
        Print total analysis time.

    Returns
    -------
    :class:`RobustForceResult`
        Support points and values, supporting halfspaces, and inner and outer
        polygons.

    Raises
    ------
    ValueError
        If the input is invalid or the complete safe load set is empty.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
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
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set(problem, options)

    statuses = []
    support_values = []
    support_points = []
    halfspaces = []
    for direction in directions:
        status, support, point = _solve_primal_support(problem, direction, options)
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
    _report(result, start_time, verbose, timer)
    return result


def rbe_robust_support_dual(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Approximate a two-dimensional safe load range with dual supports.

    Each solve is the linear-programming dual of the primal support problem
    and produces a supporting halfspace ``d^T u <= h``. The intersection of
    finite halfspaces is an outer approximation of the safe load set.

    Parameters
    ----------
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        Rigid-block assembly.
    load_dofs : sequence[tuple[node, str]]
        Exactly two unique load degrees of freedom. Components are ``"fx"``,
        ``"fy"``, ``"fz"``, ``"mx"``, ``"my"``, or ``"mz"``.
    mu : float, optional
        Friction coefficient.
    density : float, optional
        Default block density.
    external_forces : dict, optional
        Known baseline wrenches keyed by assembly node.
    load_application_points : dict, optional
        Candidate global load-application points keyed by the loaded node. When
        omitted, the two visible load components are applied directly at the
        block center.
    application_force_bound : float, optional
        Symmetric bound for each hidden load-application force component. Only
        used when ``load_application_points`` is provided.
    num_directions : int, optional
        Number of uniformly spaced support directions.
    tolerance : float, optional
        Geometric tolerance used for polygon construction.
    solver_options : dict, optional
        Options passed to SciPy HiGHS.
    verbose : bool, optional
        Print a result summary.
    timer : bool, optional
        Print total analysis time.

    Returns
    -------
    :class:`RobustForceResult`
        Support values, supporting halfspaces, and an outer polygon.

    Raises
    ------
    ValueError
        If the input is invalid or the complete safe load set is empty.
    RuntimeError
        If HiGHS reports an unexpected numerical or solver failure.
    """
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
    directions = _directions(num_directions)
    options = dict(solver_options or {})
    _validate_tolerance(tolerance)
    analysis = _analyze_load_set(problem, options)

    statuses = []
    support_values = []
    halfspaces = []
    for direction in directions:
        status, support = _solve_dual_support(problem, direction, options)
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
    _report(result, start_time, verbose, timer)
    return result


def rbe_robust_support(
    assembly: Assembly,
    load_dofs: Sequence[Tuple[Any, str]],
    mu: float = 0.84,
    density: float = 1.0,
    external_forces: Optional[dict] = None,
    load_application_points: Optional[dict] = None,
    application_force_bound: Optional[float] = None,
    num_directions: int = 36,
    tolerance: float = 1e-8,
    solver_options: Optional[dict] = None,
    verbose: bool = False,
    timer: bool = False,
) -> RobustForceResult:
    """Backward-compatible alias for :func:`rbe_robust_support_dual`."""
    return rbe_robust_support_dual(
        assembly=assembly,
        load_dofs=load_dofs,
        mu=mu,
        density=density,
        external_forces=external_forces,
        load_application_points=load_application_points,
        application_force_bound=application_force_bound,
        num_directions=num_directions,
        tolerance=tolerance,
        solver_options=solver_options,
        verbose=verbose,
        timer=timer,
    )


def plot_rbe_robust_results(
    results: Union[RobustForceResult, Sequence[RobustForceResult]],
    labels: Optional[Sequence[str]] = None,
    ax=None,
    colors: Optional[Sequence[Any]] = None,
    show_points: bool = True,
    show_center: bool = True,
    show_origin: bool = True,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
):
    """Plot one or more two-dimensional robust RBE result regions.

    Matplotlib is imported only when this function is called. Inner polygons
    are filled, outer polygons are drawn with dashed boundaries, and finite
    sample or support points remain visible for unbounded or degenerate sets.

    Parameters
    ----------
    results : :class:`RobustForceResult` or sequence
        One result or multiple results with identical ``load_dofs``.
    labels : sequence[str], optional
        Legend labels. Defaults describe each result method.
    ax : :class:`matplotlib.axes.Axes`, optional
        Existing axes. A new figure and axes are created when omitted.
    colors : sequence, optional
        Matplotlib colors for the plotted results. Defaults to the CRA color
        cycle and repeats only when there are more results than colors.
    show_points : bool, optional
        Draw radial boundary points or primal support points.
    show_center : bool, optional
        Draw each feasible sampling center.
    show_origin : bool, optional
        Draw the load-increment origin.
    xlim : sequence[float], optional
        Two finite increasing x-axis limits. Together with ``ylim``, these
        limits clip unbounded result regions for visualization.
    ylim : sequence[float], optional
        Two finite increasing y-axis limits. Together with ``xlim``, these
        limits clip unbounded result regions for visualization.

    Returns
    -------
    tuple
        Matplotlib ``(figure, axes)``.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    ValueError
        If no results are provided, labels do not match, colors are empty, or
        load DOFs differ.
    TypeError
        If an item is not a :class:`RobustForceResult`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_rbe_robust_results requires the optional Matplotlib package.") from error

    if isinstance(results, RobustForceResult):
        result_list = [results]
    else:
        result_list = list(results)
    if not result_list:
        raise ValueError("results must contain at least one RobustForceResult.")
    if not all(isinstance(result, RobustForceResult) for result in result_list):
        raise TypeError("results must contain only RobustForceResult instances.")

    load_dofs = result_list[0].load_dofs
    if any(result.load_dofs != load_dofs for result in result_list[1:]):
        raise ValueError("All robust results must use the same load_dofs.")

    plot_limits = _validate_plot_limits(xlim, ylim)

    if labels is None:
        labels = [_result_label(result) for result in result_list]
    else:
        labels = list(labels)
        if len(labels) != len(result_list):
            raise ValueError("labels must have the same length as results.")

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = axes.figure

    if colors is None:
        colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    else:
        colors = list(colors)
        if not colors:
            raise ValueError("colors must contain at least one color.")

    for index, (result, label) in enumerate(zip(result_list, labels)):
        color = colors[index % len(colors)]
        if result.inner_polygon:
            _plot_polygon(
                axes,
                result.inner_polygon,
                color=color,
                linestyle="-",
                alpha=0.18,
                label="{} inner".format(label),
            )
        if result.outer_polygon:
            _plot_polygon(
                axes,
                result.outer_polygon,
                color=color,
                linestyle="--",
                alpha=0.0,
                label="{} outer".format(label),
            )
        if not result.is_bounded and plot_limits:
            x_limits, y_limits = plot_limits
            if result.method == "sample":
                clipped_polygon = _clipped_radial_polygon(result, x_limits, y_limits)
                clipped_label = "{} clipped unbounded inner".format(label)
                clipped_linestyle = "-"
                clipped_alpha = 0.18
            else:
                clipped_polygon = _clipped_support_polygon(result, x_limits, y_limits)
                clipped_label = "{} clipped unbounded outer".format(label)
                clipped_linestyle = "--"
                clipped_alpha = 0.08
            if clipped_polygon:
                _plot_polygon(
                    axes,
                    clipped_polygon,
                    color=color,
                    linestyle=clipped_linestyle,
                    alpha=clipped_alpha,
                    label=clipped_label,
                )
        if show_points:
            points = result.support_points if result.support_points else result.boundary_points
            if points:
                coordinates = np.asarray(points)
                axes.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color=color,
                    marker=".",
                    label="{} points".format(label),
                )
        if show_center and result.feasible_center:
            axes.scatter(
                [result.feasible_center[0]],
                [result.feasible_center[1]],
                color=color,
                marker="x",
                s=55,
                label="{} center".format(label),
            )

    if show_origin:
        axes.scatter([0.0], [0.0], color="black", marker="+", s=65, label="origin")

    axes.set_xlabel(_load_dof_label(load_dofs[0]))
    axes.set_ylabel(_load_dof_label(load_dofs[1]))
    if plot_limits:
        axes.set_xlim(plot_limits[0])
        axes.set_ylim(plot_limits[1])
        axes.set_aspect("equal", adjustable="box")
    else:
        axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, alpha=0.3)
    axes.legend()
    return figure, axes


def _prepare_problem(
    assembly,
    load_dofs,
    mu,
    density,
    external_forces,
    load_application_points=None,
    application_force_bound=None,
):
    """Construct ``A``, ``H``, ``h``, ``p0``, and ``P`` for robust analysis.

    ``A`` has one six-component equilibrium row block per free body and three
    contact-force columns per interface vertex. ``H`` contains eight
    linearized friction inequalities, one compression inequality per interface
    vertex, and optional hidden load-application force bounds. ``P`` projects
    decision variables to the two visible load components.
    """
    normalized_dofs = _validate_load_dofs(assembly, load_dofs)
    base_equilibrium = equilibrium_setup(assembly, penalty=False).tocsr()
    friction = friction_setup(assembly, mu, penalty=False).tocsr()
    baseline_load = external_force_setup(assembly, density, external_forces).flatten()

    vertex_count = num_vertices(assembly)
    force_count = vertex_count * 3
    normal_nonnegative = csr_matrix(
        (
            -np.ones(vertex_count),
            (np.arange(vertex_count), np.arange(vertex_count) * 3),
        ),
        shape=(vertex_count, force_count),
    )
    base_inequalities = vstack([friction, normal_nonnegative], format="csr")

    if load_application_points is None:
        if application_force_bound is not None:
            raise ValueError("application_force_bound requires load_application_points.")
        return _center_load_problem(
            base_equilibrium,
            base_inequalities,
            baseline_load,
            assembly,
            normalized_dofs,
        )

    return _application_load_problem(
        base_equilibrium,
        base_inequalities,
        baseline_load,
        assembly,
        normalized_dofs,
        load_application_points,
        application_force_bound,
    )


def _center_load_problem(base_equilibrium, base_inequalities, baseline_load, assembly, load_dofs):
    force_count = base_equilibrium.shape[1]
    load_basis = _load_basis(assembly, load_dofs, base_equilibrium.shape[0])
    equilibrium = hstack([base_equilibrium, load_basis], format="csr")
    inequalities = hstack(
        [base_inequalities, csr_matrix((base_inequalities.shape[0], 2))],
        format="csr",
    )
    load_projection = csr_matrix(
        (
            np.ones(2),
            ([0, 1], [force_count, force_count + 1]),
        ),
        shape=(2, force_count + 2),
    )

    return _RobustProblem(
        equilibrium=equilibrium,
        inequalities=inequalities,
        inequality_rhs=np.zeros(inequalities.shape[0]),
        baseline_load=baseline_load,
        load_projection=load_projection,
        load_dofs=load_dofs,
    )


def _application_load_problem(
    base_equilibrium,
    base_inequalities,
    baseline_load,
    assembly,
    load_dofs,
    load_application_points,
    application_force_bound,
):
    load_node, points = _validate_application_points(assembly, load_dofs, load_application_points)
    bound = _validate_application_force_bound(application_force_bound)
    application_basis, load_projection = _application_load_basis(
        assembly,
        load_dofs,
        load_node,
        points,
        base_equilibrium.shape[0],
        base_equilibrium.shape[1],
    )
    equilibrium = hstack([base_equilibrium, application_basis], format="csr")
    inequalities = hstack(
        [base_inequalities, csr_matrix((base_inequalities.shape[0], application_basis.shape[1]))],
        format="csr",
    )
    inequality_rhs = np.zeros(inequalities.shape[0])

    if bound is not None:
        bound_rows = _application_bound_rows(base_equilibrium.shape[1], application_basis.shape[1], bound)
        inequalities = vstack([inequalities, bound_rows[0]], format="csr")
        inequality_rhs = np.concatenate([inequality_rhs, bound_rows[1]])

    return _RobustProblem(
        equilibrium=equilibrium,
        inequalities=inequalities,
        inequality_rhs=inequality_rhs,
        baseline_load=baseline_load,
        load_projection=load_projection,
        load_dofs=load_dofs,
    )


def _validate_load_dofs(assembly, load_dofs):
    if len(load_dofs) != 2:
        raise ValueError("load_dofs must contain exactly two (node, component) pairs.")

    node_keys = list(assembly.graph.nodes())
    normalized = []
    for load_dof in load_dofs:
        if not isinstance(load_dof, (list, tuple)) or len(load_dof) != 2:
            raise ValueError("Each load degree of freedom must be a (node, component) pair.")
        node, component = load_dof
        if node not in node_keys:
            raise ValueError("Unknown assembly node: {!r}.".format(node))
        if assembly.graph.node_attribute(node, "is_support"):
            raise ValueError("Load degree of freedom node {!r} is a support.".format(node))
        if not isinstance(component, str) or component.lower() not in _COMPONENT_INDEX:
            raise ValueError("Unknown load component: {!r}.".format(component))
        normalized.append((node, component.lower()))

    if normalized[0] == normalized[1]:
        raise ValueError("load_dofs must contain two unique degrees of freedom.")
    return normalized[0], normalized[1]


def _validate_application_points(assembly, load_dofs, load_application_points):
    load_node = load_dofs[0][0]
    if load_dofs[1][0] != load_node:
        raise ValueError("load_application_points require both load_dofs on the same node.")
    if any(component not in _FORCE_COMPONENTS for _, component in load_dofs):
        raise ValueError("load_application_points currently support only force components.")

    if isinstance(load_application_points, dict):
        if load_node not in load_application_points:
            raise ValueError("load_application_points must include the loaded node {!r}.".format(load_node))
        if any(node != load_node for node in load_application_points):
            raise ValueError("load_application_points must be supplied only for the loaded node.")
        raw_points = load_application_points[load_node]
    else:
        raw_points = load_application_points

    try:
        raw_points = list(raw_points)
    except TypeError as error:
        raise ValueError("load_application_points must contain point coordinates.") from error
    if not raw_points:
        raise ValueError("load_application_points must contain at least one point.")

    points = []
    for index, point in enumerate(raw_points):
        points.append(_xyz_array(point, "load_application_points[{}]".format(index)))
    return load_node, points


def _validate_application_force_bound(application_force_bound):
    if application_force_bound is None:
        return None
    if isinstance(application_force_bound, bool):
        raise ValueError("application_force_bound must be a positive finite number.")
    bound = float(application_force_bound)
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("application_force_bound must be a positive finite number.")
    return bound


def _application_load_basis(assembly, load_dofs, load_node, points, row_count, base_force_count):
    node_keys = list(assembly.graph.nodes())
    node_index = {node: index for index, node in enumerate(node_keys)}
    free = free_nodes(assembly)
    free_position = free.index(node_index[load_node])
    base_row = free_position * 6

    block = assembly.graph.node_attribute(load_node, "block")
    center = _xyz_array(block.center(), "loaded block center")

    rows = []
    columns = []
    data = []
    projection_rows = []
    projection_columns = []
    projection_data = []

    column = 0
    for point in points:
        offset = point - center
        for load_index, (_, component) in enumerate(load_dofs):
            unit = np.zeros(3)
            unit[_COMPONENT_INDEX[component]] = 1.0
            moment = np.cross(offset, unit)

            rows.append(base_row + _COMPONENT_INDEX[component])
            columns.append(column)
            data.append(1.0)
            for moment_index, value in enumerate(moment):
                if value:
                    rows.append(base_row + 3 + moment_index)
                    columns.append(column)
                    data.append(float(value))

            projection_rows.append(load_index)
            projection_columns.append(base_force_count + column)
            projection_data.append(1.0)
            column += 1

    application_basis = csr_matrix((data, (rows, columns)), shape=(row_count, column))
    load_projection = csr_matrix(
        (projection_data, (projection_rows, projection_columns)),
        shape=(2, base_force_count + column),
    )
    return application_basis, load_projection


def _application_bound_rows(base_force_count, application_force_count, bound):
    total_count = base_force_count + application_force_count
    rows = np.concatenate(
        [np.arange(application_force_count), np.arange(application_force_count, 2 * application_force_count)]
    )
    columns = np.concatenate(
        [
            base_force_count + np.arange(application_force_count),
            base_force_count + np.arange(application_force_count),
        ]
    )
    data = np.concatenate([np.ones(application_force_count), -np.ones(application_force_count)])
    inequalities = csr_matrix((data, (rows, columns)), shape=(2 * application_force_count, total_count))
    rhs = np.full(2 * application_force_count, bound)
    return inequalities, rhs


def _xyz_array(point, name):
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        coordinates = np.asarray([point.x, point.y, point.z], dtype=float)
    else:
        try:
            coordinates = np.asarray(point, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("{} must be a finite three-dimensional point.".format(name)) from error
    if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
        raise ValueError("{} must be a finite three-dimensional point.".format(name))
    return coordinates


def _load_basis(assembly, load_dofs, row_count):
    node_keys = list(assembly.graph.nodes())
    node_index = {node: index for index, node in enumerate(node_keys)}
    free = free_nodes(assembly)
    rows = []
    columns = []
    for column, (node, component) in enumerate(load_dofs):
        free_position = free.index(node_index[node])
        rows.append(free_position * 6 + _COMPONENT_INDEX[component])
        columns.append(column)
    return csr_matrix((np.ones(2), (rows, columns)), shape=(row_count, 2))


def _directions(num_directions):
    if isinstance(num_directions, bool) or not isinstance(num_directions, int) or num_directions < 4:
        raise ValueError("num_directions must be an integer greater than or equal to 4.")
    angles = np.arange(num_directions, dtype=float) * (2.0 * math.pi / num_directions)
    return [np.array([math.cos(angle), math.sin(angle)]) for angle in angles]


def _validate_tolerance(tolerance):
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")


def _analyze_load_set(problem, options):
    """Find a feasible load center and determine global boundedness."""
    origin_feasible = _is_origin_feasible(problem, options)
    feasibility_witness = _solve_load_set_feasibility(problem, options)
    cardinal_directions = (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    )
    cardinal_results = [_solve_primal_support(problem, direction, options) for direction in cardinal_directions]
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


def _is_origin_feasible(problem, options):
    """Return whether the zero unknown-load increment belongs to the load set."""
    variable_count = problem.equilibrium.shape[1]
    equality = vstack([problem.equilibrium, problem.load_projection], format="csr")
    result = linprog(
        np.zeros(variable_count),
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=equality,
        b_eq=np.concatenate([-problem.baseline_load, np.zeros(2)]),
        bounds=[(None, None)] * variable_count,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return True
    if result.status == 2:
        return False
    raise RuntimeError("Origin feasibility solve failed: {}.".format(result.message))


def _solve_load_set_feasibility(problem, options):
    """Return one feasible ``u`` or raise when the complete safe set is empty."""
    variable_count = problem.equilibrium.shape[1]
    result = linprog(
        np.zeros(variable_count),
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=problem.equilibrium,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * variable_count,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return np.asarray(problem.load_projection.dot(result.x), dtype=float)
    if result.status == 2:
        raise ValueError("The safe load set is empty for compression-only RBE.")
    raise RuntimeError("Safe load-set feasibility solve failed: {}.".format(result.message))


def _solve_radial(problem, direction, center, options):
    """Maximize distance ``alpha`` from a verified feasible load center."""
    variable_count = problem.equilibrium.shape[1]
    zero_equilibrium_alpha = csr_matrix((problem.equilibrium.shape[0], 1))
    load_column = csr_matrix(-np.asarray(direction, dtype=float).reshape((2, 1)))
    equilibrium_rows = hstack([problem.equilibrium, zero_equilibrium_alpha], format="csr")
    load_rows = hstack([problem.load_projection, load_column], format="csr")
    equality = vstack([equilibrium_rows, load_rows], format="csr")
    inequalities = hstack(
        [problem.inequalities, csr_matrix((problem.inequalities.shape[0], 1))],
        format="csr",
    )
    objective = np.zeros(variable_count + 1)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=equality,
        b_eq=np.concatenate([-problem.baseline_load, np.asarray(center, dtype=float)]),
        bounds=[(None, None)] * variable_count + [(0, None)],
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", max(0.0, float(result.x[-1]))
    if result.status == 3:
        return "unbounded", None
    raise RuntimeError("Radial load solve failed: {}.".format(result.message))


def _solve_primal_support(problem, direction, options):
    """Solve ``max d^T u`` and return its value and feasible support point."""
    variable_count = problem.equilibrium.shape[1]
    objective = -np.asarray(problem.load_projection.transpose().dot(np.asarray(direction, dtype=float))).ravel()
    result = linprog(
        objective,
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=problem.equilibrium,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * variable_count,
        method="highs",
        options=options,
    )
    if result.status == 0:
        point = np.asarray(problem.load_projection.dot(result.x), dtype=float)
        support = float(np.asarray(direction, dtype=float).dot(point))
        return "optimal", support, point.tolist()
    if result.status == 3:
        return "unbounded", None, None
    raise RuntimeError("Primal support solve failed: {}.".format(result.message))


def _solve_dual_support(problem, direction, options):
    """Solve the dual support LP for one direction ``d``."""
    equilibrium_rows = problem.equilibrium.shape[0]
    inequality_rows = problem.inequalities.shape[0]
    dual_equalities = hstack([problem.equilibrium.transpose(), problem.inequalities.transpose()], format="csr")
    right_hand_side = np.asarray(problem.load_projection.transpose().dot(np.asarray(direction, dtype=float))).ravel()
    objective = np.concatenate([-problem.baseline_load, problem.inequality_rhs])
    result = linprog(
        objective,
        A_eq=dual_equalities,
        b_eq=right_hand_side,
        bounds=[(None, None)] * equilibrium_rows + [(0, None)] * inequality_rows,
        method="highs",
        options=options,
    )
    if result.status == 0:
        return "optimal", float(result.fun)
    if result.status == 2:
        return "unbounded", None
    raise RuntimeError("Dual support solve failed: {}.".format(result.message))


def _inner_polygon(points, tolerance):
    unique_points = _deduplicate_points(points, tolerance)
    if len(unique_points) < 3:
        return []
    try:
        hull = ConvexHull(np.asarray(unique_points))
    except QhullError:
        return []
    polygon = [unique_points[index] for index in hull.vertices]
    return _counterclockwise_polygon(polygon, tolerance)


def _outer_polygon(halfspaces, tolerance):
    if len(halfspaces) < 3:
        return []
    coefficients = np.asarray([halfspace[:2] for halfspace in halfspaces])
    limits = np.asarray([halfspace[2] for halfspace in halfspaces])
    intersections = []
    for first in range(len(halfspaces)):
        for second in range(first + 1, len(halfspaces)):
            matrix = coefficients[[first, second], :]
            determinant = np.linalg.det(matrix)
            if abs(determinant) <= tolerance:
                continue
            point = np.linalg.solve(matrix, limits[[first, second]])
            if np.all(coefficients.dot(point) <= limits + tolerance):
                intersections.append(point.tolist())

    unique_points = _deduplicate_points(intersections, tolerance)
    if len(unique_points) < 3:
        return []
    center = np.mean(np.asarray(unique_points), axis=0)
    polygon = sorted(unique_points, key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))
    return _counterclockwise_polygon(polygon, tolerance)


def _validate_plot_limits(xlim, ylim):
    if xlim is None and ylim is None:
        return None
    if xlim is None or ylim is None:
        raise ValueError("xlim and ylim must be provided together.")

    normalized = []
    for name, limits in (("xlim", xlim), ("ylim", ylim)):
        if len(limits) != 2:
            raise ValueError("{} must contain exactly two values.".format(name))
        lower, upper = (float(value) for value in limits)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("{} must contain two finite increasing values.".format(name))
        normalized.append((lower, upper))
    return normalized[0], normalized[1]


def _clipped_radial_polygon(result, xlim, ylim, tolerance=1e-9):
    center = np.asarray(result.feasible_center, dtype=float)
    points = []
    for direction, status, limit in zip(result.directions, result.statuses, result.radial_limits):
        maximum = math.inf if status == "unbounded" else limit
        if maximum is None:
            continue
        points.extend(
            _clip_ray_to_box(
                center,
                np.asarray(direction, dtype=float),
                maximum,
                xlim,
                ylim,
                tolerance,
            )
        )
    return _inner_polygon(points, tolerance)


def _clip_ray_to_box(origin, direction, maximum, xlim, ylim, tolerance):
    lower = 0.0
    upper = maximum
    for coordinate, delta, limits in zip(origin, direction, (xlim, ylim)):
        if abs(delta) <= tolerance:
            if coordinate < limits[0] - tolerance or coordinate > limits[1] + tolerance:
                return []
            continue
        intersections = sorted(((limits[0] - coordinate) / delta, (limits[1] - coordinate) / delta))
        lower = max(lower, intersections[0])
        upper = min(upper, intersections[1])
        if upper < lower - tolerance:
            return []

    start = origin + max(0.0, lower) * direction
    end = origin + upper * direction
    if np.linalg.norm(end - start) <= tolerance:
        return [start.tolist()]
    return [start.tolist(), end.tolist()]


def _clipped_support_polygon(result, xlim, ylim, tolerance=1e-9):
    halfspaces = list(result.halfspaces)
    halfspaces.extend(
        [
            [1.0, 0.0, xlim[1]],
            [-1.0, 0.0, -xlim[0]],
            [0.0, 1.0, ylim[1]],
            [0.0, -1.0, -ylim[0]],
        ]
    )
    return _outer_polygon(halfspaces, tolerance)


def _deduplicate_points(points, tolerance):
    unique_points = []
    for point in points:
        candidate = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(candidate - np.asarray(existing)) <= tolerance for existing in unique_points):
            unique_points.append(candidate.tolist())
    return unique_points


def _counterclockwise_polygon(polygon, tolerance):
    area_twice = sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )
    if abs(area_twice) <= tolerance:
        return []
    if area_twice < 0:
        polygon.reverse()
    return polygon


def _plot_polygon(axes, polygon, color, linestyle, alpha, label):
    """Draw a closed polygon boundary and optional translucent fill."""
    coordinates = np.asarray(polygon + [polygon[0]])
    axes.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=color,
        linestyle=linestyle,
        label=label,
    )
    if alpha:
        axes.fill(coordinates[:, 0], coordinates[:, 1], color=color, alpha=alpha)


def _result_label(result):
    if result.method == "sample":
        return "radial sample"
    return "{} support".format(result.support_formulation or "dual")


def _load_dof_label(load_dof):
    node, component = load_dof
    return "node {!r} {}".format(node, component)


def _report(result, start_time, verbose, timer):
    if verbose:
        optimal_count = result.statuses.count("optimal")
        print(
            "robust RBE {}: {} of {} directions bounded; load set bounded={}".format(
                result.method,
                optimal_count,
                len(result.statuses),
                result.is_bounded,
            )
        )
    if timer:
        print("--- robust RBE time: {} seconds ---".format(time.time() - start_time))
