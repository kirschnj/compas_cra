"""Diagnose failure modes just outside example-18 robust safe-load boundaries."""

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from compas_cra.equilibrium.cra_helper import equilibrium_setup
from compas_cra.equilibrium.cra_helper import external_force_setup
from compas_cra.equilibrium.cra_helper import friction_setup
from compas_cra.equilibrium.cra_helper import num_vertices
from compas_cra.equilibrium.rbe_robust import RobustForceResult
from compas_cra.equilibrium.rbe_robust import _analyze_load_set
from compas_cra.equilibrium.rbe_robust import _application_bound_rows
from compas_cra.equilibrium.rbe_robust import _application_load_basis
from compas_cra.equilibrium.rbe_robust import _directions
from compas_cra.equilibrium.rbe_robust import _outer_polygon
from compas_cra.equilibrium.rbe_robust import _solve_dual_support

LOAD_NODE = 2
LOAD_DOFS = [(LOAD_NODE, "fx"), (LOAD_NODE, "fz")]
MU = 0.8
DENSITY = 1.0
NUM_DIRECTIONS = 72
APPLICATION_FORCE_BOUND = 1e3
BOUNDARY_OFFSET = 1e-3
TENSION_TOLERANCE = 1e-7
OPEN_CONTACT_TOLERANCE = 1e-7
FRICTION_LIMIT_TOLERANCE = 1e-6
HALFSPACE_TOLERANCE = 1e-7
SUPPORT_TOLERANCE = 1e-8
ENFORCE_2D_TIES = True
PAIR_COORDINATE_TOLERANCE = 1e-8
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
XLIM = (-1.0, 0.1)
YLIM = (-0.5, 1.0)


@dataclass
class MatrixProblem:
    """Sparse LP matrices for a fixed hidden-load model."""

    equilibrium: csr_matrix
    inequalities: csr_matrix
    inequality_rhs: np.ndarray
    baseline_load: np.ndarray
    load_projection: csr_matrix
    load_dofs: tuple
    hidden_force_count: int
    contact_pair_count: int
    contact_tie_count: int
    application_pair_count: int
    application_tie_count: int


@dataclass
class BoundaryDiagnostic:
    """One visible-load boundary and its outside-load diagnostic."""

    label: str
    halfspace: np.ndarray
    segment: np.ndarray
    midpoint: np.ndarray
    outside_load: np.ndarray
    compression_result: object
    penalty_result: object


def load_example18_module():
    """Load the numeric example-18 module by file path."""
    path = Path(__file__).with_name("18_rbe_robust_3block.py")
    spec = importlib.util.spec_from_file_location("example18", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load example-18 module from {}.".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example18_geometry_kwargs():
    """Return the nominal example-18 geometry parameters."""
    return {
        "b0_width": 0.5,
        "b1_height": 0.5,
        "b1_base": 0.5,
        "b2_base": 0.6,
        "b2_top": 0.9,
        "alpha": math.pi / 2,
        "beta": 2 * math.pi / 3,
        "gamma": 2 * math.pi / 3,
        "thickness": 1,
    }


def build_example18_assembly(example18):
    """Build the full three-block example-18 assembly with block 0 fixed."""
    geometry = example18.Arch(**example18_geometry_kwargs())
    return example18.build_assembly(geometry, block_nodes=[0, 1, 2], support_nodes=[0])


def point_coordinates(point):
    """Return point coordinates as a plain list."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    return list(point)


def xyz_array(point):
    """Return a point as a three-component NumPy array."""
    return np.asarray(point_coordinates(point), dtype=float)


def load_setup(example18, assembly):
    """Return the visible load and hidden application-point setup."""
    block = assembly.graph.node_attribute(LOAD_NODE, "block")
    application_points = {
        LOAD_NODE: [xyz_array(point) for point in example18.rightmost_side_vertices(block)],
    }
    return LOAD_NODE, LOAD_DOFS, application_points


def xz_pair_key(point):
    """Return a tolerance-rounded key for front/back points in a 2D extrusion."""
    coordinates = xyz_array(point)
    return (
        int(round(coordinates[0] / PAIR_COORDINATE_TOLERANCE)),
        int(round(coordinates[2] / PAIR_COORDINATE_TOLERANCE)),
    )


def front_back_pairs(indexed_points):
    """Return index pairs for points sharing the same ``x`` and ``z`` coordinates."""
    groups = {}
    for index, point in indexed_points:
        groups.setdefault(xz_pair_key(point), []).append((index, xyz_array(point)))

    pairs = []
    for group in groups.values():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda item: item[1][1])
        first_index = ordered[0][0]
        for index, _ in ordered[1:]:
            pairs.append((first_index, index))
    return pairs


def contact_front_back_pairs(assembly):
    """Return contact force-variable pairs that represent the same 2D contact point."""
    pairs = []
    contact_index = 0
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        for interface in interfaces:
            indexed_points = []
            for point in interface.points:
                indexed_points.append((contact_index, point))
                contact_index += 1
            pairs.extend(front_back_pairs(indexed_points))
    return pairs


def application_front_back_pairs(points):
    """Return hidden load-variable point pairs for the rightmost application face."""
    return front_back_pairs(list(enumerate(points)))


def variable_tie_rows(variable_count, pairs, stride, components, offset=0):
    """Create equality rows enforcing equal variable components for index pairs."""
    rows = []
    columns = []
    data = []
    row = 0
    for first, second in pairs:
        for component in components:
            rows.extend([row, row])
            columns.extend([offset + first * stride + component, offset + second * stride + component])
            data.extend([1.0, -1.0])
            row += 1
    return csr_matrix((data, (rows, columns)), shape=(row, variable_count))


def two_dimensional_tie_rows(
    assembly, load_node, load_dofs, application_points, base_force_count, total_count, penalty
):
    """Return 2D front/back equality rows for contact and hidden load variables."""
    if not ENFORCE_2D_TIES:
        return csr_matrix((0, total_count)), 0, 0, 0, 0

    contact_pairs = contact_front_back_pairs(assembly)
    contact_stride = 4 if penalty else 3
    contact_components = range(contact_stride)
    contact_rows = variable_tie_rows(total_count, contact_pairs, contact_stride, contact_components)

    application_pairs = application_front_back_pairs(application_points[load_node])
    application_stride = len(load_dofs)
    application_components = range(application_stride)
    application_rows = variable_tie_rows(
        total_count,
        application_pairs,
        application_stride,
        application_components,
        offset=base_force_count,
    )

    rows = vstack([contact_rows, application_rows], format="csr")
    return (
        rows,
        len(contact_pairs),
        contact_rows.shape[0],
        len(application_pairs),
        application_rows.shape[0],
    )


def hidden_load_problem(assembly, load_node, load_dofs, application_points, penalty):
    """Build compression-only or penalty LP matrices with hidden load variables."""
    base_equilibrium = equilibrium_setup(assembly, penalty=penalty).tocsr()
    friction = friction_setup(assembly, MU, penalty=penalty).tocsr()
    baseline_load = np.asarray(external_force_setup(assembly, DENSITY, None), dtype=float).ravel()

    if penalty:
        base_inequalities = friction
    else:
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

    application_basis, load_projection = _application_load_basis(
        assembly,
        load_dofs,
        load_node,
        application_points[load_node],
        base_equilibrium.shape[0],
        base_equilibrium.shape[1],
    )
    hidden_force_count = application_basis.shape[1]
    equilibrium = hstack([base_equilibrium, application_basis], format="csr")
    total_count = equilibrium.shape[1]
    tie_rows, contact_pair_count, contact_tie_count, application_pair_count, application_tie_count = (
        two_dimensional_tie_rows(
            assembly,
            load_node,
            load_dofs,
            application_points,
            base_equilibrium.shape[1],
            total_count,
            penalty,
        )
    )
    if tie_rows.shape[0]:
        equilibrium = vstack([equilibrium, tie_rows], format="csr")
        baseline_load = np.concatenate([baseline_load, np.zeros(tie_rows.shape[0])])

    inequalities = hstack(
        [base_inequalities, csr_matrix((base_inequalities.shape[0], hidden_force_count))],
        format="csr",
    )
    inequality_rhs = np.zeros(inequalities.shape[0])
    bound_rows, bound_rhs = _application_bound_rows(
        base_equilibrium.shape[1],
        hidden_force_count,
        APPLICATION_FORCE_BOUND,
    )
    inequalities = vstack([inequalities, bound_rows], format="csr")
    inequality_rhs = np.concatenate([inequality_rhs, bound_rhs])

    return MatrixProblem(
        equilibrium=equilibrium,
        inequalities=inequalities,
        inequality_rhs=inequality_rhs,
        baseline_load=baseline_load,
        load_projection=load_projection,
        load_dofs=tuple(load_dofs),
        hidden_force_count=hidden_force_count,
        contact_pair_count=contact_pair_count,
        contact_tie_count=contact_tie_count,
        application_pair_count=application_pair_count,
        application_tie_count=application_tie_count,
    )


def solve_support_dual(problem, num_directions=NUM_DIRECTIONS, tolerance=SUPPORT_TOLERANCE):
    """Solve the tied compression-only support problem in all directions."""
    directions = _directions(num_directions)
    analysis = _analyze_load_set(problem, {})

    statuses = []
    support_values = []
    halfspaces = []
    for direction in directions:
        status, support = _solve_dual_support(problem, direction, {})
        statuses.append(status)
        support_values.append(support)
        if support is not None:
            halfspaces.append([float(direction[0]), float(direction[1]), support])

    outer_polygon = _outer_polygon(halfspaces, tolerance) if analysis.is_bounded else []
    return RobustForceResult(
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


def solve_fixed_visible_load(problem, load):
    """Check compression-only feasibility with the visible load fixed."""
    equality = vstack([problem.equilibrium, problem.load_projection], format="csr")
    equality_rhs = np.concatenate([-problem.baseline_load, np.asarray(load, dtype=float)])
    return linprog(
        np.zeros(problem.equilibrium.shape[1]),
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=[(None, None)] * problem.equilibrium.shape[1],
        method="highs",
    )


def solve_penalty_tension(problem, assembly, load):
    """Minimize total tensile normal force for one fixed visible load."""
    equality = vstack([problem.equilibrium, problem.load_projection], format="csr")
    equality_rhs = np.concatenate([-problem.baseline_load, np.asarray(load, dtype=float)])
    contact_count = num_vertices(assembly)
    objective = np.zeros(problem.equilibrium.shape[1])
    objective[1 : 4 * contact_count : 4] = 1.0

    bounds = []
    for _ in range(contact_count):
        bounds.extend([(0.0, None), (0.0, None), (None, None), (None, None)])
    bounds.extend([(None, None)] * problem.hidden_force_count)

    return linprog(
        objective,
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )


def contact_table(assembly):
    """Return contact metadata in the same order as penalty force variables."""
    contacts = []
    contact_index = 0
    for edge in assembly.graph.edges(False):
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        for interface_index, interface in enumerate(interfaces):
            for point_index, point in enumerate(interface.points):
                contacts.append(
                    {
                        "index": contact_index,
                        "edge": edge,
                        "interface_index": interface_index,
                        "point_index": point_index,
                        "point": point,
                    }
                )
                contact_index += 1
    return contacts


def penalty_contact_forces(result, assembly):
    """Return penalty contact variables as ``[fn_plus, fn_minus, fu, fv]`` rows."""
    contact_count = num_vertices(assembly)
    return np.asarray(result.x[: 4 * contact_count], dtype=float).reshape((-1, 4))


def penalty_total_tension(result, assembly):
    """Return the total tensile normal force in one penalty LP result."""
    if result.status != 0:
        return math.nan
    return float(np.sum(penalty_contact_forces(result, assembly)[:, 1]))


def friction_utilization(fn_plus, fu, fv):
    """Return friction utilization or ``None`` for open contacts."""
    if fn_plus <= OPEN_CONTACT_TOLERANCE:
        return None
    tangent = math.sqrt(fu * fu + fv * fv)
    return tangent / (MU * fn_plus)


def feasibility_label(result):
    """Return a compact LP status label."""
    if result.status == 0:
        return "feasible"
    if result.status == 2:
        return "infeasible"
    return "solver status {}".format(result.status)


def format_point(point):
    """Format a 3D point for compact console output."""
    coordinates = xyz_array(point)
    return "[{:.6g}, {:.6g}, {:.6g}]".format(coordinates[0], coordinates[1], coordinates[2])


def boundary_halfspace(midpoint, halfspaces):
    """Return the support halfspace active at one polygon-edge midpoint."""
    halfspaces = np.asarray(halfspaces, dtype=float)
    residuals = np.abs(halfspaces[:, :2].dot(midpoint) - halfspaces[:, 2])
    return halfspaces[int(np.argmin(residuals))]


def line_box_segment(halfspace, xlim, ylim, tolerance=1e-9):
    """Return one support-line segment clipped to a rectangular viewport."""
    a, b, c = halfspace
    points = []

    if abs(b) > tolerance:
        for x in xlim:
            z = (c - a * x) / b
            if ylim[0] - tolerance <= z <= ylim[1] + tolerance:
                points.append(np.asarray([x, z], dtype=float))
    if abs(a) > tolerance:
        for z in ylim:
            x = (c - b * z) / a
            if xlim[0] - tolerance <= x <= xlim[1] + tolerance:
                points.append(np.asarray([x, z], dtype=float))

    unique = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def clipped_boundary_segment(halfspace, halfspaces, xlim, ylim, tolerance=1e-8):
    """Return the visible active segment of one support line inside the safe set."""
    segment = line_box_segment(halfspace, xlim, ylim)
    if segment is None:
        return None

    start, end = segment
    direction = end - start
    lower = 0.0
    upper = 1.0
    for other in np.asarray(halfspaces, dtype=float):
        value = float(other[:2].dot(start) - other[2])
        slope = float(other[:2].dot(direction))
        if abs(slope) <= tolerance:
            if value > tolerance:
                return None
            continue
        limit = -value / slope
        if slope > 0.0:
            upper = min(upper, limit)
        else:
            lower = max(lower, limit)
        if upper < lower - tolerance:
            return None

    if upper - lower <= tolerance:
        return None
    return np.vstack([start + lower * direction, start + upper * direction])


def boundary_diagnostics(result, compression_problem, penalty_problem, assembly, xlim=XLIM, ylim=YLIM):
    """Create diagnostics for support boundary segments visible in the viewport."""
    if len(result.halfspaces) < 3:
        raise ValueError("Example 18-4 requires at least three support halfspaces.")

    diagnostics = []
    seen = set()
    for halfspace in np.asarray(result.halfspaces, dtype=float):
        normal = halfspace[:2]
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= HALFSPACE_TOLERANCE:
            continue
        segment = clipped_boundary_segment(halfspace, result.halfspaces, xlim, ylim)
        if segment is None:
            continue
        key = tuple(np.round(halfspace, 10))
        if key in seen:
            continue
        seen.add(key)

        midpoint = 0.5 * (segment[0] + segment[1])
        outside_load = midpoint + BOUNDARY_OFFSET * normal / normal_length
        compression = solve_fixed_visible_load(compression_problem, outside_load)
        penalty = solve_penalty_tension(penalty_problem, assembly, outside_load)
        diagnostics.append(
            BoundaryDiagnostic(
                label="B{}".format(len(diagnostics) + 1),
                halfspace=halfspace,
                segment=segment,
                midpoint=midpoint,
                outside_load=outside_load,
                compression_result=compression,
                penalty_result=penalty,
            )
        )
    if not diagnostics:
        raise ValueError("No safe-load boundary segments intersect the requested diagnostic viewport.")
    return diagnostics


def report_tensile_contacts(diagnostic, assembly):
    """Print tensile and open contacts for one outside boundary load."""
    halfspace = diagnostic.halfspace
    print("\n{}: {:.9g} Fx + {:.9g} Fz <= {:.9g}".format(*((diagnostic.label,) + tuple(halfspace))))
    print("  midpoint:     Fx={:.9g}, Fz={:.9g}".format(diagnostic.midpoint[0], diagnostic.midpoint[1]))
    print("  outside load: Fx={:.9g}, Fz={:.9g}".format(diagnostic.outside_load[0], diagnostic.outside_load[1]))
    print("  compression-only check: {}".format(feasibility_label(diagnostic.compression_result)))
    if diagnostic.penalty_result.status != 0:
        print(
            "  penalty tension check: solver status {}, {}".format(
                diagnostic.penalty_result.status,
                diagnostic.penalty_result.message,
            )
        )
        return

    contact_forces = penalty_contact_forces(diagnostic.penalty_result, assembly)
    total_tension = penalty_total_tension(diagnostic.penalty_result, assembly)
    print("  minimum total fn_minus={:.9g}".format(total_tension))
    printed = False
    contacts = contact_table(assembly)
    for contact in contacts:
        fn_plus, fn_minus, fu, fv = contact_forces[contact["index"]]
        if fn_minus <= TENSION_TOLERANCE:
            continue
        utilization = friction_utilization(fn_plus, fu, fv)
        utilization_text = "nan" if utilization is None else "{:.9g}".format(utilization)
        print(
            "    edge {edge}, contact {index}, point {point_index}: position={position}, "
            "fn+={fn_plus:.9g}, fn-={fn_minus:.9g}, fn_eff={fn_eff:.9g}, "
            "fu={fu:.9g}, fv={fv:.9g}, |ft|/(mu*fn+)={utilization}".format(
                edge=contact["edge"],
                index=contact["index"],
                point_index=contact["point_index"],
                position=format_point(contact["point"]),
                fn_plus=fn_plus,
                fn_minus=fn_minus,
                fn_eff=fn_plus - fn_minus,
                fu=fu,
                fv=fv,
                utilization=utilization_text,
            )
        )
        printed = True
    if not printed:
        print("    no tensile contacts above {:.1e}".format(TENSION_TOLERANCE))

    open_contacts = []
    for contact in contacts:
        fn_plus, fn_minus, fu, fv = contact_forces[contact["index"]]
        if abs(fn_plus) <= OPEN_CONTACT_TOLERANCE and abs(fn_minus) <= OPEN_CONTACT_TOLERANCE:
            open_contacts.append((contact, fn_plus, fn_minus, fu, fv))

    if open_contacts:
        print("  open contacts with zero compression and zero tension:")
        for contact, fn_plus, fn_minus, fu, fv in open_contacts:
            print(
                "    edge {edge}, contact {index}, point {point_index}: position={position}, "
                "fn+={fn_plus:.9g}, fn-={fn_minus:.9g}, fn_eff={fn_eff:.9g}, "
                "fu={fu:.9g}, fv={fv:.9g}".format(
                    edge=contact["edge"],
                    index=contact["index"],
                    point_index=contact["point_index"],
                    position=format_point(contact["point"]),
                    fn_plus=fn_plus,
                    fn_minus=fn_minus,
                    fn_eff=fn_plus - fn_minus,
                    fu=fu,
                    fv=fv,
                )
            )
    else:
        print("  no open contacts below {:.1e}".format(OPEN_CONTACT_TOLERANCE))

    sliding_contacts = []
    for contact in contacts:
        fn_plus, fn_minus, fu, fv = contact_forces[contact["index"]]
        utilization = friction_utilization(fn_plus, fu, fv)
        if utilization is None or utilization < 1.0 - FRICTION_LIMIT_TOLERANCE:
            continue
        sliding_contacts.append((contact, fn_plus, fn_minus, fu, fv, utilization))

    if sliding_contacts:
        print("  contacts at friction limit (about to slide):")
        for contact, fn_plus, fn_minus, fu, fv, utilization in sliding_contacts:
            print(
                "    edge {edge}, contact {index}, point {point_index}: position={position}, "
                "fn+={fn_plus:.9g}, fn-={fn_minus:.9g}, fn_eff={fn_eff:.9g}, "
                "fu={fu:.9g}, fv={fv:.9g}, |ft|/(mu*fn+)={utilization:.9g}, about to slide".format(
                    edge=contact["edge"],
                    index=contact["index"],
                    point_index=contact["point_index"],
                    position=format_point(contact["point"]),
                    fn_plus=fn_plus,
                    fn_minus=fn_minus,
                    fn_eff=fn_plus - fn_minus,
                    fu=fu,
                    fv=fv,
                    utilization=utilization,
                )
            )
    else:
        print("  no contacts at friction limit within {:.1e}".format(FRICTION_LIMIT_TOLERANCE))


def plot_diagnostics(result, diagnostics):
    """Render the safe region and outward boundary test points."""
    plt.rcParams["svg.fonttype"] = "none"
    polygon = np.asarray(result.outer_polygon, dtype=float)
    closed_polygon = np.vstack([polygon, polygon[0]])
    figure, axes = plt.subplots(figsize=(9, 6))
    axes.fill(polygon[:, 0], polygon[:, 1], color="#0072B2", alpha=0.12, label="safe region")
    axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], color="#0072B2", linewidth=1.5)
    for diagnostic in diagnostics:
        axes.plot(
            diagnostic.segment[:, 0],
            diagnostic.segment[:, 1],
            color="black",
            linewidth=1.3,
            alpha=0.75,
        )

    midpoints = np.asarray([diagnostic.midpoint for diagnostic in diagnostics])
    outside = np.asarray([diagnostic.outside_load for diagnostic in diagnostics])
    axes.scatter(midpoints[:, 0], midpoints[:, 1], color="black", s=24, label="boundary midpoint", zorder=5)
    axes.scatter(outside[:, 0], outside[:, 1], color="red", marker="x", s=36, label="outside test load", zorder=6)
    for diagnostic in diagnostics:
        axes.annotate(
            diagnostic.label,
            xy=diagnostic.midpoint,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    axes.set_xlim(XLIM)
    axes.set_ylim(YLIM)
    axes.set_xlabel("Block 2 load Fx")
    axes.set_ylabel("Block 2 load Fz")
    axes.set_title("Example 18-4 viewport boundary failure-mode check")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.3)
    axes.text(
        0.02,
        0.02,
        "diagnostic viewport: Fx={} Fz={}\nred x = visible boundary midpoint offset outward by {:.1e}".format(
            XLIM,
            YLIM,
            BOUNDARY_OFFSET,
        ),
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    legend = axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    legend.set_draggable(True)
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    example18 = load_example18_module()
    assembly = build_example18_assembly(example18)
    load_node, load_dofs, application_points = load_setup(example18, assembly)
    print("example 18-4 full three-block assembly: load node {}".format(load_node))
    print("hidden hand-force component bound: {:.6g}".format(APPLICATION_FORCE_BOUND))
    print("diagnostic viewport: Fx={}, Fz={}".format(XLIM, YLIM))
    print("2D front/back tie constraints enabled: {}".format(ENFORCE_2D_TIES))
    compression_problem = hidden_load_problem(assembly, load_node, load_dofs, application_points, penalty=False)
    penalty_problem = hidden_load_problem(assembly, load_node, load_dofs, application_points, penalty=True)
    print(
        "compression ties: contact pairs={}, contact equalities={}, end-load pairs={}, end-load equalities={}".format(
            compression_problem.contact_pair_count,
            compression_problem.contact_tie_count,
            compression_problem.application_pair_count,
            compression_problem.application_tie_count,
        )
    )
    print(
        "penalty ties:     contact pairs={}, contact equalities={}, end-load pairs={}, end-load equalities={}".format(
            penalty_problem.contact_pair_count,
            penalty_problem.contact_tie_count,
            penalty_problem.application_pair_count,
            penalty_problem.application_tie_count,
        )
    )

    result = solve_support_dual(compression_problem)
    print(
        "2D-tied dual support result: bounded={}, boundary edges={}, bounded directions={}/{}".format(
            result.is_bounded,
            len(result.outer_polygon),
            result.statuses.count("optimal"),
            len(result.statuses),
        )
    )

    diagnostics = boundary_diagnostics(result, compression_problem, penalty_problem, assembly, XLIM, YLIM)
    print("viewport boundary diagnostics: {}".format(len(diagnostics)))
    for diagnostic in diagnostics:
        report_tensile_contacts(diagnostic, assembly)

    figure = plot_diagnostics(result, diagnostics)
    save_svg(figure, OUTPUT_SVG)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
