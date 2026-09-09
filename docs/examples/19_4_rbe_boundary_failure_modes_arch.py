"""Diagnose failure modes just outside full-arch robust safe-load boundaries."""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compas_assembly.datastructures import Block
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
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
from compas_cra.geometry import Arch
from compas_cra.viewers import cra_view_ex
from compas_cra.viewers.cra_view import _load_compas_view2

HEIGHT = 5
SPAN = 10
THICKNESS = 1
DEPTH = 1
NUM_BLOCKS = 20
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 36
APPLICATION_FORCE_BOUND = 1e6
BOUNDARY_OFFSET = 1e-3
TENSION_TOLERANCE = 1e-7
OPEN_CONTACT_TOLERANCE = 1e-7
HALFSPACE_TOLERANCE = 1e-7
SUPPORT_TOLERANCE = 1e-8
ENFORCE_2D_TIES = True
PAIR_COORDINATE_TOLERANCE = 1e-8
SHOW_COMPAS_VIEW2_ASSEMBLY = plt.get_backend().lower() != "agg"
OUTPUT_SVG = Path(__file__).with_suffix(".svg")


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
    midpoint: np.ndarray
    outside_load: np.ndarray
    compression_result: object
    penalty_result: object


def point_coordinates(point):
    """Return point coordinates as a plain list."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    return list(point)


def xyz_array(point):
    """Return a point as a three-component NumPy array."""
    return np.asarray(point_coordinates(point), dtype=float)


def arch_center():
    """Return the center of the arch circular axis."""
    radius = HEIGHT / 2.0 + SPAN**2 / (8.0 * HEIGHT)
    return [0.0, 0.0, HEIGHT - radius]


def unique_xz_points(coordinates, tolerance=1e-9):
    """Return unique face points by x-z coordinates."""
    unique = []
    for point in coordinates:
        if not any(
            abs(point[0] - other[0]) <= tolerance and abs(point[2] - other[2]) <= tolerance for other in unique
        ):
            unique.append(point)
    return unique


def radial_alignment_score(points):
    """Return how far two x-z points deviate from one arch radial line."""
    center = arch_center()
    vectors = [[point[0] - center[0], point[2] - center[2]] for point in points]
    lengths = [(vector[0] ** 2 + vector[1] ** 2) ** 0.5 for vector in vectors]
    if min(lengths) <= 1e-12:
        return None
    cross = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
    return abs(cross) / (lengths[0] * lengths[1])


def exposed_radial_side_vertices(block):
    """Return vertices of the loaded block's right exposed radial face."""
    candidates = []
    for face in block.faces():
        coordinates = [point_coordinates(block.vertex_coordinates(vertex)) for vertex in block.face_vertices(face)]
        y_values = [point[1] for point in coordinates]
        if max(y_values) - min(y_values) <= 1e-9:
            continue
        unique_points = unique_xz_points(coordinates)
        if len(unique_points) != 2:
            continue
        score = radial_alignment_score(unique_points)
        if score is None or score > 1e-8:
            continue
        centroid_x = sum(point[0] for point in coordinates) / len(coordinates)
        candidates.append((centroid_x, coordinates))
    if not candidates:
        raise ValueError("Could not find an exposed radial load face for the arch block.")
    return max(candidates, key=lambda item: item[0])[1]


def build_full_arch(num_blocks=NUM_BLOCKS):
    """Build a full left-to-right arch assembly with block 0 fixed."""
    arch = Arch(
        height=HEIGHT,
        span=SPAN,
        thickness=THICKNESS,
        depth=DEPTH,
        num_blocks=num_blocks,
        extra_support=False,
    )
    assembly = CRA_Assembly()
    for node, mesh in enumerate(reversed(arch.blocks())):
        assembly.add_block(mesh.copy(cls=Block), node=node)
    assembly.set_boundary_conditions([0])
    assembly_interfaces_numpy(assembly, nmax=10, amin=1e-2, tmax=1e-2)
    return assembly


def assembly_bounds(assembly):
    """Return coordinate-wise bounds for all blocks in an assembly."""
    coordinates = []
    for node in assembly.graph.nodes():
        block = assembly.graph.node_attribute(node, "block")
        for vertex in block.vertices():
            coordinates.append(point_coordinates(block.vertex_coordinates(vertex)))

    lower = [min(coordinate[axis] for coordinate in coordinates) for axis in range(3)]
    upper = [max(coordinate[axis] for coordinate in coordinates) for axis in range(3)]
    return lower, upper


def assembly_display_scale(assembly):
    """Return a readable scale for load-direction arrows."""
    lower, upper = assembly_bounds(assembly)
    spans = [upper[axis] - lower[axis] for axis in range(3)]
    return max(spans + [0.25])


def load_application_centroid(assembly, load_node):
    """Return the centroid of the right exposed radial-face load application points."""
    points = exposed_radial_side_vertices(assembly.graph.node_attribute(load_node, "block"))
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


def add_load_direction_arrows(viewer, assembly, origin):
    """Add positive Fx and Fz load-direction arrows to a compas_view2 viewer."""
    _, _, Arrow = _load_compas_view2()
    length = 0.18 * assembly_display_scale(assembly)
    viewer.add(
        Arrow(origin, [length, 0, 0], head_portion=0.28, head_width=0.10, body_width=0.025),
        facecolor=(0.9, 0.0, 0.0),
        show_lines=False,
    )
    viewer.add(
        Arrow(origin, [0, 0, length], head_portion=0.28, head_width=0.10, body_width=0.025),
        facecolor=(0.0, 0.2, 0.9),
        show_lines=False,
    )


def show_compas_view2_assembly(assembly, load_node):
    """Show the full arch assembly in a compas_view2 window."""
    view2_app, _, _ = _load_compas_view2()
    viewer = view2_app.App(
        title="Example 19-4 boundary failure-mode arch assembly",
        width=1200,
        height=750,
        viewmode="shaded",
        show_grid=True,
    )
    cra_view_ex(
        viewer,
        assembly,
        blocks=True,
        interfaces=True,
        forces=False,
        forcesdirect=False,
        forcesline=False,
        weights=False,
        displacements=False,
    )
    add_load_direction_arrows(viewer, assembly, load_application_centroid(assembly, load_node))
    print("compas_view2 arch view: red arrow = +Fx, blue arrow = +Fz")
    viewer.run()


def load_setup(assembly):
    """Return the visible load and hidden application-point setup."""
    load_node = max(assembly.graph.nodes())
    block = assembly.graph.node_attribute(load_node, "block")
    load_dofs = [(load_node, "fx"), (load_node, "fz")]
    application_points = {load_node: [np.asarray(point, dtype=float) for point in exposed_radial_side_vertices(block)]}
    return load_node, load_dofs, application_points


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
    """Return hidden load-variable point pairs for the right exposed radial face."""
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


def boundary_diagnostics(result, compression_problem, penalty_problem, assembly):
    """Create diagnostics for unique visible-load polygon boundary edges."""
    polygon = np.asarray(result.outer_polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        raise ValueError("Example 19-4 requires a bounded dual outer polygon with at least three vertices.")

    diagnostics = []
    seen = set()
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        midpoint = 0.5 * (point + next_point)
        halfspace = boundary_halfspace(midpoint, result.halfspaces)
        normal = halfspace[:2]
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= HALFSPACE_TOLERANCE:
            continue
        key = tuple(np.round(halfspace, 10))
        if key in seen:
            continue
        seen.add(key)

        outside_load = midpoint + BOUNDARY_OFFSET * normal / normal_length
        compression = solve_fixed_visible_load(compression_problem, outside_load)
        penalty = solve_penalty_tension(penalty_problem, assembly, outside_load)
        diagnostics.append(
            BoundaryDiagnostic(
                label="B{}".format(len(diagnostics) + 1),
                halfspace=halfspace,
                midpoint=midpoint,
                outside_load=outside_load,
                compression_result=compression,
                penalty_result=penalty,
            )
        )
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
        tangent = math.sqrt(fu * fu + fv * fv)
        if fn_plus > TENSION_TOLERANCE:
            utilization = "{:.9g}".format(tangent / (MU * fn_plus))
        else:
            utilization = "nan"
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
                utilization=utilization,
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


def plot_diagnostics(result, diagnostics):
    """Render the safe region and outward boundary test points."""
    plt.rcParams["svg.fonttype"] = "none"
    polygon = np.asarray(result.outer_polygon, dtype=float)
    closed_polygon = np.vstack([polygon, polygon[0]])
    figure, axes = plt.subplots(figsize=(9, 6))
    axes.fill(polygon[:, 0], polygon[:, 1], color="#0072B2", alpha=0.12, label="safe region")
    axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], color="#0072B2", linewidth=1.5)

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

    coordinates = np.vstack([polygon, midpoints, outside])
    spans = coordinates.max(axis=0) - coordinates.min(axis=0)
    padding = np.maximum(0.12 * spans, 0.1)
    axes.set_xlim(float(coordinates[:, 0].min() - padding[0]), float(coordinates[:, 0].max() + padding[0]))
    axes.set_ylim(float(coordinates[:, 1].min() - padding[1]), float(coordinates[:, 1].max() + padding[1]))
    axes.set_xlabel("Last-block load Fx")
    axes.set_ylabel("Last-block load Fz")
    axes.set_title("Example 19-4 2D-tied boundary failure-mode check")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.3)
    axes.text(
        0.02,
        0.02,
        "red x = boundary midpoint offset outward by {:.1e}".format(BOUNDARY_OFFSET),
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
    assembly = build_full_arch()
    load_node, load_dofs, application_points = load_setup(assembly)
    print("example 19-4 full arch: {} blocks, load node {}".format(NUM_BLOCKS, load_node))
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

    diagnostics = boundary_diagnostics(result, compression_problem, penalty_problem, assembly)
    for diagnostic in diagnostics:
        report_tensile_contacts(diagnostic, assembly)

    figure = plot_diagnostics(result, diagnostics)
    save_svg(figure, OUTPUT_SVG)
    if SHOW_COMPAS_VIEW2_ASSEMBLY:
        plt.show(block=False)
        show_compas_view2_assembly(assembly, load_node)
    elif plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
