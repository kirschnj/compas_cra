"""Compare two L10 boundary diagnostics for example 18.

This script checks the full three-block, block-0-fixed robust model from
``18_rbe_robust_3block.py``. It solves the 45-degree support LP, then checks
points just inside, on, and outside the resulting ``Fx + Fz`` boundary.

Two hidden load-application models are compared:

* independent four-point loads, matching the current robust example;
* pair-tied 2D loads, with ``q0 = q1`` and ``q2 = q3`` for hand checking.
"""

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
from compas_cra.equilibrium.cra_helper import free_nodes
from compas_cra.equilibrium.cra_helper import friction_setup
from compas_cra.equilibrium.cra_helper import num_vertices
from compas_cra.equilibrium.rbe_robust import _application_bound_rows
from compas_cra.equilibrium.rbe_robust import _application_load_basis

MU = 0.8
DENSITY = 1.0
APPLICATION_FORCE_BOUND = 1e3
LOAD_NODE = 2
LOAD_DOFS = [(LOAD_NODE, "fx"), (LOAD_NODE, "fz")]
COMPONENT_INDEX = {"fx": 0, "fy": 1, "fz": 2}
L10_DIRECTION = np.asarray([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
DELTA = 1e-3
OUTSIDE_LOAD = np.asarray([-0.2, 0.8])
XLIM = (-1.0, 0.1)
YLIM = (-0.5, 1.0)
TENSION_TOLERANCE = 1e-7
RUN_GEOMETRY_SWEEP = True


@dataclass(frozen=True)
class LoadModel:
    """Description of one hidden load-application model."""

    name: str
    title: str
    kind: str
    hidden_force_count: int


@dataclass
class RobustProblem:
    """Sparse matrices defining one fixed hidden-load robust LP."""

    equilibrium: csr_matrix
    inequalities: csr_matrix
    inequality_rhs: np.ndarray
    baseline_load: np.ndarray
    load_projection: csr_matrix
    base_force_count: int
    hidden_force_count: int
    load_model: LoadModel


@dataclass
class L10Case:
    """Computed L10 diagnostics for one load model."""

    problem: RobustProblem
    support_result: object
    support_load: np.ndarray
    support_value: float
    l10_constant: float
    inside_load: np.ndarray
    outside_load: np.ndarray


INDEPENDENT_MODEL = LoadModel(
    name="Independent four-point load model",
    title="Independent q0..q3",
    kind="independent",
    hidden_force_count=8,
)
PAIR_TIED_MODEL = LoadModel(
    name="Pair-tied 2D load model",
    title="Pair-tied q0=q1, q2=q3",
    kind="pair_tied",
    hidden_force_count=4,
)


def load_example18_module():
    """Load the numeric example-18 module by file path."""
    path = Path(__file__).with_name("18_rbe_robust_3block.py")
    spec = importlib.util.spec_from_file_location("example18", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_geometry_kwargs():
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


def build_full_example18_assembly(example18):
    """Build the full three-block example-18 assembly with block 0 fixed."""
    geometry = example18.Arch(**base_geometry_kwargs())
    return example18.build_assembly(geometry, block_nodes=[0, 1, 2], support_nodes=[0])


def prepare_assembly_and_points(example18):
    """Return the example-18 assembly and block-2 rightmost-side points."""
    assembly = build_full_example18_assembly(example18)
    return assembly, {
        LOAD_NODE: example18.rightmost_side_vertices(assembly.graph.node_attribute(LOAD_NODE, "block")),
    }


def xyz_array(point):
    """Return a point as a three-component NumPy array."""
    return np.asarray(point, dtype=float)


def loaded_node_base_row(assembly):
    """Return the first equilibrium row for the loaded free body."""
    node_keys = list(assembly.graph.nodes())
    node_index = {node: index for index, node in enumerate(node_keys)}
    free = free_nodes(assembly)
    free_position = free.index(node_index[LOAD_NODE])
    return free_position * 6


def pair_tied_application_load_basis(assembly, points, row_count, base_force_count):
    """Build hidden load rows for ``q0 = q1`` and ``q2 = q3``."""
    points = [xyz_array(point) for point in points]
    if len(points) != 4:
        raise ValueError("Pair-tied L10 check requires exactly four application points.")

    base_row = loaded_node_base_row(assembly)
    block = assembly.graph.node_attribute(LOAD_NODE, "block")
    center = np.asarray([block.center().x, block.center().y, block.center().z], dtype=float)

    rows = []
    columns = []
    data = []
    projection_rows = []
    projection_columns = []
    projection_data = []

    for point_index, point in enumerate(points):
        pair_offset = 0 if point_index < 2 else 2
        offset = point - center
        for load_index, (_, component) in enumerate(LOAD_DOFS):
            component_index = COMPONENT_INDEX[component]
            column = pair_offset + load_index
            unit = np.zeros(3)
            unit[component_index] = 1.0
            moment = np.cross(offset, unit)

            rows.append(base_row + component_index)
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

    application_basis = csr_matrix((data, (rows, columns)), shape=(row_count, PAIR_TIED_MODEL.hidden_force_count))
    load_projection = csr_matrix(
        (projection_data, (projection_rows, projection_columns)),
        shape=(2, base_force_count + PAIR_TIED_MODEL.hidden_force_count),
    )
    return application_basis, load_projection


def application_load_basis(assembly, application_points, load_model, row_count, base_force_count):
    """Build hidden load rows for one load model."""
    points = [xyz_array(point) for point in application_points[LOAD_NODE]]
    if load_model.kind == "independent":
        return _application_load_basis(
            assembly,
            LOAD_DOFS,
            LOAD_NODE,
            points,
            row_count,
            base_force_count,
        )
    if load_model.kind == "pair_tied":
        return pair_tied_application_load_basis(assembly, points, row_count, base_force_count)
    raise ValueError("Unknown load model: {}".format(load_model.kind))


def add_hidden_load_rows(base_equilibrium, base_inequalities, baseline_load, assembly, application_points, load_model):
    """Append hidden load variables and their bounds to a base LP."""
    application_basis, load_projection = application_load_basis(
        assembly,
        application_points,
        load_model,
        base_equilibrium.shape[0],
        base_equilibrium.shape[1],
    )
    hidden_force_count = application_basis.shape[1]
    if hidden_force_count != load_model.hidden_force_count:
        raise ValueError("Unexpected hidden force count for {}.".format(load_model.name))

    equilibrium = hstack([base_equilibrium, application_basis], format="csr")
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

    return RobustProblem(
        equilibrium=equilibrium,
        inequalities=inequalities,
        inequality_rhs=inequality_rhs,
        baseline_load=baseline_load,
        load_projection=load_projection,
        base_force_count=base_equilibrium.shape[1],
        hidden_force_count=hidden_force_count,
        load_model=load_model,
    )


def build_compression_problem(assembly, application_points, load_model):
    """Build a compression-only robust LP for one hidden load model."""
    base_equilibrium = equilibrium_setup(assembly, penalty=False).tocsr()
    friction = friction_setup(assembly, MU, penalty=False).tocsr()
    baseline_load = np.asarray(external_force_setup(assembly, DENSITY, None), dtype=float).ravel()

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
    return add_hidden_load_rows(
        base_equilibrium,
        base_inequalities,
        baseline_load,
        assembly,
        application_points,
        load_model,
    )


def solve_support_direction(problem, direction):
    """Solve ``max direction dot u`` and return the HiGHS result."""
    objective = -np.asarray(problem.load_projection.transpose().dot(direction)).ravel()
    return linprog(
        objective,
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=problem.equilibrium,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * problem.equilibrium.shape[1],
        method="highs",
    )


def solve_fixed_visible_load(problem, load):
    """Check feasibility with the visible load fixed to ``load``."""
    equality = vstack([problem.equilibrium, problem.load_projection], format="csr")
    rhs = np.concatenate([-problem.baseline_load, np.asarray(load, dtype=float)])
    return linprog(
        np.zeros(problem.equilibrium.shape[1]),
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(None, None)] * problem.equilibrium.shape[1],
        method="highs",
    )


def solve_penalty_tension_check(assembly, application_points, load_model, load):
    """Minimize tensile normal force needed to equilibrate one visible load."""
    base_equilibrium = equilibrium_setup(assembly, penalty=True).tocsr()
    friction = friction_setup(assembly, MU, penalty=True).tocsr()
    baseline_load = np.asarray(external_force_setup(assembly, DENSITY, None), dtype=float).ravel()
    problem = add_hidden_load_rows(
        base_equilibrium,
        friction,
        baseline_load,
        assembly,
        application_points,
        load_model,
    )
    equality = vstack([problem.equilibrium, problem.load_projection], format="csr")
    equality_rhs = np.concatenate([-baseline_load, np.asarray(load, dtype=float)])

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


def compute_l10_case(problem):
    """Compute support point and nearby loads for one problem."""
    support = solve_support_direction(problem, L10_DIRECTION)
    if support.status != 0:
        raise RuntimeError("{} support solve failed: {}".format(problem.load_model.name, support.message))

    support_load = np.asarray(problem.load_projection.dot(support.x), dtype=float).ravel()
    support_value = float(L10_DIRECTION.dot(support_load))
    l10_constant = support_value / L10_DIRECTION[1]
    return L10Case(
        problem=problem,
        support_result=support,
        support_load=support_load,
        support_value=support_value,
        l10_constant=l10_constant,
        inside_load=support_load - DELTA * L10_DIRECTION,
        outside_load=OUTSIDE_LOAD,
    )


def feasibility_label(result):
    """Return a compact text label for one fixed-load LP result."""
    if result.status == 0:
        return "feasible"
    if result.status == 2:
        return "infeasible"
    return "solver status {}".format(result.status)


def contact_table(assembly):
    """Return contact metadata in the same order as RBE force variables."""
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
                        "frame": interface.frame,
                    }
                )
                contact_index += 1
    return contacts


def format_point(point):
    """Format a 3D point for compact console output."""
    coordinates = xyz_array(point)
    return "[{:.6g}, {:.6g}, {:.6g}]".format(coordinates[0], coordinates[1], coordinates[2])


def penalty_contact_forces(result, assembly):
    """Return penalty contact variables as ``[fn_plus, fn_minus, fu, fv]`` rows."""
    contact_count = num_vertices(assembly)
    return np.asarray(result.x[: 4 * contact_count], dtype=float).reshape((-1, 4))


def penalty_tension(result, assembly):
    """Return the total tensile normal force for one penalty LP result."""
    return float(np.sum(penalty_contact_forces(result, assembly)[:, 1]))


def report_penalty_tension(label, result, assembly):
    """Print the tensile contacts needed by one penalty LP result."""
    print("\n  penalty tension diagnostic for {}:".format(label))
    if result.status != 0:
        print("    penalty LP failed: status {}, {}".format(result.status, result.message))
        return

    contact_forces = penalty_contact_forces(result, assembly)
    total_tension = penalty_tension(result, assembly)
    print("    minimum total fn_minus={:.9g}".format(total_tension))

    tensile_contacts = []
    for contact in contact_table(assembly):
        contact_index = contact["index"]
        fn_plus, fn_minus, fu, fv = contact_forces[contact_index]
        if fn_minus <= TENSION_TOLERANCE:
            continue
        effective_normal = fn_plus - fn_minus
        tangent = math.sqrt(fu * fu + fv * fv)
        if fn_plus > TENSION_TOLERANCE:
            utilization = tangent / (MU * fn_plus)
            utilization_text = "{:.9g}".format(utilization)
        else:
            utilization_text = "nan"
        tensile_contacts.append(
            "    edge {edge}, contact {index}, point {point_index}: "
            "position={position}, "
            "fn+={fn_plus:.9g}, fn-={fn_minus:.9g}, fn_eff={fn_eff:.9g}, "
            "fu={fu:.9g}, fv={fv:.9g}, |ft|/(mu*fn+)={utilization}".format(
                edge=contact["edge"],
                index=contact_index,
                point_index=contact["point_index"],
                position=format_point(contact["point"]),
                fn_plus=fn_plus,
                fn_minus=fn_minus,
                fn_eff=effective_normal,
                fu=fu,
                fv=fv,
                utilization=utilization_text,
            )
        )

    if not tensile_contacts:
        print("    no tensile contacts above {:.1e}".format(TENSION_TOLERANCE))
        return
    print("    contacts requiring tensile normal force:")
    for line in tensile_contacts:
        print(line)


def report_contact_state(case, assembly, title, edge_filter=None):
    """Print normal and friction usage at the L10 support solution."""
    problem = case.problem
    solution = case.support_result.x
    contact_forces = np.asarray(solution[: problem.base_force_count], dtype=float).reshape((-1, 3))
    slacks = problem.inequality_rhs - problem.inequalities.dot(solution)
    friction_rows = friction_setup(assembly, MU, penalty=False).shape[0]
    normal_row_start = friction_rows

    print("\n{}:".format(title))
    for contact in contact_table(assembly):
        if edge_filter is not None and contact["edge"] not in edge_filter:
            continue
        contact_index = contact["index"]
        fn, fu, fv = contact_forces[contact_index]
        normal_force = max(0.0, fn)
        friction_slacks = slacks[8 * contact_index : 8 * contact_index + 8]
        normal_slack = slacks[normal_row_start + contact_index]
        if normal_force <= 1e-8:
            fv_ratio = math.nan
            state = "open"
        else:
            fv_ratio = fv / (MU * normal_force)
            state = "carrying"
        print(
            "  contact {index} point {point_index}: state={state}, "
            "position={position}, "
            "fn={fn:.9g}, fu={fu:.9g}, fv={fv:.9g}, "
            "fv/(mu*fn)={ratio}, min_friction_slack={min_slack:.3g}, normal_slack={normal_slack:.3g}".format(
                index=contact_index,
                point_index=contact["point_index"],
                state=state,
                position=format_point(contact["point"]),
                fn=fn,
                fu=fu,
                fv=fv,
                ratio="nan" if math.isnan(fv_ratio) else "{:.9g}".format(fv_ratio),
                min_slack=float(np.min(friction_slacks)),
                normal_slack=float(normal_slack),
            )
        )


def report_l10_contact_state(case, assembly):
    """Print the b0-b1 and b1-b2 contact state at the L10 support solution."""
    report_contact_state(
        case,
        assembly,
        "b0-b1 and b1-b2 contact state at L10 support point",
        edge_filter={(0, 1), (1, 2)},
    )


def hidden_point_force(hidden, load_model, point_index):
    """Return the hidden point force vector for one application point."""
    if load_model.kind == "independent":
        return np.asarray([hidden[2 * point_index], 0.0, hidden[2 * point_index + 1]])
    if load_model.kind == "pair_tied":
        pair_offset = 0 if point_index < 2 else 2
        return np.asarray([hidden[pair_offset], 0.0, hidden[pair_offset + 1]])
    raise ValueError("Unknown load model: {}".format(load_model.kind))


def report_hidden_forces(case, assembly, application_points):
    """Print hidden point forces and their generated moment."""
    problem = case.problem
    hidden = np.asarray(
        case.support_result.x[problem.base_force_count : problem.base_force_count + problem.hidden_force_count],
        dtype=float,
    )
    block = assembly.graph.node_attribute(LOAD_NODE, "block")
    center = np.asarray([block.center().x, block.center().y, block.center().z], dtype=float)

    print("\nhidden load variables at L10 support point:")
    if problem.load_model.kind == "pair_tied":
        print("  q01=q0=q1: qx={:.9g}, qz={:.9g}".format(hidden[0], hidden[1]))
        print("  q23=q2=q3: qx={:.9g}, qz={:.9g}".format(hidden[2], hidden[3]))

    resultant = np.zeros(3)
    resultant_moment = np.zeros(3)
    for point_index, point in enumerate(application_points[LOAD_NODE]):
        point = xyz_array(point)
        force = hidden_point_force(hidden, problem.load_model, point_index)
        moment = np.cross(point - center, force)
        resultant += force
        resultant_moment += moment
        print(
            "  q{} at {}: qx={:.9g}, qz={:.9g}, My={:.9g}".format(
                point_index,
                [round(value, 6) for value in point.tolist()],
                force[0],
                force[2],
                moment[1],
            )
        )
    print("  resultant Fx={:.9g}, Fz={:.9g}, My={:.9g}".format(resultant[0], resultant[2], resultant_moment[1]))


def active_contact_centroid(case, assembly, edge):
    """Return the x-z centroid of active normal contacts on one edge."""
    problem = case.problem
    solution = case.support_result.x
    contact_forces = np.asarray(solution[: problem.base_force_count], dtype=float).reshape((-1, 3))
    positions = []
    for contact in contact_table(assembly):
        if contact["edge"] != edge:
            continue
        if contact_forces[contact["index"], 0] <= 1e-8:
            continue
        point = xyz_array(contact["point"])
        positions.append(point[[0, 2]])
    if not positions:
        raise ValueError("No active contacts found on edge {}.".format(edge))
    return np.mean(np.asarray(positions, dtype=float), axis=0)


def interface_frame_for_edge(assembly, edge):
    """Return the first contact frame for an assembly edge."""
    for contact in contact_table(assembly):
        if contact["edge"] == edge:
            return contact["frame"]
    raise ValueError("No contact frame found for edge {}.".format(edge))


def block_weight_and_center(assembly, node):
    """Return block weight and center as NumPy data."""
    block = assembly.graph.node_attribute(node, "block")
    center = block.center()
    return block.volume() * DENSITY, np.asarray([center.x, center.z], dtype=float)


def report_pair_tied_hand_check(case, assembly):
    """Print a reduced 2D calculation that reproduces the pair-tied L10 line."""
    if case.problem.load_model.kind != "pair_tied":
        return

    support_point = active_contact_centroid(case, assembly, (0, 1))
    load_transfer_point = active_contact_centroid(case, assembly, (1, 2))
    frame = interface_frame_for_edge(assembly, (1, 2))
    normal = np.asarray(frame.zaxis, dtype=float)
    tangent = np.asarray(frame.yaxis, dtype=float)
    contact_direction = normal + MU * tangent
    contact_x_per_n = contact_direction[0]
    contact_down_per_n = -contact_direction[2]

    weight_1, center_1 = block_weight_and_center(assembly, 1)
    weight_2, _ = block_weight_and_center(assembly, 2)
    arm = load_transfer_point - support_point
    denominator = arm[0] * contact_down_per_n + arm[1] * contact_x_per_n
    total_normal = (center_1[0] - support_point[0]) * weight_1 / denominator
    fx = -contact_x_per_n * total_normal
    fz = weight_2 + contact_down_per_n * total_normal

    print("\npair-tied 2D hand check for the L10 support point:")
    print(
        "  active b0-b1 support point A=(x={:.9g}, z={:.9g})".format(
            support_point[0],
            support_point[1],
        )
    )
    print(
        "  active b1-b2 transfer point B=(x={:.9g}, z={:.9g})".format(
            load_transfer_point[0],
            load_transfer_point[1],
        )
    )
    print(
        "  b1-b2 active friction direction w + mu*v = [{:.9g}, {:.9g}, {:.9g}]".format(
            contact_direction[0],
            contact_direction[1],
            contact_direction[2],
        )
    )
    print(
        "  contact force on block 2 per total normal N: Cx={:.9g} N, Cz=-{:.9g} N".format(
            contact_x_per_n,
            contact_down_per_n,
        )
    )
    print("  block 1 moment about A gives N = (xG1-xA)*W1 / ((xB-xA)*Cdown + (zB-zA)*Cx)")
    print(
        "  N = ({:.9g}-{:.9g})*{:.9g} / (({:.9g})*{:.9g} + ({:.9g})*{:.9g}) = {:.9g}".format(
            center_1[0],
            support_point[0],
            weight_1,
            arm[0],
            contact_down_per_n,
            arm[1],
            contact_x_per_n,
            total_normal,
        )
    )
    print("  block 2 force balance gives Fx = -Cx = {:.9g}".format(fx))
    print("  block 2 force balance gives Fz = W2 + Cdown = {:.9g}".format(fz))
    print("  therefore Fx + Fz = {:.9g}".format(fx + fz))
    print("  solver Fx + Fz = {:.9g}".format(case.l10_constant))


def report_case(case, assembly, application_points):
    """Print all numerical diagnostics for one load model."""
    print("\n" + "=" * 80)
    print(case.problem.load_model.name)
    print("=" * 80)
    print("L10 support direction: {}".format(L10_DIRECTION.tolist()))
    print("support point: Fx={:.9g}, Fz={:.9g}".format(case.support_load[0], case.support_load[1]))
    print("support value d.u={:.9g}".format(case.support_value))
    print("L10 line: Fx + Fz = {:.9g}".format(case.l10_constant))
    print("L10 line: Fz = -Fx + {:.9g}".format(case.l10_constant))
    print("outside check point: Fx={:.9g}, Fz={:.9g}".format(case.outside_load[0], case.outside_load[1]))

    checks = [
        ("inside", case.inside_load),
        ("on L10", case.support_load),
        ("outside", case.outside_load),
    ]
    print("\nfixed visible-load feasibility checks:")
    for label, load in checks:
        result = solve_fixed_visible_load(case.problem, load)
        penalty_result = solve_penalty_tension_check(assembly, application_points, case.problem.load_model, load)
        if penalty_result.status == 0:
            penalty_state = "penalty min fn-={:.9g}".format(penalty_tension(penalty_result, assembly))
        else:
            penalty_state = "penalty status {}".format(penalty_result.status)
        print(
            "  {label}: Fx={fx:.9g}, Fz={fz:.9g}, Fx+Fz={total:.9g} -> {state}; {penalty_state}".format(
                label=label,
                fx=load[0],
                fz=load[1],
                total=float(load[0] + load[1]),
                state=feasibility_label(result),
                penalty_state=penalty_state,
            )
        )
        report_penalty_tension(label, penalty_result, assembly)

    report_l10_contact_state(case, assembly)
    report_hidden_forces(case, assembly, application_points)
    report_pair_tied_hand_check(case, assembly)


def pair_tied_l10_constant(example18, geometry_kwargs, application_force_bound):
    """Compute the pair-tied L10 constant for one geometry and hidden-force bound."""
    global APPLICATION_FORCE_BOUND  # noqa: PLW0603

    old_bound = APPLICATION_FORCE_BOUND
    APPLICATION_FORCE_BOUND = application_force_bound
    try:
        geometry = example18.Arch(**geometry_kwargs)
        assembly = example18.build_assembly(geometry, block_nodes=[0, 1, 2], support_nodes=[0])
        application_points = {
            LOAD_NODE: example18.rightmost_side_vertices(assembly.graph.node_attribute(LOAD_NODE, "block")),
        }
        problem = build_compression_problem(assembly, application_points, PAIR_TIED_MODEL)
        case = compute_l10_case(problem)
        return case.l10_constant
    finally:
        APPLICATION_FORCE_BOUND = old_bound


def report_geometry_sweep(example18, base_geometry_kwargs):
    """Print a compact sweep showing what moves the L10 line."""
    if not RUN_GEOMETRY_SWEEP:
        return

    print("\n" + "=" * 80)
    print("Pair-tied L10 sensitivity sweep")
    print("=" * 80)
    print("Hidden-force bound sweep: this should not move the line if geometry/contact is governing.")
    for bound in (1.0, 100.0, 1000.0, 10000.0):
        constant = pair_tied_l10_constant(example18, base_geometry_kwargs, bound)
        print("  application_force_bound={:.6g}: Fx+Fz={:.9g}".format(bound, constant))

    variations = [
        ("b0_width", (0.3, 1.0), "does not move the b0-b1 interface"),
        ("b1_height", (0.35, 0.65), "moves block-1 centroid/contact arms"),
        ("b2_base", (0.45, 0.75), "changes block-2 weight/centroid/contact geometry"),
        ("b2_top", (0.7, 1.1), "changes block-2 weight/centroid/contact geometry"),
        ("beta", (math.radians(115), math.radians(125)), "changes b1-b2 interface angle"),
        ("gamma", (math.radians(115), math.radians(125)), "changes block-2 shape/interface angle"),
        ("thickness", (0.5, 2.0), "scales weights"),
    ]
    print("\nGeometry sweep with application_force_bound=1000:")
    for name, values, note in variations:
        print("  {} ({})".format(name, note))
        for value in values:
            geometry_kwargs = dict(base_geometry_kwargs)
            geometry_kwargs[name] = value
            constant = pair_tied_l10_constant(example18, geometry_kwargs, APPLICATION_FORCE_BOUND)
            display_value = math.degrees(value) if name in {"beta", "gamma"} else value
            units = " deg" if name in {"beta", "gamma"} else ""
            print("    {}{} -> Fx+Fz={:.9g}".format(display_value, units, constant))


def plot_case(axes, case):
    """Draw one L10 line and its checked points."""
    xs = np.asarray(XLIM)
    axes.plot(
        xs,
        -xs + case.l10_constant,
        color="black",
        linewidth=1.5,
        label="L10: Fx + Fz = {:.6g}".format(case.l10_constant),
    )
    points = np.asarray([case.inside_load, case.support_load, case.outside_load])
    axes.scatter(points[:, 0], points[:, 1], color=["green", "black", "red"], marker="o", zorder=5)
    for label, point in zip(("inside", "on L10", "outside (-0.2, 0.8)"), points):
        axes.annotate(label, xy=point, xytext=(8, 8), textcoords="offset points")
    axes.set_title(case.problem.load_model.title)
    axes.set_xlabel("Fx")
    axes.set_xlim(XLIM)
    axes.set_ylim(YLIM)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.3)
    legend = axes.legend(loc="upper right", fontsize=7)
    legend.set_draggable(True)


def plot_l10_checks(cases):
    """Save a two-panel SVG comparing the two L10 diagnostics."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for axes_item, case in zip(axes, cases):
        plot_case(axes_item, case)
    axes[0].set_ylabel("Fz")
    figure.suptitle("Example 18 L10 checks: independent vs pair-tied hidden loads")
    figure.tight_layout()
    figure.savefig("docs/examples/18_special_check_l10.svg", bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    example18 = load_example18_module()
    assembly, application_points = prepare_assembly_and_points(example18)
    cases = []
    for load_model in (INDEPENDENT_MODEL, PAIR_TIED_MODEL):
        problem = build_compression_problem(assembly, application_points, load_model)
        case = compute_l10_case(problem)
        report_case(case, assembly, application_points)
        cases.append(case)
    report_geometry_sweep(example18, base_geometry_kwargs())
    plot_l10_checks(cases)
