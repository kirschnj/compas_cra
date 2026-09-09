import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Translation
from compas_assembly.datastructures import Block
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import RobustForceResult
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support
from compas_cra.equilibrium import rbe_robust_support_dual
from compas_cra.equilibrium import rbe_robust_support_primal
from compas_cra.equilibrium.rbe_robust import _prepare_problem
from compas_cra.equilibrium.cra_helper import equilibrium_setup
from compas_cra.equilibrium.cra_helper import external_force_setup
from compas_cra.equilibrium.cra_helper import friction_setup
from compas_cra.equilibrium.cra_helper import num_vertices
from compas_cra.geometry import Arch


def two_block_assembly():
    support = Box(1, 1, 1)
    free = Box(1, 1, 1, frame=Frame.worldXY().transformed(Translation.from_vector([0, 0, 1])))

    assembly = CRA_Assembly()
    assembly.add_block(Block.from_shape(support), node=0)
    assembly.add_block(Block.from_shape(free), node=1)
    assembly.set_boundary_conditions([0])
    try:
        assembly_interfaces_numpy(assembly, amin=1e-6, tmax=1e-4)
    except TypeError as error:
        if "cannot unpack non-iterable int object" not in str(error):
            raise
        interface = Mesh()
        corners = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]
        for index, xyz in enumerate(corners):
            interface.add_vertex(key=index, x=xyz[0], y=xyz[1], z=xyz[2])
        interface.add_face([0, 1, 2, 3])
        assembly.add_interfaces_from_meshes([interface], 0, 1)
    return assembly


def xyz(point):
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return np.asarray([point.x, point.y, point.z], dtype=float)
    return np.asarray(point, dtype=float)


def point_from_offset(assembly, node, offset):
    block = assembly.graph.node_attribute(node, "block")
    return (xyz(block.center()) + np.asarray(offset, dtype=float)).tolist()


def four_grasp_points(assembly, node):
    return [
        point_from_offset(assembly, node, [0.5, -0.5, 0.5]),
        point_from_offset(assembly, node, [0.5, 0.5, 0.5]),
        point_from_offset(assembly, node, [0.5, -0.5, -0.5]),
        point_from_offset(assembly, node, [0.5, 0.5, -0.5]),
    ]


def rightmost_side_vertices(block):
    face_coordinates = []
    for face in block.faces():
        coordinates = [xyz(block.vertex_coordinates(vertex)) for vertex in block.face_vertices(face)]
        face_coordinates.append(coordinates)
    return [point.tolist() for point in max(face_coordinates, key=lambda coordinates: np.mean(coordinates, axis=0)[0])]


def construction_arch_stage(stage):
    meshes = list(
        reversed(
            Arch(
                height=5,
                span=10,
                thickness=0.5,
                depth=0.5,
                num_blocks=20,
                extra_support=False,
            ).blocks()
        )
    )
    assembly = CRA_Assembly()
    for node, mesh in enumerate(meshes[:stage]):
        assembly.add_block(mesh.copy(cls=Block), node=node)
    assembly.set_boundary_conditions([0])
    assembly_interfaces_numpy(assembly, nmax=10, amin=1e-2, tmax=1e-2)
    return assembly


def polygon_area(polygon):
    return 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )


def primal_support(assembly, direction, mu=0.84, density=1.0):
    equilibrium = equilibrium_setup(assembly, penalty=False)
    friction = friction_setup(assembly, mu, penalty=False)
    baseline = external_force_setup(assembly, density).flatten()
    vertex_count = num_vertices(assembly)
    force_count = vertex_count * 3

    normal_nonnegative = csr_matrix(
        (
            -np.ones(vertex_count),
            (np.arange(vertex_count), np.arange(vertex_count) * 3),
        ),
        shape=(vertex_count, force_count),
    )
    inequalities = vstack([friction, normal_nonnegative], format="csr")
    load_basis = csr_matrix((np.ones(2), ([0, 1], [0, 1])), shape=(equilibrium.shape[0], 2))

    objective = np.zeros(force_count + 2)
    objective[-2:] = -np.asarray(direction)
    result = linprog(
        objective,
        A_ub=hstack([inequalities, csr_matrix((inequalities.shape[0], 2))]),
        b_ub=np.zeros(inequalities.shape[0]),
        A_eq=hstack([equilibrium, load_basis]),
        b_eq=-baseline,
        bounds=[(None, None)] * (force_count + 2),
        method="highs",
    )
    assert result.status == 0
    return -result.fun


def test_robust_sample_and_support_bound_horizontal_loads():
    assembly = two_block_assembly()
    interface = assembly.graph.edge_attribute((0, 1), "interfaces")[0]

    sampled = rbe_robust_sample(assembly, [(1, "fx"), (1, "fy")], density=1)
    supported = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1)

    assert isinstance(sampled, RobustForceResult)
    assert sampled.origin_feasible
    assert supported.origin_feasible
    assert sampled.feasible_center == pytest.approx([0.0, 0.0])
    assert supported.feasible_center == pytest.approx([0.0, 0.0])
    assert sampled.is_bounded
    assert supported.is_bounded
    assert len(sampled.directions) == 36
    assert len(sampled.radial_limits) == 36
    assert len(supported.support_values) == 36
    assert len(sampled.polygon) >= 3
    assert len(supported.polygon) >= 3
    assert sampled.polygon == sampled.inner_polygon
    assert supported.polygon == supported.outer_polygon
    assert polygon_area(sampled.polygon) > 0
    assert polygon_area(supported.polygon) > 0
    assert interface.forces is None

    for point in sampled.boundary_points:
        for direction_0, direction_1, support in supported.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


def test_dual_support_matches_independent_primal_lp():
    assembly = two_block_assembly()
    result = rbe_robust_support(assembly, [(1, "fx"), (1, "fy")], density=1)

    assert result.support_values[0] == pytest.approx(primal_support(assembly, [1.0, 0.0]))
    assert result.support_values[9] == pytest.approx(primal_support(assembly, [0.0, 1.0]))


def test_primal_and_dual_support_formulations_match():
    assembly = two_block_assembly()
    primal = rbe_robust_support_primal(assembly, [(1, "fx"), (1, "fy")], density=1)
    dual = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1)

    assert primal.support_formulation == "primal"
    assert dual.support_formulation == "dual"
    assert primal.support_values == pytest.approx(dual.support_values)
    assert np.asarray(primal.halfspaces) == pytest.approx(np.asarray(dual.halfspaces))
    assert np.asarray(primal.outer_polygon) == pytest.approx(np.asarray(dual.outer_polygon))
    assert len(primal.support_points) == len(primal.directions)
    assert len(primal.inner_polygon) >= 3
    assert polygon_area(primal.inner_polygon) > 0
    assert primal.polygon == primal.outer_polygon

    for point in primal.support_points:
        for direction_0, direction_1, support in dual.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


def test_legacy_support_api_uses_dual_formulation():
    assembly = two_block_assembly()
    legacy = rbe_robust_support(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=12)
    explicit = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=12)

    assert legacy.support_formulation == "dual"
    assert legacy.support_values == pytest.approx(explicit.support_values)
    assert np.asarray(legacy.polygon) == pytest.approx(np.asarray(explicit.polygon))


def test_four_point_application_load_projection_and_moments():
    assembly = two_block_assembly()
    points = four_grasp_points(assembly, 1)
    problem = _prepare_problem(
        assembly,
        [(1, "fx"), (1, "fz")],
        0.84,
        1.0,
        None,
        {1: points},
        10.0,
    )

    base_force_count = equilibrium_setup(assembly, penalty=False).shape[1]
    hidden_force_count = 8
    assert problem.equilibrium.shape[1] == base_force_count + hidden_force_count

    projection = problem.load_projection.toarray()
    assert projection[0, base_force_count::2] == pytest.approx(np.ones(4))
    assert projection[0, base_force_count + 1 :: 2] == pytest.approx(np.zeros(4))
    assert projection[1, base_force_count::2] == pytest.approx(np.zeros(4))
    assert projection[1, base_force_count + 1 :: 2] == pytest.approx(np.ones(4))

    block = assembly.graph.node_attribute(1, "block")
    offset = xyz(points[0]) - xyz(block.center())
    equilibrium = problem.equilibrium.toarray()
    first_qx = base_force_count
    first_qz = base_force_count + 1
    assert equilibrium[0, first_qx] == pytest.approx(1.0)
    assert equilibrium[2, first_qz] == pytest.approx(1.0)
    assert equilibrium[4, first_qx] == pytest.approx(offset[2])
    assert equilibrium[4, first_qz] == pytest.approx(-offset[0])

    base_inequality_count = friction_setup(assembly, 0.84, penalty=False).shape[0] + num_vertices(assembly)
    assert problem.inequalities.shape[0] == base_inequality_count + 2 * hidden_force_count
    assert problem.inequality_rhs[:base_inequality_count] == pytest.approx(np.zeros(base_inequality_count))
    assert problem.inequality_rhs[base_inequality_count:] == pytest.approx(np.full(2 * hidden_force_count, 10.0))

    bound_rows = problem.inequalities.toarray()[base_inequality_count:]
    assert np.count_nonzero(bound_rows[:, :base_force_count]) == 0
    assert bound_rows[:hidden_force_count, base_force_count:] == pytest.approx(np.eye(hidden_force_count))
    assert bound_rows[hidden_force_count:, base_force_count:] == pytest.approx(-np.eye(hidden_force_count))


def test_four_point_application_primal_and_dual_supports_match():
    assembly = two_block_assembly()
    load_application_points = {1: four_grasp_points(assembly, 1)}
    primal = rbe_robust_support_primal(
        assembly,
        [(1, "fx"), (1, "fz")],
        density=1,
        load_application_points=load_application_points,
        application_force_bound=10.0,
        num_directions=12,
    )
    dual = rbe_robust_support_dual(
        assembly,
        [(1, "fx"), (1, "fz")],
        density=1,
        load_application_points=load_application_points,
        application_force_bound=10.0,
        num_directions=12,
    )

    assert primal.is_bounded
    assert dual.is_bounded
    assert primal.support_values == pytest.approx(dual.support_values)
    assert np.asarray(primal.halfspaces) == pytest.approx(np.asarray(dual.halfspaces))


def test_construction_arch_stage_20_is_feasible_with_four_point_application():
    assembly = construction_arch_stage(20)
    load_node = 19
    block = assembly.graph.node_attribute(load_node, "block")
    result = rbe_robust_support_primal(
        assembly,
        [(load_node, "fx"), (load_node, "fz")],
        mu=0.7,
        density=1,
        load_application_points={load_node: rightmost_side_vertices(block)},
        application_force_bound=1e6,
        num_directions=12,
    )

    assert result.is_bounded
    assert result.statuses.count("optimal") == len(result.directions)
    assert len(result.feasible_center) == 2


@pytest.mark.parametrize(
    "load_dofs",
    [
        [(1, "fx")],
        [(1, "fx"), (1, "fx")],
        [(0, "fx"), (1, "fy")],
        [(1, "invalid"), (1, "fy")],
        [(99, "fx"), (1, "fy")],
    ],
)
def test_invalid_load_dofs_raise(load_dofs):
    with pytest.raises(ValueError):
        rbe_robust_sample(two_block_assembly(), load_dofs, density=1, num_directions=4)


@pytest.mark.parametrize(
    "load_dofs, load_application_points, application_force_bound, message",
    [
        ([(1, "fx"), (1, "fz")], {}, 10.0, "must include the loaded node"),
        ([(1, "fx"), (1, "fz")], {1: []}, 10.0, "at least one point"),
        ([(1, "fx"), (1, "fz")], {1: [[0.0, 0.0]]}, 10.0, "three-dimensional"),
        ([(1, "fx"), (1, "mz")], {1: [[0.0, 0.0, 1.0]]}, 10.0, "force components"),
        ([(1, "fx"), (1, "fz")], {1: [[0.0, 0.0, 1.0]]}, 0.0, "positive finite"),
    ],
)
def test_invalid_load_application_points_raise(
    load_dofs,
    load_application_points,
    application_force_bound,
    message,
):
    with pytest.raises(ValueError, match=message):
        rbe_robust_sample(
            two_block_assembly(),
            load_dofs,
            density=1,
            load_application_points=load_application_points,
            application_force_bound=application_force_bound,
            num_directions=4,
        )


def test_application_force_bound_requires_application_points():
    with pytest.raises(ValueError, match="requires load_application_points"):
        rbe_robust_sample(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            application_force_bound=10.0,
            num_directions=4,
        )


@pytest.mark.parametrize(
    "solver",
    [rbe_robust_sample, rbe_robust_support_primal, rbe_robust_support_dual],
)
def test_empty_safe_load_set_raises(solver):
    upward_force = {1: [0, 0, 2, 0, 0, 0]}
    with pytest.raises(ValueError, match="safe load set is empty"):
        solver(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            external_forces=upward_force,
            num_directions=4,
        )


def test_shifted_safe_load_set_does_not_require_feasible_origin():
    assembly = two_block_assembly()
    external_forces = {1: [2.0, 0, 0, 0, 0, 0]}
    sampled = rbe_robust_sample(assembly, [(1, "fx"), (1, "fy")], density=1, external_forces=external_forces)
    primal = rbe_robust_support_primal(assembly, [(1, "fx"), (1, "fy")], density=1, external_forces=external_forces)
    dual = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1, external_forces=external_forces)

    for result in (sampled, primal, dual):
        assert not result.origin_feasible
        assert result.is_bounded
        assert result.feasible_center[0] < 0

    assert sampled.feasible_center == pytest.approx(primal.feasible_center)
    assert primal.feasible_center == pytest.approx(dual.feasible_center)
    assert primal.support_values == pytest.approx(dual.support_values)

    for direction, limit, point in zip(sampled.directions, sampled.radial_limits, sampled.boundary_points):
        expected = np.asarray(sampled.feasible_center) + limit * np.asarray(direction)
        assert point == pytest.approx(expected)
        for direction_0, direction_1, support in dual.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


def test_known_baseline_force_translates_increment_range():
    assembly = two_block_assembly()
    reference = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1)
    shifted = rbe_robust_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        external_forces={1: [0.1, 0, 0, 0, 0, 0]},
    )

    assert shifted.support_values[0] == pytest.approx(reference.support_values[0] - 0.1)
    assert shifted.support_values[18] == pytest.approx(reference.support_values[18] + 0.1)


@pytest.mark.parametrize(
    "solver",
    [rbe_robust_sample, rbe_robust_support_primal, rbe_robust_support_dual],
)
def test_downward_vertical_load_is_unbounded(solver):
    result = solver(two_block_assembly(), [(1, "fz"), (1, "fx")], density=1, num_directions=12)

    assert not result.is_bounded
    assert "unbounded" in result.statuses
    assert len(result.feasible_center) == 2
    assert result.inner_polygon == []
    assert result.outer_polygon == []
    assert result.polygon == []


def test_plot_robust_results_with_matplotlib_agg():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    assembly = two_block_assembly()
    sampled = rbe_robust_sample(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=12)
    primal = rbe_robust_support_primal(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=12)
    dual = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=12)

    figure, axes = plot_rbe_robust_results(sampled)
    assert axes.figure is figure
    assert all("clipped unbounded" not in line.get_label() for line in axes.lines)

    comparison_figure, comparison_axes = plot_rbe_robust_results(
        [sampled, primal, dual],
        labels=["ray", "primal", "dual"],
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )
    assert comparison_axes.figure is comparison_figure
    assert comparison_axes.get_xlim() == pytest.approx((-1.0, 1.0))
    assert comparison_axes.get_ylim() == pytest.approx((-1.0, 1.0))
    assert all("clipped unbounded" not in line.get_label() for line in comparison_axes.lines)

    import matplotlib.pyplot as plt

    # figure.savefig("robust_result.png", dpi=200, bbox_inches="tight")
    # comparison_figure.savefig("robust_comparison.png", dpi=200, bbox_inches="tight")

    plt.close(figure)
    plt.close(comparison_figure)


def test_plot_clips_unbounded_results_to_explicit_limits():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    assembly = two_block_assembly()
    load_dofs = [(1, "fz"), (1, "fx")]
    sampled = rbe_robust_sample(assembly, load_dofs, density=1, num_directions=12)
    primal = rbe_robust_support_primal(assembly, load_dofs, density=1, num_directions=12)
    dual = rbe_robust_support_dual(assembly, load_dofs, density=1, num_directions=12)

    figure, axes = plot_rbe_robust_results(
        [sampled, primal, dual],
        labels=["ray", "primal", "dual"],
        xlim=(-2.0, 1.0),
        ylim=(-1.0, 1.0),
    )

    line_labels = [line.get_label() for line in axes.lines]
    assert "ray clipped unbounded inner" in line_labels
    assert "primal clipped unbounded outer" in line_labels
    assert "dual clipped unbounded outer" in line_labels
    assert axes.get_xlim() == pytest.approx((-2.0, 1.0))
    assert axes.get_ylim() == pytest.approx((-1.0, 1.0))

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_plot_limits_must_be_complete_finite_and_increasing():
    result = rbe_robust_sample(two_block_assembly(), [(1, "fx"), (1, "fy")], density=1, num_directions=12)

    with pytest.raises(ValueError, match="provided together"):
        plot_rbe_robust_results(result, xlim=(-1.0, 1.0))
    with pytest.raises(ValueError, match="exactly two"):
        plot_rbe_robust_results(result, xlim=(-1.0, 0.0, 1.0), ylim=(-1.0, 1.0))
    with pytest.raises(ValueError, match="finite increasing"):
        plot_rbe_robust_results(result, xlim=(1.0, -1.0), ylim=(-1.0, 1.0))
