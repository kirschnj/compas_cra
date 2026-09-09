import math

import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Translation
from compas_assembly.datastructures import Block
from scipy.sparse import csr_matrix

from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support_dual
from compas_cra.equilibrium import rbe_robust_support_primal
from compas_cra.equilibrium import rbe_uncertainty_disturb
from compas_cra.equilibrium import rbe_uncertainty_disturb_sample
from compas_cra.equilibrium import rbe_uncertainty_disturb_support
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_dual
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_primal
from compas_cra.equilibrium.rbe_robust import _prepare_problem
from compas_cra.equilibrium.rbe_robust import _solve_dual_support
from compas_cra.equilibrium.rbe_uncertainty_disturb import _scenario_loads
from compas_cra.equilibrium.rbe_uncertainty_disturb import _tilted_gravity_load
from compas_cra.equilibrium.rbe_uncertainty_disturb import _tilted_gravity_vector
from compas_cra.equilibrium.rbe_uncertainty_disturb import _tilt_load_transform
from compas_cra.equilibrium.rbe_uncertainty_disturb import _uncertainty_shifts


def two_block_assembly():
    support = Box(1, 1, 1)
    free = Box(1, 1, 1, frame=Frame.worldXY().transformed(Translation.from_vector([0, 0, 1])))

    assembly = CRA_Assembly()
    assembly.add_block(Block.from_shape(support), node=0)
    assembly.add_block(Block.from_shape(free), node=1)
    assembly.set_boundary_conditions([0])

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


def tilted_external_force(theta):
    """Return the external force shift that converts vertical gravity to tilted gravity."""
    return {1: [-math.sin(theta), 0.0, 1.0 - math.cos(theta), 0.0, 0.0, 0.0]}


def support_values_or_infinity(result):
    return np.asarray([np.inf if support is None else support for support in result.support_values], dtype=float)


def test_zero_uncertainty_matches_existing_robust_solvers():
    assembly = two_block_assembly()
    load_dofs = [(1, "fx"), (1, "fy")]
    robust_sample = rbe_robust_sample(assembly, load_dofs, density=1, num_directions=12)
    robust_primal = rbe_robust_support_primal(assembly, load_dofs, density=1, num_directions=12)
    robust_dual = rbe_robust_support_dual(assembly, load_dofs, density=1, num_directions=12)

    uncertain_sample = rbe_uncertainty_disturb_sample(assembly, load_dofs, density=1, num_directions=12)
    uncertain_primal = rbe_uncertainty_disturb_support_primal(assembly, load_dofs, density=1, num_directions=12)
    uncertain_dual = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, density=1, num_directions=12)

    assert uncertain_sample.radial_limits == pytest.approx(robust_sample.radial_limits)
    assert uncertain_primal.support_values == pytest.approx(robust_primal.support_values)
    assert uncertain_dual.support_values == pytest.approx(robust_dual.support_values)
    assert np.asarray(uncertain_dual.outer_polygon) == pytest.approx(np.asarray(robust_dual.outer_polygon))


def test_uncertainty_shifts_from_dofs_match_direct_basis():
    assembly = two_block_assembly()
    problem = _prepare_problem(assembly, [(1, "fx"), (1, "fy")], 0.84, 1.0, None)
    vertices = np.asarray([[-0.2, 0.3], [0.4, -0.5]])
    basis = csr_matrix((np.ones(2), ([0, 2], [0, 1])), shape=(problem.equilibrium.shape[0], 2))

    dof_shifts = _uncertainty_shifts(
        problem,
        assembly=assembly,
        uncertainty_load_dofs=[(1, "fx"), (1, "fz")],
        uncertainty_vertices=vertices,
    )
    basis_shifts = _uncertainty_shifts(
        problem,
        uncertainty_vertices=vertices,
        uncertainty_basis=basis,
    )

    assert np.asarray(dof_shifts) == pytest.approx(np.asarray(basis_shifts))


def test_uncertain_support_matches_axis_intersection_of_vertex_sets():
    assembly = two_block_assembly()
    vertices = np.asarray([[-0.1], [0.2]])
    uncertain = rbe_uncertainty_disturb_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        uncertainty_vertices=vertices,
        uncertainty_load_dofs=[(1, "fx")],
        num_directions=4,
    )
    vertex_results = [
        rbe_robust_support_dual(
            assembly,
            [(1, "fx"), (1, "fy")],
            density=1,
            external_forces={1: [float(vertex[0]), 0, 0, 0, 0, 0]},
            num_directions=4,
        )
        for vertex in vertices
    ]
    expected = np.min(np.asarray([result.support_values for result in vertex_results]), axis=0)

    assert uncertain.support_values == pytest.approx(expected)


def test_uncertain_primal_and_dual_support_formulations_match():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "uncertainty_vertices": [[-0.1], [0.2]],
        "uncertainty_load_dofs": [(1, "fx")],
        "num_directions": 12,
    }
    primal = rbe_uncertainty_disturb_support_primal(assembly, [(1, "fx"), (1, "fy")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fy")], **kwargs)

    assert primal.support_values == pytest.approx(dual.support_values)
    assert np.asarray(primal.halfspaces) == pytest.approx(np.asarray(dual.halfspaces))
    assert rbe_uncertainty_disturb(assembly, [(1, "fx"), (1, "fy")], **kwargs).support_formulation == "dual"
    assert rbe_uncertainty_disturb_support(assembly, [(1, "fx"), (1, "fy")], **kwargs).support_formulation == "dual"


def test_uncertain_radial_points_satisfy_dual_halfspaces():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "uncertainty_vertices": [[-0.1], [0.2]],
        "uncertainty_load_dofs": [(1, "fx")],
        "num_directions": 12,
    }
    sampled = rbe_uncertainty_disturb_sample(assembly, [(1, "fx"), (1, "fy")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fy")], **kwargs)

    for point in sampled.boundary_points:
        for direction_0, direction_1, support in dual.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


@pytest.mark.parametrize(
    "uncertainty_vertices, uncertainty_load_dofs, uncertainty_basis, message",
    [
        ([1.0, 2.0], [(1, "fx")], None, "2D array"),
        ([[1.0, 2.0]], [(1, "fx")], None, "dimension"),
        ([[1.0]], None, None, "Provide uncertainty_basis or uncertainty_load_dofs"),
        ([[1.0]], [(0, "fx")], None, "is a support"),
        ([[1.0]], [(1, "bad")], None, "Unknown uncertainty load component"),
        ([[1.0]], None, csr_matrix((5, 1)), "row count"),
    ],
)
def test_invalid_uncertainty_inputs_raise(
    uncertainty_vertices,
    uncertainty_load_dofs,
    uncertainty_basis,
    message,
):
    with pytest.raises(ValueError, match=message):
        rbe_uncertainty_disturb_support_dual(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            uncertainty_vertices=uncertainty_vertices,
            uncertainty_load_dofs=uncertainty_load_dofs,
            uncertainty_basis=uncertainty_basis,
            num_directions=4,
        )


def test_uncertain_empty_safe_load_set_raises():
    with pytest.raises(ValueError, match="safe load set is empty"):
        rbe_uncertainty_disturb_support_primal(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            uncertainty_vertices=[[2.0]],
            uncertainty_load_dofs=[(1, "fz")],
            num_directions=4,
        )


def test_four_point_hidden_application_works_with_uncertainty():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "load_application_points": {1: four_grasp_points(assembly, 1)},
        "application_force_bound": 10.0,
        "uncertainty_vertices": [[-0.1], [0.1]],
        "uncertainty_load_dofs": [(1, "fx")],
        "num_directions": 8,
    }
    primal = rbe_uncertainty_disturb_support_primal(assembly, [(1, "fx"), (1, "fz")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fz")], **kwargs)

    assert primal.is_bounded
    assert dual.is_bounded
    assert primal.support_values == pytest.approx(dual.support_values)


def test_zero_tilt_gravity_load_matches_existing_baseline():
    assembly = two_block_assembly()
    problem = _prepare_problem(assembly, [(1, "fx"), (1, "fy")], 0.84, 1.0, None)

    assert _tilted_gravity_vector(0.0) == pytest.approx([0.0, 0.0, -1.0])
    assert _tilted_gravity_load(assembly, 1.0, None, 0.0, problem.equilibrium.shape[0]) == pytest.approx(
        problem.baseline_load
    )


def test_structure_frame_small_tilt_matches_linearized_load_uncertainty():
    assembly = two_block_assembly()
    angle = 1e-5
    row_count = _prepare_problem(assembly, [(1, "fx"), (1, "fy")], 0.84, 1.0, None).equilibrium.shape[0]
    linearized_basis = csr_matrix(([-1.0], ([0], [0])), shape=(row_count, 1))

    tilt = rbe_uncertainty_disturb_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        tilt_angle_bounds=(-angle, angle),
        tilt_load_frame="structure",
        num_directions=8,
    )
    linearized = rbe_uncertainty_disturb_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        uncertainty_vertices=[[-angle], [angle]],
        uncertainty_basis=linearized_basis,
        num_directions=8,
    )

    assert tilt.support_values == pytest.approx(linearized.support_values, abs=1e-7)


def test_structure_frame_tilt_support_matches_intersection_of_manual_tilt_scenarios():
    assembly = two_block_assembly()
    angle = math.radians(3)
    angles = [-angle, 0.0, angle]
    tilt = rbe_uncertainty_disturb_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        tilt_angle_bounds=(-angle, angle),
        tilt_angle_count=3,
        tilt_load_frame="structure",
        num_directions=8,
    )
    manual_results = [
        rbe_robust_support_dual(
            assembly,
            [(1, "fx"), (1, "fy")],
            density=1,
            external_forces=tilted_external_force(theta),
            num_directions=8,
        )
        for theta in angles
    ]
    expected = np.min(np.asarray([result.support_values for result in manual_results]), axis=0)

    assert tilt.support_values == pytest.approx(expected)


def test_world_frame_single_tilt_support_rotates_query_direction():
    assembly = two_block_assembly()
    load_dofs = [(1, "fx"), (1, "fz")]
    angle = math.radians(12)
    world = rbe_uncertainty_disturb_support_dual(
        assembly,
        load_dofs,
        density=1,
        tilt_angles=[angle],
        num_directions=4,
    )
    local_problem = _prepare_problem(
        assembly,
        load_dofs,
        0.84,
        1.0,
        tilted_external_force(angle),
    )
    transform = _tilt_load_transform(angle, load_dofs)
    expected_statuses = []
    expected = []
    for direction in world.directions:
        status, support = _solve_dual_support(local_problem, transform.dot(np.asarray(direction)), {})
        expected_statuses.append(status)
        expected.append(support)

    assert world.statuses == expected_statuses
    for actual, support in zip(world.support_values, expected):
        if support is None:
            assert actual is None
        else:
            assert actual == pytest.approx(support)


def test_world_frame_tilt_radial_points_satisfy_dual_halfspaces():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "tilt_angle_bounds": (-math.radians(2), math.radians(2)),
        "tilt_angle_count": 3,
        "num_directions": 8,
    }
    sampled = rbe_uncertainty_disturb_sample(assembly, [(1, "fx"), (1, "fz")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fz")], **kwargs)

    for point in sampled.boundary_points:
        for direction_0, direction_1, support in dual.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


def test_combined_tilt_and_load_uncertainty_uses_cartesian_product_and_is_conservative():
    assembly = two_block_assembly()
    load_dofs = [(1, "fx"), (1, "fz")]
    problem = _prepare_problem(assembly, load_dofs, 0.84, 1.0, None)
    angle = math.radians(2)
    scenario_loads = _scenario_loads(
        problem,
        assembly=assembly,
        density=1,
        external_forces=None,
        uncertainty_load_dofs=[(1, "fx")],
        uncertainty_vertices=[[-0.05], [0.05]],
        uncertainty_basis=None,
        tilt_angle_bounds=(-angle, angle),
        tilt_angle_count=3,
    )

    load_only = rbe_uncertainty_disturb_support_dual(
        assembly,
        load_dofs,
        density=1,
        uncertainty_vertices=[[-0.05], [0.05]],
        uncertainty_load_dofs=[(1, "fx")],
        num_directions=8,
    )
    tilt_only = rbe_uncertainty_disturb_support_dual(
        assembly,
        load_dofs,
        density=1,
        tilt_angle_bounds=(-angle, angle),
        tilt_angle_count=3,
        num_directions=8,
    )
    combined = rbe_uncertainty_disturb_support_dual(
        assembly,
        load_dofs,
        density=1,
        uncertainty_vertices=[[-0.05], [0.05]],
        uncertainty_load_dofs=[(1, "fx")],
        tilt_angle_bounds=(-angle, angle),
        tilt_angle_count=3,
        num_directions=8,
    )

    assert len(scenario_loads) == 6
    assert np.all(support_values_or_infinity(combined) <= support_values_or_infinity(load_only) + 1e-8)
    assert np.all(support_values_or_infinity(combined) <= support_values_or_infinity(tilt_only) + 1e-8)


def test_tilt_primal_and_dual_support_formulations_match():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "tilt_angle_bounds": (-math.radians(2), math.radians(2)),
        "tilt_angle_count": 3,
        "num_directions": 8,
    }
    primal = rbe_uncertainty_disturb_support_primal(assembly, [(1, "fx"), (1, "fz")], **kwargs)
    dual = rbe_uncertainty_disturb_support_dual(assembly, [(1, "fx"), (1, "fz")], **kwargs)

    assert primal.support_values == pytest.approx(dual.support_values)


@pytest.mark.parametrize(
    "tilt_angle_bounds, tilt_angle_count, message",
    [
        ((0.0, 0.0), 2, "finite increasing"),
        ((0.1, -0.1), 2, "finite increasing"),
        ((0.0,), 2, "exactly two"),
        ((0.0, 0.1), 1, "greater than or equal to 2"),
        ((0.0, 0.1), 2.5, "integer"),
    ],
)
def test_invalid_tilt_inputs_raise(tilt_angle_bounds, tilt_angle_count, message):
    with pytest.raises(ValueError, match=message):
        rbe_uncertainty_disturb_support_dual(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            tilt_angle_bounds=tilt_angle_bounds,
            tilt_angle_count=tilt_angle_count,
            num_directions=4,
        )


@pytest.mark.parametrize(
    "kwargs, load_dofs, message",
    [
        ({"tilt_angles": []}, [(1, "fx"), (1, "fz")], "tilt_angles"),
        ({"tilt_angles": [0.0], "tilt_angle_bounds": (-0.1, 0.1)}, [(1, "fx"), (1, "fz")], "not both"),
        ({"tilt_load_frame": "bad"}, [(1, "fx"), (1, "fz")], "tilt_load_frame"),
        ({"tilt_angles": [0.1]}, [(1, "fx"), (1, "fy")], "world-frame tilt"),
    ],
)
def test_invalid_explicit_tilt_inputs_raise(kwargs, load_dofs, message):
    with pytest.raises(ValueError, match=message):
        rbe_uncertainty_disturb_support_dual(
            two_block_assembly(),
            load_dofs,
            density=1,
            num_directions=4,
            **kwargs,
        )
