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
from compas_cra.equilibrium import GeometryScenarioProblem
from compas_cra.equilibrium import rbe_robust_sample
from compas_cra.equilibrium import rbe_robust_support_dual
from compas_cra.equilibrium import rbe_robust_support_primal
from compas_cra.equilibrium import rbe_uncertainty_geometry
from compas_cra.equilibrium import rbe_uncertainty_geometry_sample
from compas_cra.equilibrium import rbe_uncertainty_geometry_support
from compas_cra.equilibrium import rbe_uncertainty_geometry_support_dual
from compas_cra.equilibrium import rbe_uncertainty_geometry_support_primal
from compas_cra.equilibrium.rbe_robust import _prepare_problem
from compas_cra.equilibrium.rbe_uncertainty_geometry import _normal_tilt_scenarios
from compas_cra.equilibrium.rbe_uncertainty_geometry import _point_offset_scenarios
from compas_cra.equilibrium.rbe_uncertainty_geometry import _prepare_geometry_scenarios


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


def geometry_scenario_from_problem(problem):
    return GeometryScenarioProblem(
        equilibrium=problem.equilibrium,
        inequalities=problem.inequalities,
        inequality_rhs=problem.inequality_rhs,
        baseline_load=problem.baseline_load,
        load_projection=problem.load_projection,
        variable_count=problem.equilibrium.shape[1],
        name="explicit nominal",
    )


def interface_points(assembly):
    interface = assembly.graph.edge_attribute((0, 1), "interfaces")[0]
    return [xyz(point) for point in interface.points]


def test_nominal_geometry_matches_existing_robust_solvers():
    assembly = two_block_assembly()
    load_dofs = [(1, "fx"), (1, "fy")]

    robust_sample = rbe_robust_sample(assembly, load_dofs, density=1, num_directions=12)
    robust_primal = rbe_robust_support_primal(assembly, load_dofs, density=1, num_directions=12)
    robust_dual = rbe_robust_support_dual(assembly, load_dofs, density=1, num_directions=12)

    geometry_sample = rbe_uncertainty_geometry_sample(assembly, load_dofs, density=1, num_directions=12)
    geometry_primal = rbe_uncertainty_geometry_support_primal(assembly, load_dofs, density=1, num_directions=12)
    geometry_dual = rbe_uncertainty_geometry_support_dual(assembly, load_dofs, density=1, num_directions=12)

    assert geometry_sample.radial_limits == pytest.approx(robust_sample.radial_limits)
    assert geometry_primal.support_values == pytest.approx(robust_primal.support_values)
    assert geometry_dual.support_values == pytest.approx(robust_dual.support_values)
    assert np.asarray(geometry_dual.outer_polygon) == pytest.approx(np.asarray(robust_dual.outer_polygon))


def test_duplicate_nominal_geometry_scenarios_match_nominal_result():
    assembly = two_block_assembly()
    nominal = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=8)
    duplicated = rbe_uncertainty_geometry_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        interface_scale_factors=[1.0, 1.0],
        num_directions=8,
    )

    assert duplicated.support_values == pytest.approx(nominal.support_values)


def test_explicit_geometry_scenario_matches_nominal_result():
    assembly = two_block_assembly()
    problem = _prepare_problem(assembly, [(1, "fx"), (1, "fy")], 0.84, 1.0, None)
    scenario = geometry_scenario_from_problem(problem)

    nominal = rbe_robust_support_dual(assembly, [(1, "fx"), (1, "fy")], density=1, num_directions=8)
    explicit = rbe_uncertainty_geometry_support_dual(
        assembly,
        [(1, "fx"), (1, "fy")],
        density=1,
        geometry_scenarios=[scenario],
        num_directions=8,
    )

    assert explicit.support_values == pytest.approx(nominal.support_values)


def test_region_scaling_changes_moment_arms_and_preserves_input_assembly():
    assembly = two_block_assembly()
    original_points = interface_points(assembly)

    _, nominal = _prepare_geometry_scenarios(
        assembly,
        [(1, "fx"), (1, "fy")],
        0.84,
        1.0,
        None,
        None,
        None,
        [1.0],
        None,
        None,
        None,
        None,
    )
    _, scaled = _prepare_geometry_scenarios(
        assembly,
        [(1, "fx"), (1, "fy")],
        0.84,
        1.0,
        None,
        None,
        None,
        [0.5],
        None,
        None,
        None,
        None,
    )

    assert scaled[0].equilibrium.toarray() != pytest.approx(nominal[0].equilibrium.toarray())
    assert np.asarray(interface_points(assembly)) == pytest.approx(np.asarray(original_points))


def test_point_offset_bounds_change_equilibrium_moment_arms():
    assembly = two_block_assembly()
    _, nominal = _prepare_geometry_scenarios(
        assembly=assembly,
        load_dofs=[(1, "fx"), (1, "fy")],
        mu=0.84,
        density=1.0,
        external_forces=None,
        load_application_points=None,
        application_force_bound=None,
        interface_scale_factors=None,
        point_offset_vectors=None,
        normal_tilt_vectors=None,
        contact_failure_scenarios=None,
        geometry_scenarios=None,
    )
    _, sampled = _prepare_geometry_scenarios(
        assembly=assembly,
        load_dofs=[(1, "fx"), (1, "fy")],
        mu=0.84,
        density=1.0,
        external_forces=None,
        load_application_points=None,
        application_force_bound=None,
        interface_scale_factors=None,
        point_offset_vectors=None,
        normal_tilt_vectors=None,
        contact_failure_scenarios=None,
        geometry_scenarios=None,
        point_offset_bounds=[0.1, 0.0],
        point_offset_sample_count=3,
        point_offset_seed=10,
    )

    assert len(sampled) == 3
    assert any(
        scenario.equilibrium.toarray() != pytest.approx(nominal[0].equilibrium.toarray()) for scenario in sampled
    )


def test_point_offset_bounds_generate_independent_in_plane_samples():
    assembly = two_block_assembly()
    scenarios = _point_offset_scenarios(
        assembly,
        point_offset_vectors=None,
        point_offset_bounds=[0.1, 0.2],
        sample_count=4,
        seed=5,
    )
    offsets = scenarios[1][0][((0, 1), 0)]

    assert len(scenarios) == 4
    assert scenarios[0][0] is None
    assert offsets.shape == (4, 2)
    assert np.all(offsets[:, 0] <= 0.1)
    assert np.all(offsets[:, 0] >= -0.1)
    assert np.all(offsets[:, 1] <= 0.2)
    assert np.all(offsets[:, 1] >= -0.2)
    assert not np.allclose(offsets, offsets[0])


def test_point_offset_bounds_change_equilibrium_without_normal_offset():
    assembly = two_block_assembly()
    _, bounded = _prepare_geometry_scenarios(
        assembly=assembly,
        load_dofs=[(1, "fx"), (1, "fy")],
        mu=0.84,
        density=1.0,
        external_forces=None,
        load_application_points=None,
        application_force_bound=None,
        interface_scale_factors=None,
        point_offset_vectors=None,
        normal_tilt_vectors=None,
        contact_failure_scenarios=None,
        geometry_scenarios=None,
        point_offset_bounds=[0.1, 0.0],
        point_offset_sample_count=3,
        point_offset_seed=7,
    )

    assert len(bounded) == 3
    assert any(
        bounded[0].equilibrium.toarray() != pytest.approx(scenario.equilibrium.toarray()) for scenario in bounded[1:]
    )


def test_normal_tilts_rotate_frame_based_equilibrium_directions():
    assembly = two_block_assembly()
    _, nominal = _prepare_geometry_scenarios(
        assembly,
        [(1, "fx"), (1, "fy")],
        0.84,
        1.0,
        None,
        None,
        None,
        None,
        None,
        [[0.0, 0.0]],
        None,
        None,
    )
    _, tilted = _prepare_geometry_scenarios(
        assembly,
        [(1, "fx"), (1, "fy")],
        0.84,
        1.0,
        None,
        None,
        None,
        None,
        None,
        [[math.radians(5.0), 0.0]],
        None,
        None,
    )

    assert tilted[0].equilibrium.toarray() != pytest.approx(nominal[0].equilibrium.toarray())
    assert tilted[0].inequalities.shape == nominal[0].inequalities.shape


def test_normal_tilt_bounds_generate_per_interface_samples():
    assembly = two_block_assembly()
    scenarios = _normal_tilt_scenarios(
        assembly,
        normal_tilt_vectors=None,
        normal_tilt_bounds=[0.0, math.radians(5.0)],
        sample_count=4,
        seed=6,
    )
    tilts = [scenario[0][((0, 1), 0)] for scenario in scenarios[1:]]

    assert len(scenarios) == 4
    assert scenarios[0][0] is None
    assert all(tilt[0] == pytest.approx(0.0) for tilt in tilts)
    assert all(-math.radians(5.0) <= tilt[1] <= math.radians(5.0) for tilt in tilts)
    assert len({float(tilt[1]) for tilt in tilts}) > 1


def test_contact_failure_constraints_zero_selected_contact_variables_only():
    assembly = two_block_assembly()
    _, nominal = _prepare_geometry_scenarios(
        assembly,
        [(1, "fx"), (1, "fy")],
        0.84,
        1.0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    _, failed = _prepare_geometry_scenarios(
        assembly,
        [(1, "fx"), (1, "fy")],
        0.84,
        1.0,
        None,
        None,
        None,
        None,
        None,
        None,
        [{"points": [(0, 1, 0, 0)]}],
        None,
    )

    extra_rows = failed[0].inequalities.toarray()[nominal[0].inequalities.shape[0] :]
    assert extra_rows.shape == (6, nominal[0].variable_count)
    assert extra_rows[:, :3] == pytest.approx(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
    )
    assert np.count_nonzero(extra_rows[:, 3:]) == 0
    assert failed[0].inequality_rhs[-6:] == pytest.approx(np.zeros(6))


def test_geometry_primal_and_dual_support_formulations_match():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "interface_scale_factors": [1.0, 0.95],
        "point_offset_bounds": [0.02, 0.0],
        "point_offset_sample_count": 4,
        "point_offset_seed": 11,
        "normal_tilt_vectors": [[0.0, 0.0], [math.radians(2.0), 0.0]],
        "num_directions": 8,
    }

    primal = rbe_uncertainty_geometry_support_primal(assembly, [(1, "fx"), (1, "fy")], **kwargs)
    dual = rbe_uncertainty_geometry_support_dual(assembly, [(1, "fx"), (1, "fy")], **kwargs)

    assert primal.support_values == pytest.approx(dual.support_values)
    assert rbe_uncertainty_geometry(assembly, [(1, "fx"), (1, "fy")], **kwargs).support_formulation == "dual"
    assert rbe_uncertainty_geometry_support(assembly, [(1, "fx"), (1, "fy")], **kwargs).support_formulation == "dual"


def test_geometry_radial_points_satisfy_dual_halfspaces():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "interface_scale_factors": [1.0, 0.95],
        "point_offset_bounds": [0.02, 0.0],
        "point_offset_sample_count": 4,
        "point_offset_seed": 12,
        "num_directions": 8,
    }
    sampled = rbe_uncertainty_geometry_sample(assembly, [(1, "fx"), (1, "fy")], **kwargs)
    dual = rbe_uncertainty_geometry_support_dual(assembly, [(1, "fx"), (1, "fy")], **kwargs)

    for point in sampled.boundary_points:
        for direction_0, direction_1, support in dual.halfspaces:
            assert direction_0 * point[0] + direction_1 * point[1] <= support + 1e-7


def test_four_point_hidden_application_works_with_geometry_uncertainty():
    assembly = two_block_assembly()
    kwargs = {
        "density": 1,
        "load_application_points": {1: four_grasp_points(assembly, 1)},
        "application_force_bound": 10.0,
        "interface_scale_factors": [1.0, 0.95],
        "num_directions": 8,
    }

    primal = rbe_uncertainty_geometry_support_primal(assembly, [(1, "fx"), (1, "fz")], **kwargs)
    dual = rbe_uncertainty_geometry_support_dual(assembly, [(1, "fx"), (1, "fz")], **kwargs)

    assert primal.is_bounded
    assert dual.is_bounded
    assert primal.support_values == pytest.approx(dual.support_values)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"interface_scale_factors": []}, "interface_scale_factors"),
        ({"interface_scale_factors": [0.0]}, "positive finite"),
        ({"point_offset_vectors": [[0.0, 0.0, 0.0]]}, "no longer supported"),
        ({"point_offset_bounds": [0.0, 0.0, 0.0]}, "2D bound"),
        ({"point_offset_bounds": [-0.1, 0.1]}, "nonnegative"),
        ({"point_offset_bounds": [0.1, 0.1], "point_offset_sample_count": 0}, "positive integer"),
        ({"point_offset_bounds": [0.1, 0.1], "point_offset_seed": 0.5}, "integer or None"),
        ({"normal_tilt_vectors": [[0.0, 0.0, 0.0]]}, "2D vector"),
        ({"normal_tilt_vectors": [[0.0, 0.0]], "normal_tilt_bounds": [0.1, 0.1]}, "cannot be combined"),
        ({"normal_tilt_bounds": [0.0, 0.0, 0.0]}, "2D bound"),
        ({"normal_tilt_bounds": [0.0, -0.1]}, "nonnegative"),
        ({"normal_tilt_bounds": [0.1, 0.1], "normal_tilt_sample_count": 0}, "positive integer"),
        ({"normal_tilt_bounds": [0.1, 0.1], "normal_tilt_seed": 0.5}, "integer or None"),
        ({"contact_failure_scenarios": []}, "contact_failure_scenarios"),
        ({"contact_failure_scenarios": [{"points": [(0, 1, 0, 99)]}]}, "Unknown failed point"),
    ],
)
def test_invalid_generated_geometry_inputs_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rbe_uncertainty_geometry_support_dual(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            num_directions=4,
            **kwargs,
        )


def test_geometry_scenarios_cannot_be_combined_with_generated_inputs():
    assembly = two_block_assembly()
    problem = _prepare_problem(assembly, [(1, "fx"), (1, "fy")], 0.84, 1.0, None)
    scenario = geometry_scenario_from_problem(problem)

    with pytest.raises(ValueError, match="cannot be combined"):
        rbe_uncertainty_geometry_support_dual(
            assembly,
            [(1, "fx"), (1, "fy")],
            density=1,
            geometry_scenarios=[scenario],
            interface_scale_factors=[1.0],
            num_directions=4,
        )


def test_invalid_explicit_geometry_scenario_raises():
    invalid = GeometryScenarioProblem(
        equilibrium=csr_matrix((2, 3)),
        inequalities=csr_matrix((1, 3)),
        inequality_rhs=np.zeros(1),
        baseline_load=np.zeros(2),
        load_projection=csr_matrix((1, 3)),
        variable_count=3,
    )

    with pytest.raises(ValueError, match="load_projection"):
        rbe_uncertainty_geometry_support_dual(
            two_block_assembly(),
            [(1, "fx"), (1, "fy")],
            density=1,
            geometry_scenarios=[invalid],
            num_directions=4,
        )
