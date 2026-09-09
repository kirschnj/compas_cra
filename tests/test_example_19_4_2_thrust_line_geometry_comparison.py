import importlib.util
import io
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial import ConvexHull

EXAMPLE_PATH = Path(__file__).parents[1] / "docs" / "examples" / "19_4_2_rbe_thrust_line_geometry_comparison.py"


@pytest.fixture(scope="module")
def example():
    specification = importlib.util.spec_from_file_location("example_19_4_2_tests", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results(example):
    return example.analyze_geometries(support_directions=72, contour_directions=72)


def test_centerline_profiles_share_endpoints_and_height(example):
    profiles = [example.build_centerline_profile(spec) for spec in example.PROFILE_SPECS]
    for profile in profiles:
        np.testing.assert_allclose(
            profile.point(profile.parameter_start),
            [-5.5, 0.0],
            atol=2e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            profile.point(profile.parameter_end),
            [5.5, 0.0],
            atol=2e-10,
            rtol=0.0,
        )
        assert profile.point(profile.apex_parameter)[1] == pytest.approx(5.5, abs=2e-10)

    symmetric = profiles[1]
    assert symmetric.catenary_a == pytest.approx(3.403175752758407, abs=1e-10)
    for profile in profiles[2:]:
        assert profile.catenary_a == pytest.approx(3.1700306352287755, abs=1e-9)
        assert abs(profile.catenary_u0) == pytest.approx(0.5493819637545261, abs=1e-9)
        assert profile.catenary_c == pytest.approx(8.898123755078469, abs=1e-9)


def test_asymmetric_profiles_are_mirrored(example):
    negative = example.build_centerline_profile(example.PROFILE_SPECS[2])
    positive = example.build_centerline_profile(example.PROFILE_SPECS[3])
    for parameter in np.linspace(negative.parameter_start, negative.parameter_end, 51):
        negative_point = negative.point(parameter)
        positive_point = positive.point(-parameter)
        np.testing.assert_allclose(negative_point, [-positive_point[0], positive_point[1]], atol=2e-10, rtol=0.0)


def test_generated_arches_have_expected_discrete_geometry(example, results):
    assert len(results) == 4
    for result in results:
        geometry = result.geometry
        assert len(geometry.block_polygons) == example.NUM_BLOCKS
        assert len(geometry.interfaces) == example.NUM_BLOCKS - 1
        assert len(geometry.weights) == example.NUM_BLOCKS
        assert np.all(geometry.weights > 0.0)
        np.testing.assert_allclose(
            geometry.weights,
            [result.assembly.node_block(node).volume() * example.DENSITY for node in result.assembly.graph.nodes()],
            atol=1e-12,
            rtol=0.0,
        )
        assert np.ptp(geometry.segment_arclengths) <= 1e-12
        assert np.allclose(geometry.left_ground[:, 1], 0.0, atol=1e-12)
        assert np.allclose(geometry.right_ground[:, 1], 0.0, atol=1e-12)
        assert geometry.left_ground[0, 0] < geometry.left_ground[1, 0]
        assert geometry.right_ground[0, 0] < geometry.right_ground[1, 0]

        for polygon in geometry.block_polygons:
            assert len(ConvexHull(polygon).vertices) == len(polygon)
        for parameter in geometry.joint_parameters[1:-1]:
            outer = example.offset_point(geometry.profile, parameter, 0.5 * example.THICKNESS)
            inner = example.offset_point(geometry.profile, parameter, -0.5 * example.THICKNESS)
            assert np.linalg.norm(outer - inner) == pytest.approx(example.THICKNESS, abs=1e-12)


def test_all_four_stability_checks_and_right_load_regions_are_available(results):
    for result in results:
        assert result.self_weight.feasible
        assert result.self_weight.equilibrium_residual <= 1e-10
        assert result.rbe_boundary is not None
        assert len(result.rbe_boundary) >= 3
        assert ConvexHull(result.rbe_boundary).volume > 0.0
        assert result.rbe_center is not None
        assert result.right_load_status == "available"
        assert result.family


def test_actual_gravity_remains_vertical_for_every_profile(example, results):
    for result in results:
        loads = np.asarray(
            example.external_force_setup(result.assembly, example.DENSITY, None),
            dtype=float,
        ).reshape((-1, 6))
        np.testing.assert_allclose(loads[:, [0, 1, 3, 4, 5]], 0.0, atol=0.0, rtol=0.0)
        assert np.all(loads[:, 2] < 0.0)


def test_force_diagrams_use_geometry_specific_block_weights(example, results):
    total_weights = []
    for result in results:
        sample = example.representative_sample(result)
        free_weights = result.geometry.weights[:0:-1]
        diagram = example.force_diagram(sample.load, free_weights)
        np.testing.assert_allclose(diagram.pole, sample.load, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.diff(diagram.nodes[:, 1]), free_weights, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            np.diff(diagram.directions, axis=0),
            np.column_stack([np.zeros(example.NUM_BLOCKS - 1), free_weights]),
            atol=1e-12,
            rtol=0.0,
        )
        trace_points = example.trace_polyline(sample.trace)
        for segment, direction in zip(np.diff(trace_points, axis=0), sample.trace.directions):
            assert example.cross_2d(segment, direction) == pytest.approx(0.0, abs=2e-9)
        np.testing.assert_array_equal(sample.trace.block_indices, np.arange(19, 0, -1))
        np.testing.assert_array_equal(sample.trace.interface_indices, np.arange(18, -1, -1))
        assert sample.trace.right_load_point[0] == pytest.approx(sample.trace.right_insertion_x)
        assert sample.trace.right_load_point[1] == pytest.approx(0.0)
        total_weights.append(float(np.sum(result.geometry.weights)))
    assert len({round(weight, 6) for weight in total_weights}) == 3


def test_every_family_trace_is_joint_pressure_admissible(example, results):
    for result in results:
        assert len(result.family) == 72
        for sample in result.family:
            assert sample.interval.feasible
            assert sample.admissible_radius <= sample.rbe_radius + 1e-10
            checks = example.joint_checks(sample.trace, result.geometry)
            assert len(checks) == example.NUM_BLOCKS - 1
            assert all(check.valid for check in checks)
            assert all(-1e-9 <= check.parameter <= 1.0 + 1e-9 for check in checks)
            assert all(check.normal_component > 0.0 for check in checks)
            assert all(check.friction_utilization <= 1.0 + 1e-9 for check in checks)
            assert {check.interface_index for check in checks} == set(range(example.NUM_BLOCKS - 1))


def test_pressure_path_boundaries_coincide_with_rbe_boundaries(example, results):
    for result in results:
        np.testing.assert_allclose(
            [sample.admissible_radius for sample in result.family],
            [sample.rbe_radius for sample in result.family],
            atol=1e-12,
            rtol=0.0,
        )
        rbe_area = example.polygon_area(result.rbe_boundary)
        pressure_path_area = example.polygon_area([sample.load for sample in result.family])
        assert pressure_path_area == pytest.approx(rbe_area, rel=5e-3)


def test_asymmetric_pair_has_matching_areas_and_subcritical_friction(example, results):
    negative, positive = results[2:]
    assert example.polygon_area(negative.rbe_boundary) == pytest.approx(
        example.polygon_area(positive.rbe_boundary), rel=3e-10
    )
    assert example.overlay_metrics(negative)["pressure_path_area"] == pytest.approx(
        example.overlay_metrics(positive)["pressure_path_area"], rel=3e-10
    )
    for result in (negative, positive):
        utilization = max(
            check.friction_utilization
            for sample in result.family
            for check in example.joint_checks(sample.trace, result.geometry)
        )
        assert utilization < 1.0


def test_no_solution_branch_and_both_figures_render_valid_svg(example, results):
    unavailable = replace(
        results[0],
        family=[],
        rbe_center=None,
        right_load_status="no joint-admissible right-load solution space",
    )
    figures = (
        example.plot_geometry_comparison([unavailable] + results[1:]),
        example.plot_comparison_overlay([unavailable] + results[1:]),
    )
    for figure in figures:
        stream = io.StringIO()
        figure.savefig(stream, format="svg", bbox_inches="tight")
        ET.fromstring(stream.getvalue())
        plt.close(figure)
