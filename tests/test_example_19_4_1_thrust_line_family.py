import io
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

EXAMPLE_PATH = Path(__file__).parents[1] / "docs" / "examples" / "19_4_1_rbe_thrust_line_family_arch.py"
SUPPLIED_CASES = (
    ((-5.0, -8.6), -0.46),
    ((-3.5, -9.05), 0.35),
    ((-2.9, -8.6), 0.47),
    ((-3.5, -8.15), -0.26),
)
EXPECTED_JOINT_OVERRUNS = {
    "R1": {0: 0.0333542962, 18: 0.0278495518},
    "R2": {10: 0.0643839969, 11: 0.0337962589},
    "R3": {9: 0.0039287934},
    "R4": {7: 0.0071919364, 8: 0.0342437519},
}
EXPECTED_FITS = {
    "R1": ((-5.0, -8.6), -0.4273186255),
    "R2": ((-3.5, -9.05), 0.3750543939),
    "R3": ((-2.9, -8.6), 0.4713248264),
    "R4": ((-3.5, -8.15), -0.2452079639),
}


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE_PATH))


@pytest.fixture(scope="module")
def arch_data(example):
    base = example["load_example_19_4"]()
    assembly = base.build_full_arch()
    geometry = example["build_arch_geometry"](assembly)
    return base, assembly, geometry


@pytest.fixture(scope="module")
def admissible_model(example, arch_data):
    base, assembly, geometry = arch_data
    load_node, load_dofs, application_points = base.load_setup(assembly)
    problem = base.hidden_load_problem(assembly, load_node, load_dofs, application_points, penalty=False)
    rbe_boundary = example["solve_primal_rbe_boundary"](base, problem)
    center, family = example["build_admissible_contour"](rbe_boundary, geometry)
    return rbe_boundary, center, family


@pytest.fixture(scope="module")
def case_diagnostics(example, arch_data, admissible_model):
    base, _, geometry = arch_data
    rbe_boundary, _, _ = admissible_model
    return example["supplied_case_diagnostics"](geometry, base.THICKNESS, rbe_boundary)


def test_graphic_statics_weights_are_exact_mesh_weights(example, arch_data):
    _, assembly, geometry = arch_data
    expected = np.asarray(
        [assembly.graph.node_attribute(node, "block").volume() for node in assembly.graph.nodes()],
        dtype=float,
    )
    np.testing.assert_allclose(geometry.weights, expected, atol=1e-12, rtol=0.0)
    assert not np.any(np.isclose(geometry.weights, 0.863938, atol=1e-7, rtol=0.0))


@pytest.mark.parametrize("anchor_load,insertion_fraction", SUPPLIED_CASES)
def test_supplied_cases_use_mesh_weight_force_directions(example, arch_data, anchor_load, insertion_fraction):
    base, _, geometry = arch_data
    repository_load = np.asarray([anchor_load[0], -anchor_load[1]])
    insertion_x = geometry.left_support_x + insertion_fraction * base.THICKNESS
    trace = example["trace_thrust_line"](repository_load, insertion_x, geometry)
    points = example["trace_polyline"](trace)
    diagram = example["force_diagram"](repository_load, geometry.weights)

    for segment, direction in zip(np.diff(points, axis=0), trace.directions):
        assert example["cross_2d"](segment, direction) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(np.diff(diagram.nodes[:, 1]), -geometry.weights, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        np.diff(trace.directions, axis=0),
        np.column_stack([np.zeros(len(geometry.weights)), -geometry.weights]),
        atol=1e-12,
        rtol=0.0,
    )
    assert trace.insertion_x == pytest.approx(geometry.left_support_x + insertion_fraction * base.THICKNESS)


def test_cog_points_are_concurrency_points_not_containment_constraints(example, arch_data):
    base, _, geometry = arch_data
    anchor_load, insertion_fraction = SUPPLIED_CASES[1]
    repository_load = np.asarray([anchor_load[0], -anchor_load[1]])
    trace = example["trace_thrust_line"](
        repository_load,
        geometry.left_support_x + insertion_fraction * base.THICKNESS,
        geometry,
    )

    outside_assigned_block = 0
    for block_index, kink in enumerate(trace.kinks):
        segments = example["block_action_segments"](trace, block_index, len(geometry.block_polygons))
        left_point = segments[0][0]
        right_point = segments[1][1]
        assert example["cross_2d"](kink - left_point, trace.directions[block_index]) == pytest.approx(
            0.0,
            abs=1e-9,
        )
        assert example["cross_2d"](right_point - kink, trace.directions[block_index + 1]) == pytest.approx(
            0.0,
            abs=1e-9,
        )
        assert kink[0] == pytest.approx(geometry.centers[block_index, 0], abs=1e-12)
        np.testing.assert_allclose(
            trace.directions[block_index] + [0.0, -geometry.weights[block_index]],
            trace.directions[block_index + 1],
            atol=1e-12,
            rtol=0.0,
        )
        if 0 < block_index < len(geometry.block_polygons) - 1:
            if example["point_halfspace_violation"](kink, geometry.block_halfspaces[block_index]) > 1e-9:
                outside_assigned_block += 1
    assert outside_assigned_block > 0


def test_supplied_loads_are_rbe_safe_and_friction_admissible(case_diagnostics):
    assert [diagnostic.label for diagnostic in case_diagnostics] == ["R1", "R2", "R3", "R4"]
    assert all(diagnostic.rbe_margin > 0.0 for diagnostic in case_diagnostics)
    assert all(diagnostic.fitted_rbe_margin > 0.0 for diagnostic in case_diagnostics)
    for diagnostic in case_diagnostics:
        assert max(check.friction_utilization for check in diagnostic.supplied_checks) <= 1.0
        assert all(check.normal_component > 0.0 for check in diagnostic.supplied_checks)


def test_exact_supplied_cases_have_only_documented_joint_misses(case_diagnostics):
    for diagnostic in case_diagnostics:
        observed = {
            check.interface_index: check.overrun for check in diagnostic.supplied_checks if check.overrun > 1e-12
        }
        assert observed == pytest.approx(EXPECTED_JOINT_OVERRUNS[diagnostic.label], abs=1e-9)
        assert not all(check.valid for check in diagnostic.supplied_checks)


def test_nearest_fitted_cases_are_joint_admissible(example, arch_data, case_diagnostics):
    _, _, geometry = arch_data
    for diagnostic in case_diagnostics:
        expected_load, expected_insertion = EXPECTED_FITS[diagnostic.label]
        np.testing.assert_allclose(diagnostic.fitted_anchor_load, expected_load, atol=2e-6, rtol=0.0)
        assert diagnostic.fitted_insertion_fraction == pytest.approx(expected_insertion, abs=2e-6)
        assert diagnostic.fitted_interval.feasible
        assert example["joint_violations"](diagnostic.fitted_trace, geometry) == []


def test_fitted_pressure_paths_stay_inside_convex_blocks(example, arch_data, case_diagnostics):
    _, _, geometry = arch_data
    for diagnostic in case_diagnostics:
        for block_index, start, end in example["pressure_path_segments"](
            diagnostic.fitted_trace,
            len(geometry.block_polygons),
        ):
            if block_index in (0, len(geometry.block_polygons) - 1):
                continue
            halfspaces = geometry.block_halfspaces[block_index]
            for point in (start, 0.5 * (start + end), end):
                assert example["point_halfspace_violation"](point, halfspaces) <= 2e-9


def test_supplied_case_diagnostic_figure_has_expected_panels(example, arch_data, case_diagnostics):
    _, _, geometry = arch_data
    figure = example["plot_supplied_case_diagnostics"](case_diagnostics, geometry)
    assert len(figure.axes) == 10
    plt.close(figure)


def test_both_figures_generate_valid_svg(example, arch_data, admissible_model, case_diagnostics):
    _, _, geometry = arch_data
    rbe_boundary, _, family = admissible_model
    representative = example["nearest_admissible_sample"](
        family,
        case_diagnostics[0].fitted_anchor_load,
    )
    figures = (
        example["plot_thrust_family"](rbe_boundary, geometry, family, case_diagnostics, representative),
        example["plot_supplied_case_diagnostics"](case_diagnostics, geometry),
    )
    for figure in figures:
        svg = io.StringIO()
        figure.savefig(svg, format="svg", bbox_inches="tight")
        ET.fromstring(svg.getvalue())
        plt.close(figure)


def test_center_load_has_joint_admissible_insertion(example, arch_data):
    _, _, geometry = arch_data
    center = np.asarray([-4.0, 8.60389558])
    interval = example["insertion_interval"](center, geometry)
    assert interval.feasible

    trace = example["trace_thrust_line"](center, interval.midpoint, geometry)
    assert example["joint_violations"](trace, geometry) == []


def test_maximal_admissible_contour_contains_only_valid_lines(example, arch_data, admissible_model):
    _, _, geometry = arch_data
    _, _, family = admissible_model

    assert len(family) == example["NUM_CONTOUR_DIRECTIONS"]
    assert len(example["plotted_family"](family)) == 90
    for sample in family:
        assert sample.interval.feasible
        assert 0.0 < sample.admissible_radius <= sample.rbe_radius + 1e-12
        assert example["joint_violations"](sample.trace, geometry) == []


def test_pressure_path_boundary_coincides_with_rbe_boundary(admissible_model):
    _, _, family = admissible_model
    np.testing.assert_allclose(
        [sample.admissible_radius for sample in family],
        [sample.rbe_radius for sample in family],
        atol=1e-12,
        rtol=0.0,
    )


def test_maximal_admissible_contour_has_expected_extents(example, admissible_model):
    _, _, family = admissible_model
    anchor_loads = example["anchor_plot_coordinates"]([sample.admissible_load for sample in family])

    assert float(np.min(anchor_loads[:, 0])) == pytest.approx(-5.0373, abs=0.01)
    assert float(np.max(anchor_loads[:, 0])) == pytest.approx(-2.7804, abs=0.01)
    assert float(np.min(anchor_loads[:, 1])) == pytest.approx(-9.1017, abs=0.01)
    assert float(np.max(anchor_loads[:, 1])) == pytest.approx(-8.1061, abs=0.01)
