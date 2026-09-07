import importlib.util
import io
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

EXAMPLE_PATH = Path(__file__).parents[1] / "docs" / "examples" / "19_4_3_rbe_joint_opening_arch.py"


@pytest.fixture(scope="module")
def example():
    specification = importlib.util.spec_from_file_location("example_19_4_3_tests", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results(example):
    return example._analyze(support_directions=72, sample_count=72)


def _cases(result):
    return result.facets + result.vertices + result.samples


def test_reuses_mesh_geometry_weights_and_support_conventions(example, results):
    for result in results:
        original = result.reference
        assert original.self_weight.feasible
        np.testing.assert_array_equal(result.geometry.centers, original.geometry.centers)
        np.testing.assert_array_equal(result.geometry.weights, original.geometry.weights)
        for node, weight in enumerate(result.geometry.weights):
            assert weight == pytest.approx(original.assembly.node_block(node).volume(), abs=1e-12)
        assert original.assembly.graph.node_attribute(0, "is_support")
        assert not original.assembly.graph.node_attribute(19, "is_support")
        assert result.support_gap <= example.GEOMETRIC_TOLERANCE
        assert len(result.samples) == 72
        assert len(result.vertices) == len(result.facets)
        assert len(result.modes) == len(result.facets)


def test_endpoint_names_do_not_depend_on_interface_order(example, results):
    base = example._load_geometry_example()
    for result in results:
        geometry = result.geometry
        flipped = replace(geometry, interfaces=[joint[::-1].copy() for joint in geometry.interfaces])
        np.testing.assert_allclose(example._canonical_geometry(flipped).interfaces, geometry.interfaces, atol=1e-12)
        for joint, parameter in zip(geometry.interfaces, geometry.joint_parameters[1:-1]):
            normal = base.unit_normal(geometry.profile, parameter)
            assert np.dot(joint[1] - joint[0], normal) > 0.0


def test_support_refinement_recovers_facets_missing_from_initial_hull(example, results):
    base = example._load_geometry_example()
    for result in results:
        polygon, _, gap = example._certify_boundary(result.problem, result.polygon[::5])
        assert gap <= 1e-9
        assert base.polygon_area(polygon) == pytest.approx(base.polygon_area(result.polygon), abs=2e-8)
        assert base.polygon_area(result.reference.rbe_boundary) <= base.polygon_area(result.polygon) + 1e-9
        dense_reference = base.load_example_19_4_1().solve_primal_rbe_boundary(
            base.load_example_19_4(), result.problem, num_directions=360
        )
        assert base.polygon_area(result.polygon) == pytest.approx(base.polygon_area(dense_reference), abs=2e-8)


def test_all_cases_have_compressive_friction_admissible_joint_resultants(example, results):
    base = example._load_geometry_example()
    for result in results:
        for case in _cases(result):
            assert case.interval.feasible
            assert base.trace_is_joint_admissible(case.trace, result.geometry)
            assert np.min(case.parameters) >= -1e-9
            assert np.max(case.parameters) <= 1.0 + 1e-9
            assert np.max(case.friction) < 1.0
            totals = case.normal_forces.sum(axis=1)
            assert np.min(totals) > 0.0
            np.testing.assert_allclose(case.normal_forces[:, 1] / totals, case.parameters, atol=1e-12)
            np.testing.assert_allclose(
                case.pressures,
                (1.0 - case.parameters[:, None]) * np.asarray(result.geometry.interfaces)[:, 0]
                + case.parameters[:, None] * np.asarray(result.geometry.interfaces)[:, 1],
                atol=1e-10,
            )


def test_free_blocks_satisfy_force_and_moment_equilibrium(example, results):
    base = example._load_geometry_example()
    for result in results:
        geometry = result.geometry
        for case in _cases(result):
            forces = case.trace.directions[1:][::-1]
            for block in range(1, 20):
                center = geometry.centers[block]
                left_force = forces[block - 1]
                left_point = case.pressures[block - 1]
                right_force = -forces[block] if block < 19 else case.load
                right_point = case.pressures[block] if block < 19 else case.trace.right_load_point
                residual = left_force + right_force + [0.0, -geometry.weights[block]]
                moment = base.cross_2d(left_point - center, left_force)
                moment += base.cross_2d(right_point - center, right_force)
                np.testing.assert_allclose(residual, 0.0, atol=1e-11)
                assert abs(moment) <= 1e-9


def test_inward_feasibility_outward_failure_and_work_sign(example, results):
    base = example._load_geometry_example().load_example_19_4()
    for result in results:
        for case in result.facets:
            direction = case.load - result.reference.rbe_center
            inward, outward = case.load - 1e-4 * direction, case.load + 1e-4 * direction
            assert base.solve_fixed_visible_load(result.problem, inward).success
            assert base.solve_fixed_visible_load(result.problem, outward).status == 2
            for mode_id in case.mode_ids:
                motion = result.modes[mode_id].velocity
                assert float(direction.dot(motion[-1, :2])) > 0.0


def test_all_attached_modes_satisfy_compatibility_complementarity_and_work(example, results):
    for result in results:
        for case in _cases(result):
            assert case.mode_ids, (case.label, case.status)
            for mode_id in case.mode_ids:
                mode = result.modes[mode_id]
                valid, gaps, metrics = example._mode_metrics(case, mode.velocity, result)
                assert valid, metrics
                assert np.min(gaps) >= -1e-9
                assert abs(metrics["boundary_work"]) <= 1e-8
                assert metrics["outward_work"] > 0.0
                assert metrics["stick_residual"] <= 1e-9
                assert metrics["holder_residual"] <= 1e-9
                np.testing.assert_array_equal(mode.openings, gaps > example.ACTIVITY_TOLERANCE)
                assert np.any(mode.openings)


def test_pivot_is_opposite_opening_side_and_zero_force_does_not_imply_open(example, results):
    stationary_unloaded = 0
    for result in results:
        for case in _cases(result):
            for mode_id in case.mode_ids:
                mode = result.modes[mode_id]
                np.testing.assert_array_equal(mode.pivots, mode.openings[:, ::-1] & ~case.candidates)
                assert not np.any(mode.openings & mode.pivots)
                assert np.all(case.candidates[mode.openings])
                stationary_unloaded += np.sum(case.candidates & ~mode.openings)
    assert stationary_unloaded > 0


def test_polygon_vertices_retain_distinct_adjacent_branches(example, results):
    for result in results:
        assert all(len(case.mode_ids) == 2 for case in result.vertices)
        for case in result.vertices:
            first, second = [result.modes[index] for index in case.mode_ids]
            assert not np.allclose(first.velocity, second.velocity)
            assert case.status == "multiple admissible opening branches"


def test_catenary_coverage_and_non_crown_circular_openings(example, results):
    circular_candidates, circular_openings = example._coverage(results[0])
    catenary_candidates, catenary_openings = example._coverage(results[1])
    assert np.any(circular_candidates, axis=1).sum() == 17
    assert np.any(circular_openings, axis=1).sum() == 17
    assert np.any(circular_openings[[0, 3, 15, 18]])
    assert np.all(catenary_candidates)
    assert np.all(catenary_openings)


def test_rigid_preview_preserves_pivots_and_has_no_interpenetration(example, results):
    for result in results:
        geometry = result.geometry
        for mode in result.modes:
            assert mode.preview_scale > 0.0
            assert mode.preview_scale * np.max(np.abs(np.diff(mode.velocity[:, 2]))) <= np.deg2rad(10) + 1e-12
            for amplitude in (0.0, 0.25, 0.5, 1.0):
                polygons, rotations, translations = example._preview_pose(geometry, mode, amplitude)
                assert not example._polygons_overlap(polygons)
                np.testing.assert_allclose(polygons[0], geometry.block_polygons[0], atol=1e-12)
                np.testing.assert_allclose(rotations[-1], np.eye(2), atol=1e-12)
                for old, new in zip(geometry.block_polygons, polygons):
                    np.testing.assert_allclose(
                        np.linalg.norm(np.roll(old, -1, axis=0) - old, axis=1),
                        np.linalg.norm(np.roll(new, -1, axis=0) - new, axis=1),
                        atol=1e-12,
                    )
                for joint_index, joint in enumerate(geometry.interfaces):
                    pivot = joint[mode.pivot_indices[joint_index]]
                    left = rotations[joint_index].dot(pivot) + translations[joint_index]
                    right = rotations[joint_index + 1].dot(pivot) + translations[joint_index + 1]
                    np.testing.assert_allclose(left, right, atol=1e-12)


def test_preview_derivative_matches_verified_initial_velocity(example, results):
    for result in results:
        geometry = result.geometry
        for mode in result.modes:
            epsilon = 1e-7
            _, rotations, shifts = example._preview_pose(geometry, mode, epsilon / mode.preview_scale)
            centers = np.asarray([r.dot(c) + s for r, c, s in zip(rotations, geometry.centers, shifts)])
            np.testing.assert_allclose((centers - geometry.centers) / epsilon, mode.velocity[:, :2], atol=2e-7)


def test_enlarged_preview_is_five_times_the_original_range(example, results):
    assert example.PREVIEW_DEGREES == 10.0
    assert example.PREVIEW_DEGREES / (example.PREVIEW_STEPS - 1) <= 0.1
    for result in results:
        reached_cap = []
        for mode in result.modes:
            angle = mode.preview_scale * np.max(np.abs(np.diff(mode.velocity[:, 2])))
            reached_cap.append(np.isclose(angle, np.deg2rad(10), atol=1e-10))
        assert any(reached_cap)
        mode = result.modes[reached_cap.index(True)]
        original = np.vstack(result.geometry.block_polygons)
        small = np.vstack(example._preview_pose(result.geometry, mode, 0.2)[0])
        large = np.vstack(example._preview_pose(result.geometry, mode, 1.0)[0])
        assert np.max(np.linalg.norm(large - original, axis=1)) > 4.5 * np.max(
            np.linalg.norm(small - original, axis=1)
        )


def test_overlap_and_preview_reduction_paths(example, results):
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert example._polygons_overlap([square, square + [0.5, 0.5]])
    assert not example._polygons_overlap([square, square + [1.0, 0.0]])
    assert not example._polygons_overlap([square, square + [2.0, 0.0]])
    mode = replace(results[0].modes[0], velocity=-results[0].modes[0].velocity)
    scale, _ = example._preview_limit(mode, results[0].geometry)
    assert scale == 0.0


def test_friction_limit_and_no_mode_are_not_presented_as_opening(example, results):
    result = results[0]
    limit = replace(result.facets[0], friction=np.ones(19), mode_ids=[])
    assert example._solve_mode(limit, result) is None
    assert "friction limit" in limit.status
    closed = replace(result.facets[0], candidates=np.zeros((19, 2), dtype=bool), mode_ids=[])
    assert example._solve_mode(closed, result) is None
    assert "no compatible" in closed.status


def test_viewer_orders_and_deduplicates_loads_without_losing_vertex_branches(example, results):
    for result in results:
        # A ray sample at a vertex must not replace the certified two-branch case.
        duplicate = replace(result.vertices[0], label="duplicate", kind="sample", mode_ids=[])
        augmented = replace(result, samples=result.samples + [duplicate])
        cases = example._viewer_cases(augmented)
        assert len(cases) == len(example._viewer_cases(result))
        angles = [np.arctan2(*(case.load - result.reference.rbe_center)[::-1]) % (2 * np.pi) for case in cases]
        assert np.all(np.diff(angles) >= 0.0)
        for source in _cases(result):
            matching = [case for case in cases if np.max(np.abs(case.load - source.load)) <= 1e-9]
            assert len(matching) == 1
            if source.kind == "vertex":
                assert matching[0] is source
                assert len(matching[0].mode_ids) == 2


def test_viewer_paths_and_construction_match_the_same_mechanical_case(example, results):
    base = example._load_geometry_example()
    for item, result in zip(example._viewer_data(results), results):
        sources = {case.label: case for case in _cases(result)}
        assert len({case["id"] for case in item["cases"]}) == len(item["cases"])
        for case in item["cases"]:
            source = sources[case["id"]]
            np.testing.assert_array_equal(case["load"], source.load)
            np.testing.assert_array_equal(case["pressure"], source.pressures)
            np.testing.assert_array_equal(case["construction"], base.trace_polyline(source.trace))
            np.testing.assert_array_equal(case["kinks"], source.trace.kinks)
            np.testing.assert_array_equal(
                case["extensions"][0], [source.trace.left_support_extension, source.pressures[0]]
            )
            np.testing.assert_array_equal(case["extensions"][1], [source.pressures[-1], source.trace.right_load_point])
            assert case["insertionInterval"] == [source.interval.lower, source.interval.upper]
            assert case["insertion"] == source.interval.midpoint
            # Every connecting segment stays in its convex block, including its interior.
            for block in range(1, 19):
                polygon = result.geometry.block_polygons[block]
                edges = np.roll(polygon, -1, axis=0) - polygon
                for fraction in (0.0, 0.5, 1.0):
                    point = (1 - fraction) * source.pressures[block - 1] + fraction * source.pressures[block]
                    signs = [base.cross_2d(edge, point - start) for edge, start in zip(edges, polygon)]
                    assert min(signs) >= -1e-9 or max(signs) <= 1e-9


def test_svg_and_offline_html_with_unavailable_motion(example, results):
    def unavailable(case):
        return replace(case, mode_ids=[], status="no verified opening mode; contact limits only")

    empty = replace(
        results[0],
        facets=[unavailable(case) for case in results[0].facets],
        vertices=[unavailable(case) for case in results[0].vertices],
        samples=[unavailable(case) for case in results[0].samples],
        modes=[],
    )
    assert np.all(example._opening_codes(empty, empty.samples) == -1)
    for inputs in (results, [empty, results[1]]):
        figure = example._plot_overview(inputs)
        stream = io.StringIO()
        figure.savefig(stream, format="svg", bbox_inches="tight")
        assert ET.fromstring(stream.getvalue()).tag.endswith("svg")
        plt.close(figure)
        html = example._viewer_html(inputs)
        assert "__OPENING_DATA__" not in html
        assert "fetch(" not in html
        assert 'src="http' not in html
        start = html.index('<script id="opening-data" type="application/json">')
        content = html[html.index(">", start) + 1 : html.index("</script>", start)]
        data = json.loads(content)
        assert len(data) == 2
        assert len(data[0]["cases"]) == len(example._viewer_cases(inputs[0]))
        for item in data:
            assert all(mode["maxRelativeDegrees"] <= 10.0 + 1e-10 for mode in item["modes"])
            assert all(all(0 <= index < len(item["modes"]) for index in case["modes"]) for case in item["cases"])
