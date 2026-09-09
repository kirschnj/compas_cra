"""Map finite-joint opening and verified onset mechanisms for two arch profiles.

This is a rigid-block, initial-motion diagnostic, not a dynamic collapse or
intra-block fracture simulation. The right holder supplies a reaction moment.
"""

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

GEOMETRIC_TOLERANCE = 1e-9
ACTIVITY_TOLERANCE = 1e-7
WORK_TOLERANCE = 1e-8
RANK_TOLERANCE = 1e-10
NUM_DIRECTIONS = 360
PREVIEW_DEGREES = 10.0
PREVIEW_STEPS = 101
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
OUTPUT_HTML = Path(__file__).with_suffix(".html")
VIEWER_TEMPLATE = Path(__file__).with_name("19_4_3_joint_opening_viewer.html.template")
OPENING_COLORS = ("#ECEFF1", "#0072B2", "#D55E00", "#8E44AD")


@dataclass
class _Case:
    """One actual right-end load, its contact state, and verified mode IDs."""

    label: str
    kind: str
    load: np.ndarray
    outward: np.ndarray
    interval: object
    trace: object
    parameters: np.ndarray
    pressures: np.ndarray
    normal_forces: np.ndarray
    friction: np.ndarray
    candidates: np.ndarray
    mode_ids: list = field(default_factory=list)
    status: str = "awaiting compatibility check"


@dataclass
class _Mode:
    """A verified normalized velocity and a bounded geometric illustration."""

    label: str
    velocity: np.ndarray
    gaps: np.ndarray
    openings: np.ndarray
    pivots: np.ndarray
    pivot_indices: np.ndarray
    nullity: int
    metrics: dict
    preview_scale: float = 0.0
    preview_status: str = "unavailable"


@dataclass
class _OpeningResult:
    """Local analysis data; no package-level interface is introduced."""

    reference: object
    geometry: object
    problem: object
    polygon: np.ndarray
    normals: np.ndarray
    support_gap: float
    rows: np.ndarray
    contact_normals: np.ndarray
    facets: list = field(default_factory=list)
    vertices: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    modes: list = field(default_factory=list)


def _load_geometry_example():
    """Import the unchanged 19_4_2 example using its existing import pattern."""
    name = "example_19_4_2_for_opening"
    if name not in sys.modules:
        path = Path(__file__).with_name("19_4_2_rbe_thrust_line_geometry_comparison.py")
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    return sys.modules[name]


def _canonical_geometry(geometry):
    """Order every joint from intrados to extrados using the profile normal."""
    example = _load_geometry_example()
    interfaces = []
    for joint, parameter in zip(geometry.interfaces, geometry.joint_parameters[1:-1]):
        normal = example.unit_normal(geometry.profile, parameter)
        interfaces.append(np.asarray(joint)[np.argsort(np.asarray(joint).dot(normal))].copy())
    return replace(geometry, interfaces=interfaces)


def _support_point(problem, direction):
    """Solve the original tied contact/application-force LP, without relaxation."""
    objective = -np.asarray(problem.load_projection.T.dot(direction)).ravel()
    solution = linprog(
        objective,
        A_ub=problem.inequalities,
        b_ub=problem.inequality_rhs,
        A_eq=problem.equilibrium,
        b_eq=-problem.baseline_load,
        bounds=[(None, None)] * len(objective),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError("Opening-boundary support solve failed: " + solution.message)
    return np.asarray(problem.load_projection.dot(solution.x)).ravel()


def _certify_boundary(problem, initial_polygon):
    """Refine and certify all facets, including those missed by angular sampling."""
    points = np.asarray(initial_polygon, dtype=float)
    for _ in range(40):
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        additions = []
        gaps = []
        for point, normal in zip(polygon, normals):
            support = _support_point(problem, normal)
            gap = float(normal.dot(support - point))
            gaps.append(gap)
            if gap > GEOMETRIC_TOLERANCE:
                additions.append(support)
        if not additions:
            return polygon, normals, max(0.0, max(gaps))
        points = np.vstack([polygon, additions])
    raise RuntimeError("Could not certify every opening-region facet within 40 refinements.")


def _kinematic_rows(geometry):
    """Return relative endpoint velocity matrices and left-to-right normals.

    Planar rotation is positive counterclockwise in the x-z drawing. A positive
    normal relative velocity separates the right block from the left block.
    """
    count = len(geometry.centers)
    rows = np.zeros((count - 1, 2, 2, 3 * (count - 1)))
    normals = []
    for joint_index, joint in enumerate(geometry.interfaces):
        tangent = joint[1] - joint[0]
        normal = np.asarray([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
        if normal.dot(geometry.centers[joint_index + 1] - geometry.centers[joint_index]) < 0:
            normal *= -1.0
        normals.append(normal)
        for endpoint, point in enumerate(joint):
            for block, sign in ((joint_index, -1.0), (joint_index + 1, 1.0)):
                if block == 0:
                    continue
                offset = point - geometry.centers[block]
                column = 3 * (block - 1)
                rows[joint_index, endpoint, :, column : column + 2] = sign * np.eye(2)
                rows[joint_index, endpoint, :, column + 2] = sign * np.asarray([-offset[1], offset[0]])
    return rows, np.asarray(normals)


def _contact_case(label, kind, load, outward, geometry):
    """Reconstruct the exact load; activity classification never changes it."""
    example = _load_geometry_example()
    interval = example.insertion_interval(load, geometry)
    if not interval.feasible:
        raise ValueError("Certified boundary case has no admissible insertion: " + label)
    trace = example.trace_thrust_line(load, interval.midpoint, geometry)
    checks = sorted(example.joint_checks(trace, geometry), key=lambda check: check.interface_index)
    if not all(check.valid for check in checks):
        raise ValueError("Certified boundary case failed contact checks: " + label)
    parameters = np.asarray([check.parameter for check in checks])
    fractions = np.column_stack([1.0 - parameters, parameters])
    normals = np.asarray([check.normal_component for check in checks])
    return _Case(
        label,
        kind,
        np.asarray(load),
        np.asarray(outward) / np.linalg.norm(outward),
        interval,
        trace,
        parameters,
        trace.crossings[::-1].copy(),
        normals[:, None] * fractions,
        np.asarray([check.friction_utilization for check in checks]),
        fractions <= ACTIVITY_TOLERANCE,
    )


def _mode_metrics(case, velocity, result):
    """Check kinematics, complementarity, and work at a particular load."""
    relative = result.rows.dot(velocity[1:].ravel())
    gaps = np.einsum("jek,jk->je", relative, result.contact_normals)
    loaded = ~case.candidates
    stick = float(np.max(np.abs(relative[loaded]))) if np.any(loaded) else 0.0
    gravity_work = -float(result.geometry.weights.dot(velocity[:, 1]))
    boundary_work = gravity_work + float(case.load.dot(velocity[-1, :2]))
    outward_work = float(case.outward.dot(velocity[-1, :2]))
    metrics = {
        "stick_residual": stick,
        "negative_gap": max(0.0, -float(np.min(gaps))),
        "boundary_work": boundary_work,
        "outward_work": outward_work,
        "complementarity": float(np.max(np.abs(case.normal_forces * gaps))),
        "holder_residual": max(float(np.max(np.abs(velocity[0]))), abs(float(velocity[-1, 2]))),
    }
    valid = (
        stick <= GEOMETRIC_TOLERANCE
        and metrics["negative_gap"] <= GEOMETRIC_TOLERANCE
        and abs(boundary_work) <= WORK_TOLERANCE
        and metrics["complementarity"] <= WORK_TOLERANCE
        and outward_work > GEOMETRIC_TOLERANCE
        and metrics["holder_residual"] <= GEOMETRIC_TOLERANCE
    )
    return valid, gaps, metrics


def _solve_mode(case, result):
    """Find one no-sliding opening witness; do not use an associative friction dual."""
    if np.max(case.friction) >= 1.0 - ACTIVITY_TOLERANCE:
        case.status = "friction limit: opening-only preview suppressed"
        return None
    sticking = result.rows[~case.candidates].reshape((-1, result.rows.shape[-1]))
    holder = np.zeros((1, result.rows.shape[-1]))
    holder[0, -1] = 1.0
    # NumPy's SVD avoids a second OpenMP runtime when optional PyTorch tests
    # have already loaded their native libraries. The relative rank test is
    # identical to scipy.linalg.null_space(..., rcond=RANK_TOLERANCE).
    constraints = np.vstack([sticking, holder])
    _, singular_values, vectors = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular_values > RANK_TOLERANCE * singular_values[0]))
    basis = vectors[rank:].T.copy()
    if not basis.shape[1]:
        case.status = "no compatible opening mode"
        return None
    gap_rows = np.einsum("jekl,jk->jel", result.rows, result.contact_normals).reshape((-1, holder.shape[1]))
    outward = np.zeros(holder.shape[1])
    outward[-3:-1] = case.outward
    solution = linprog(
        np.zeros(basis.shape[1]),
        A_ub=-gap_rows.dot(basis),
        b_ub=np.zeros(len(gap_rows)),
        A_eq=outward.dot(basis)[None, :],
        b_eq=[1.0],
        bounds=[(None, None)] * basis.shape[1],
        method="highs",
    )
    if not solution.success:
        case.status = "no opening direction driven by the outward load"
        return None
    velocity = np.vstack([np.zeros(3), basis.dot(solution.x).reshape((-1, 3))])
    speeds = []
    for polygon, center, motion in zip(result.geometry.block_polygons, result.geometry.centers, velocity):
        offsets = polygon - center
        point_velocities = motion[:2] + motion[2] * np.column_stack([-offsets[:, 1], offsets[:, 0]])
        speeds.extend(np.linalg.norm(point_velocities, axis=1))
    velocity /= max(speeds)
    valid, gaps, metrics = _mode_metrics(case, velocity, result)
    if not valid:
        case.status = "opening witness failed compatibility/work checks"
        return None
    openings = gaps > ACTIVITY_TOLERANCE
    pivots = openings[:, ::-1] & ~case.candidates
    mode = _Mode(
        case.label,
        velocity,
        gaps,
        openings,
        pivots,
        np.argmax(case.normal_forces, axis=1),
        basis.shape[1],
        metrics,
    )
    mode.preview_scale, mode.preview_status = _preview_limit(mode, result.geometry)
    return mode


def _rotation(angle):
    """Return a planar rigid rotation."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, -sine], [sine, cosine]])


def _preview_pose(geometry, mode, amplitude):
    """Preserve exact hinge coincidence and block rigidity in a geometric preview.

    Rotations follow the initial angular-velocity ratios. These finite poses
    are NOT claimed to satisfy equilibrium at the original applied forces.
    """
    angles = mode.preview_scale * float(amplitude) * mode.velocity[:, 2]
    rotations = [_rotation(angle) for angle in angles]
    translations = [np.zeros(2)]
    for block in range(1, len(geometry.centers)):
        pivot = geometry.interfaces[block - 1][mode.pivot_indices[block - 1]]
        translations.append(rotations[block - 1].dot(pivot) + translations[-1] - rotations[block].dot(pivot))
    polygons = [
        polygon.dot(rotation.T) + translation
        for polygon, rotation, translation in zip(geometry.block_polygons, rotations, translations)
    ]
    return polygons, rotations, np.asarray(translations)


def _polygons_overlap(polygons):
    """Detect material interpenetration by broad-phase boxes and convex SAT."""
    lower = np.asarray([polygon.min(axis=0) for polygon in polygons])
    upper = np.asarray([polygon.max(axis=0) for polygon in polygons])
    for first in range(len(polygons)):
        for second in range(first + 1, len(polygons)):
            if np.any(np.minimum(upper[first], upper[second]) - np.maximum(lower[first], lower[second]) <= 0.0):
                continue
            separated = False
            for polygon in (polygons[first], polygons[second]):
                edges = np.roll(polygon, -1, axis=0) - polygon
                axes = np.column_stack([-edges[:, 1], edges[:, 0]])
                axes /= np.linalg.norm(axes, axis=1)[:, None]
                a, b = polygons[first].dot(axes.T), polygons[second].dot(axes.T)
                overlaps = np.minimum(a.max(axis=0), b.max(axis=0)) - np.maximum(a.min(axis=0), b.min(axis=0))
                if np.any(overlaps <= GEOMETRIC_TOLERANCE):
                    separated = True
                    break
            if not separated:
                return True
    return False


def _preview_limit(mode, geometry):
    """Reduce the enlarged preview until all sampled poses are nonpenetrating."""
    angular_rate = float(np.max(np.abs(np.diff(mode.velocity[:, 2]))))
    if angular_rate <= RANK_TOLERANCE:
        return 0.0, "no rotational opening preview"
    scale = math.radians(PREVIEW_DEGREES) / angular_rate
    for _ in range(18):
        mode.preview_scale = scale
        if all(
            not _polygons_overlap(_preview_pose(geometry, mode, amplitude)[0])
            for amplitude in np.linspace(0.0, 1.0, PREVIEW_STEPS)
        ):
            return scale, "verified geometric preview"
        scale *= 0.5
    return 0.0, "geometric preview suppressed: interpenetration"


def _attach_modes(case, mode_ids, result):
    """Keep all verified adjacent-facet branches, including at degenerate vertices."""
    if np.max(case.friction) >= 1.0 - ACTIVITY_TOLERANCE:
        case.status = "friction limit: opening-only preview suppressed"
        return
    for mode_id in mode_ids:
        if mode_id is not None and _mode_metrics(case, result.modes[mode_id].velocity, result)[0]:
            if mode_id not in case.mode_ids:
                case.mode_ids.append(mode_id)
    if case.mode_ids:
        case.status = "verified opening mode" if len(case.mode_ids) == 1 else "multiple admissible opening branches"
    else:
        case.status = "no verified opening mode; contact limits only"


def _analyze_geometry(spec, support_directions=NUM_DIRECTIONS, sample_count=NUM_DIRECTIONS):
    """Analyze all facets/vertices, then associate an ordered load sweep."""
    example = _load_geometry_example()
    base = example.load_example_19_4()
    reference = example.analyze_geometry(spec, support_directions, sample_count)
    if reference.rbe_boundary is None:
        raise ValueError("Opening example requires a bounded load region: " + reference.right_load_status)
    geometry = _canonical_geometry(reference.geometry)
    last = len(geometry.centers) - 1
    points = {last: example.ground_application_points(reference.assembly.node_block(last))}
    problem = base.hidden_load_problem(reference.assembly, last, [(last, "fx"), (last, "fz")], points, False)
    polygon, normals, gap = _certify_boundary(problem, reference.rbe_boundary)
    rows, contact_normals = _kinematic_rows(geometry)
    result = _OpeningResult(reference, geometry, problem, polygon, normals, gap, rows, contact_normals)
    facet_mode_ids = []
    for index, (start, end, normal) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0), normals)):
        case = _contact_case("F{:02d}".format(index), "facet", (start + end) * 0.5, normal, geometry)
        mode = _solve_mode(case, result)
        mode_id = None
        if mode is not None:
            mode_id = len(result.modes)
            result.modes.append(mode)
            _attach_modes(case, [mode_id], result)
        facet_mode_ids.append(mode_id)
        result.facets.append(case)
    for index, point in enumerate(polygon):
        outward = normals[index - 1] + normals[index]
        case = _contact_case("V{:02d}".format(index), "vertex", point, outward, geometry)
        _attach_modes(case, [facet_mode_ids[index - 1], facet_mode_ids[index]], result)
        result.vertices.append(case)
    halfspaces = np.column_stack([normals, -np.einsum("ij,ij->i", polygon, normals)])
    center = reference.rbe_center
    for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)):
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        radius = example.rbe_ray_radius(center, direction, halfspaces)
        point = center + radius * direction
        case = _contact_case("S{:03d}".format(index), "sweep", point, direction, geometry)
        residuals = np.abs(normals.dot(point) + halfspaces[:, 2])
        active_facets = np.flatnonzero(residuals <= GEOMETRIC_TOLERANCE)
        _attach_modes(case, [facet_mode_ids[facet] for facet in active_facets], result)
        result.samples.append(case)
    return result


def _analyze(support_directions=NUM_DIRECTIONS, sample_count=NUM_DIRECTIONS):
    """Compare the reference circle and symmetric catenary only."""
    example = _load_geometry_example()
    return [_analyze_geometry(spec, support_directions, sample_count) for spec in example.PROFILE_SPECS[:2]]


def _coverage(result):
    """Return candidate opening ends and witnessed opening ends across ALL cases."""
    candidates = np.zeros((len(result.geometry.interfaces), 2), dtype=bool)
    openings = np.zeros_like(candidates)
    for case in result.facets + result.vertices + result.samples:
        candidates |= case.candidates
        for mode_id in case.mode_ids:
            openings |= result.modes[mode_id].openings
    return candidates, openings


def _opening_codes(result, cases):
    """Encode union of verified branches, not frequencies or probabilities."""
    codes = []
    for case in cases:
        if not case.mode_ids:
            codes.append(np.full(len(result.geometry.interfaces), -1, dtype=int))
            continue
        opened = np.zeros((len(result.geometry.interfaces), 2), dtype=bool)
        for mode_id in case.mode_ids:
            opened |= result.modes[mode_id].openings
        codes.append(opened[:, 0].astype(int) + 2 * opened[:, 1].astype(int))
    return np.asarray(codes).T


def _representative_case(result):
    """Choose a nondegenerate minimum-Fx facet midpoint deterministically."""
    return min(result.facets, key=lambda case: (not bool(case.mode_ids), case.load[0], case.label))


def _plot_arch(ax, result, case):
    """Show a selected mechanism without misleading funicular extensions."""
    geometry = result.geometry
    display_polygons = list(geometry.block_polygons)
    for index, polygon in enumerate(geometry.block_polygons):
        ax.add_patch(PolygonPatch(polygon, facecolor="#EEEEEE", edgecolor="#777777", linewidth=0.6))
        ax.text(*geometry.centers[index], str(index), ha="center", va="center", fontsize=7, color="#555555")
    if case.mode_ids:
        mode = result.modes[case.mode_ids[0]]
        if mode.preview_scale:
            preview_polygons = _preview_pose(geometry, mode, 1.0)[0]
            display_polygons.extend(preview_polygons)
            for polygon in preview_polygons:
                ax.add_patch(PolygonPatch(polygon, fill=False, edgecolor="#009E73", linewidth=0.8, alpha=0.8))
        for joint, openings, pivots in zip(geometry.interfaces, mode.openings, mode.pivots):
            for side in range(2):
                if openings[side]:
                    ax.scatter(*joint[side], s=40, marker="x", color=OPENING_COLORS[side + 1], zorder=6)
                if pivots[side]:
                    ax.scatter(*joint[side], s=30, facecolors="none", edgecolors="#111111", zorder=6)
    ax.scatter(case.pressures[:, 0], case.pressures[:, 1], marker="D", s=8, color="#555555", zorder=5)
    ax.text(-6.6, -0.65, "Block 0 fixed", ha="left", fontsize=8)
    ax.text(6.6, -0.65, "19: rotation held", ha="right", fontsize=8)
    points = np.vstack(display_polygons)
    lower, upper = points.min(axis=0), points.max(axis=0)
    ax.set(
        xlim=(min(-6.8, lower[0] - 0.4), max(7.0, upper[0] + 0.4)),
        ylim=(min(-1.0, lower[1] - 0.4), max(6.6, upper[1] + 0.4)),
        aspect="equal",
        xlabel="x",
        ylabel="z",
    )
    ax.set_title("{}: onset preview (not a collapse trajectory)".format(case.label), fontsize=10)


def _plot_overview(results):
    """Create paired load maps, representative mechanisms, and opening matrices."""
    figure = plt.figure(figsize=(16, 14), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=[1.4, 1.3, 1.4, 1.3], width_ratios=[1.0, 2.1])
    cmap = ListedColormap(("#B4B9BC",) + OPENING_COLORS)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    for row, result in enumerate(results):
        case = _representative_case(result)
        load_ax = figure.add_subplot(grid[2 * row, 0])
        load_ax.add_patch(PolygonPatch(result.polygon, facecolor="#DEE9ED", edgecolor="#444444", linewidth=1.2))
        load_ax.scatter(*case.load, color="#009E73", s=55, zorder=5)
        load_ax.annotate(case.label, case.load, xytext=(8, 8), textcoords="offset points", fontsize=9)
        load_ax.autoscale_view()
        load_ax.margins(0.12)
        load_ax.set(xlabel="Right-end Fx", ylabel="Right-end Fz (upward positive)")
        candidate, opening = _coverage(result)
        load_ax.set_title(
            "{}\nCandidate joints: {}/19; opening witnesses: {}/19".format(
                result.reference.spec.label, np.any(candidate, axis=1).sum(), np.any(opening, axis=1).sum()
            ),
            fontsize=11,
        )
        _plot_arch(figure.add_subplot(grid[2 * row, 1]), result, case)
        map_ax = figure.add_subplot(grid[2 * row + 1, :])
        map_ax.imshow(
            _opening_codes(result, result.samples),
            origin="lower",
            aspect="auto",
            extent=(-0.5, len(result.samples) - 0.5, -0.5, 18.5),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        map_ax.set_yticks(np.arange(19))
        map_ax.tick_params(axis="y", labelsize=8)
        map_ax.set_xticks(np.linspace(0, len(result.samples), 5)[:-1])
        map_ax.set_xticklabels(["0", "90", "180", "270"])
        map_ax.set(xlabel="Load-ray angle from the feasible center (degrees)", ylabel="Joint j: blocks j / j+1")
        map_ax.set_title(
            "Verified opening sides across different loads; NOT simultaneous cracks or probabilities", fontsize=10
        )
    legend = [
        Line2D([0], [0], color=color, linewidth=7, label=label)
        for color, label in zip(
            OPENING_COLORS, ["Closed in shown modes", "Intrados opens", "Extrados opens", "Either end across branches"]
        )
    ]
    legend.extend(
        [
            Line2D([0], [0], color="#B4B9BC", linewidth=7, label="No verified mode"),
            Line2D(
                [0], [0], marker="o", markerfacecolor="none", color="#111111", linestyle="none", label="Retained pivot"
            ),
            Line2D([0], [0], color="#009E73", label="Small rigid-motion illustration"),
        ]
    )
    figure.legend(handles=legend, loc="outside lower center", ncol=3, fontsize=9)
    figure.suptitle("19-4-3 | Joint opening: circular versus symmetric catenary", fontsize=16)
    return figure


def _viewer_cases(result):
    """Order unique loads around the boundary, preferring certified vertices.

    Deduplication affects presentation only. Keeping vertices before sampled
    loads preserves both adjacent-facet branches at a coincident boundary point.
    """
    cases = []
    for case in result.vertices + result.facets + result.samples:
        if not any(np.max(np.abs(case.load - other.load)) <= GEOMETRIC_TOLERANCE for other in cases):
            cases.append(case)
    center = result.reference.rbe_center
    return sorted(cases, key=lambda case: math.atan2(*(case.load - center)[::-1]) % (2.0 * math.pi))


def _viewer_data(results):
    """Serialize compact geometry and mode data; evaluate rigid poses in the viewer."""
    payload = []
    for result in results:
        cases = _viewer_cases(result)
        candidate, openings = _coverage(result)
        payload.append(
            {
                "label": result.reference.spec.label,
                "polygons": [polygon.tolist() for polygon in result.geometry.block_polygons],
                "centers": result.geometry.centers.tolist(),
                "joints": np.asarray(result.geometry.interfaces).tolist(),
                "region": result.polygon.tolist(),
                "center": result.reference.rbe_center.tolist(),
                "candidateCoverage": candidate.tolist(),
                "openingCoverage": openings.tolist(),
                "selfWeight": bool(result.reference.self_weight.feasible),
                "sampleCount": len(result.samples),
                "cases": [
                    {
                        "id": case.label,
                        "label": case.label,
                        "kind": case.kind,
                        "load": case.load.tolist(),
                        "angle": math.atan2(*(case.load - result.reference.rbe_center)[::-1]) % (2.0 * math.pi),
                        "insertion": case.interval.midpoint,
                        "insertionInterval": [case.interval.lower, case.interval.upper],
                        "pressure": case.pressures.tolist(),
                        "construction": _load_geometry_example().trace_polyline(case.trace).tolist(),
                        "kinks": case.trace.kinks.tolist(),
                        "extensions": [
                            [case.trace.left_support_extension.tolist(), case.pressures[0].tolist()],
                            [case.pressures[-1].tolist(), case.trace.right_load_point.tolist()],
                        ],
                        "parameters": case.parameters.tolist(),
                        "forces": case.normal_forces.tolist(),
                        "friction": case.friction.tolist(),
                        "candidates": case.candidates.tolist(),
                        "modes": case.mode_ids,
                        "status": case.status,
                    }
                    for case in cases
                ],
                "modes": [
                    {
                        "label": mode.label,
                        "velocity": mode.velocity.tolist(),
                        "scale": mode.preview_scale,
                        "maxRelativeDegrees": float(
                            math.degrees(mode.preview_scale * np.max(np.abs(np.diff(mode.velocity[:, 2]))))
                        ),
                        "gaps": mode.gaps.tolist(),
                        "openings": mode.openings.tolist(),
                        "pivots": mode.pivots.tolist(),
                        "pins": mode.pivot_indices.tolist(),
                        "previewStatus": mode.preview_status,
                        "metrics": mode.metrics,
                    }
                    for mode in result.modes
                ],
            }
        )
    return payload


def _viewer_html(results):
    """Produce a standalone, offline viewer with no external dependencies."""
    data = json.dumps(_viewer_data(results), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return VIEWER_TEMPLATE.read_text(encoding="utf-8").replace("__OPENING_DATA__", data)


def main():
    """Generate both diagnostics and print auditable coverage/work summaries."""
    results = _analyze()
    for result in results:
        candidates, openings = _coverage(result)
        print(result.reference.spec.label)
        print("  Certified facets: {}; support gap: {:.3g}".format(len(result.facets), result.support_gap))
        for name, mask in (("Candidate opening", candidates), ("Verified opening", openings)):
            print("  {} intrados joints: {}".format(name, np.flatnonzero(mask[:, 0]).tolist()))
            print("  {} extrados joints: {}".format(name, np.flatnonzero(mask[:, 1]).tolist()))
        for case in result.facets:
            print(
                "  {} load=({:.8f}, {:.8f}) insertion={:.8f}; friction={:.6f}; {}".format(
                    case.label, *case.load, case.interval.midpoint, np.max(case.friction), case.status
                )
            )
        if result.modes:
            print(
                "  Maximum boundary work residual: {:.3g}".format(
                    max(abs(m.metrics["boundary_work"]) for m in result.modes)
                )
            )
        unavailable = [case.label for case in result.samples + result.vertices if not case.mode_ids]
        print("  Cases without verified opening modes: {}".format(unavailable))
    figure = _plot_overview(results)
    figure.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(figure)
    OUTPUT_HTML.write_text(_viewer_html(results), encoding="utf-8")
    print("Saved {} and {}".format(OUTPUT_SVG.name, OUTPUT_HTML.name))


if __name__ == "__main__":
    main()
