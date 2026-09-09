"""Compare arch safe-load regions under disturbance uncertainty."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_dual
from compas_cra.geometry import Arch
from compas_cra.viewers import cra_view_ex
from compas_cra.viewers.cra_view import _load_compas_view2

HEIGHT = 5
SPAN = 10
THICKNESS = 0.8
DEPTH = 0.5
NUM_BLOCKS = 15
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 72
APPLICATION_FORCE_BOUND = 1e6
DISTURBANCE_RATIOS = (0.0, 0.01, 0.02, 0.03)
DISTURBANCE_COMPONENTS = ("fx", "fz")
RANDOM_DISTURBANCE_SAMPLES = 20
RANDOM_DISTURBANCE_SEED = 19002
VIEW_XLIM = (-2.0, 0.0)
VIEW_YLIM = (0.0, 1.0)
AUTO_VIEW_LIMITS = True
VIEW_PADDING_RATIO = 0.18
MIN_VIEW_SPAN = 0.25
SHOW_COMPAS_VIEW2_ASSEMBLY = plt.get_backend().lower() != "agg"
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
EMPTY_SAFE_SET_TEXT = "safe load set is empty"


def point_coordinates(point):
    """Return point coordinates as a plain list."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    return list(point)


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


def left_to_right_blocks():
    """Create four arch blocks ordered left to right."""
    arch = Arch(
        height=HEIGHT,
        span=SPAN,
        thickness=THICKNESS,
        depth=DEPTH,
        num_blocks=NUM_BLOCKS,
        extra_support=False,
    )
    return list(reversed(arch.blocks()))


def build_assembly():
    """Build the fixed arch assembly."""
    assembly = CRA_Assembly()
    for node, mesh in enumerate(left_to_right_blocks()):
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


def loaded_node(assembly):
    """Return the node that carries the visible safe load."""
    return max(assembly.graph.nodes())


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


def show_compas_view2_assembly(assembly):
    """Show the arch assembly in a compas_view2 window."""
    view2_app, _, _ = _load_compas_view2()
    viewer = view2_app.App(
        title="Example 19-2 disturbance arch assembly",
        width=1000,
        height=700,
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
    add_load_direction_arrows(viewer, assembly, load_application_centroid(assembly, loaded_node(assembly)))
    print("compas_view2 arch view: red arrow = +Fx, blue arrow = +Fz")
    viewer.run()


def disturbance_nodes(assembly, load_node):
    """Return nodes that carry external disturbance uncertainty."""
    return [
        node
        for node in assembly.graph.nodes()
        if node != load_node and not assembly.graph.node_attribute(node, "is_support")
    ]


def disturbance_bounds(assembly, nodes, ratio):
    """Return one disturbance-force bound per uncertain node."""
    return [ratio * assembly.graph.node_attribute(node, "block").volume() * DENSITY for node in nodes]


def disturbance_load_dofs(nodes):
    """Return uncertainty DOFs for all selected disturbance nodes."""
    return [(node, component) for node in nodes for component in DISTURBANCE_COMPONENTS]


def coherent_disturbance_vertices(bounds):
    """Return four global-direction disturbance vertices for all uncertain nodes."""
    return [
        [component for bound in bounds for component in (fx_sign * bound, fz_sign * bound)]
        for fx_sign, fz_sign in product((-1.0, 1.0), repeat=2)
    ]


def random_disturbance_vertices(bounds, ratio, sample_count=None):
    """Return reproducible random samples inside all independent disturbance boxes."""
    if sample_count is None:
        sample_count = RANDOM_DISTURBANCE_SAMPLES
    if sample_count <= 0 or not bounds:
        return []
    rng = np.random.default_rng(RANDOM_DISTURBANCE_SEED + int(round(1000.0 * ratio)))
    lower = np.asarray([component for bound in bounds for component in (-bound, -bound)], dtype=float)
    upper = np.asarray([component for bound in bounds for component in (bound, bound)], dtype=float)
    return rng.uniform(lower, upper, size=(sample_count, len(lower))).tolist()


def disturbance_vertices(bounds, ratio):
    """Return coherent corner scenarios plus random independent samples."""
    return coherent_disturbance_vertices(bounds) + random_disturbance_vertices(bounds, ratio)


def disturbance_label(ratio):
    """Return a compact label for one disturbance ratio."""
    return "{:.0f}%".format(100.0 * ratio)


def report_result(ratio, nodes, bounds, scenario_count, result):
    """Print a compact disturbance-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    max_bound = max(bounds) if bounds else 0.0
    total_weight_bound = sum(bounds)
    print(
        "{:<4} | scenarios={} | disturbed_nodes={} | max_bound={:.6g} | sum_bound={:.6g} | bounded={} | "
        "center={} | bounded_dirs={}/{} | outer_polygon_empty={}".format(
            disturbance_label(ratio),
            scenario_count,
            list(nodes),
            max_bound,
            total_weight_bound,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


def report_empty_result(ratio, nodes, bounds, scenario_count, error):
    """Print a compact summary for an empty safe-load set."""
    max_bound = max(bounds) if bounds else 0.0
    total_weight_bound = sum(bounds)
    print(
        "{:<4} | scenarios={} | disturbed_nodes={} | max_bound={:.6g} | sum_bound={:.6g} | "
        "empty safe-load set ({})".format(
            disturbance_label(ratio),
            scenario_count,
            list(nodes),
            max_bound,
            total_weight_bound,
            error,
        )
    )


def is_empty_safe_set_error(error):
    """Return whether a solver error is the expected empty-safe-set signal."""
    return EMPTY_SAFE_SET_TEXT in str(error)


def solve_disturbance_cases(assembly):
    """Solve baseline and disturbance-robust last-block safe-load regions."""
    if len(list(assembly.graph.nodes())) < 4:
        raise ValueError(
            "Example 19-2 requires at least three free/load blocks: nodes 0, 1, 2, and a loaded end node."
        )

    load_node = loaded_node(assembly)
    load_block = assembly.graph.node_attribute(load_node, "block")
    uncertain_nodes = disturbance_nodes(assembly, load_node)
    load_dofs = [(load_node, "fx"), (load_node, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {load_node: exposed_radial_side_vertices(load_block)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("visible load node: {}".format(load_node))
    print("disturbed nodes: {}".format(list(uncertain_nodes)))
    print("hidden hand-force component bound: {:.6g} (placeholder)".format(APPLICATION_FORCE_BOUND))
    print(
        "disturbance model: four coherent global +/-Fx,+/-Fz scenarios plus {} random independent samples; "
        "node i bound = r Wi".format(RANDOM_DISTURBANCE_SAMPLES)
    )
    print("case | disturbance bounds | safe-load summary")

    results = []
    labels = []
    for ratio in DISTURBANCE_RATIOS:
        bounds = disturbance_bounds(assembly, uncertain_nodes, ratio)
        if ratio == 0.0:
            scenario_count = 1
            result = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, **common_options)
        else:
            vertices = disturbance_vertices(bounds, ratio)
            scenario_count = len(vertices)
            try:
                result = rbe_uncertainty_disturb_support_dual(
                    assembly,
                    load_dofs,
                    uncertainty_vertices=vertices,
                    uncertainty_load_dofs=disturbance_load_dofs(uncertain_nodes),
                    **common_options,
                )
            except ValueError as error:
                if not is_empty_safe_set_error(error):
                    raise
                report_empty_result(ratio, uncertain_nodes, bounds, scenario_count, error)
                continue
        report_result(ratio, uncertain_nodes, bounds, scenario_count, result)
        results.append(result)
        labels.append(disturbance_label(ratio))
    return results, labels


def finite_result_points(result):
    """Collect finite visible load points from a robust result."""
    points = []
    points.extend(result.support_points)
    points.extend(result.boundary_points)
    points.extend(result.inner_polygon)
    points.extend(result.outer_polygon)
    if result.feasible_center:
        points.append(result.feasible_center)
    return [point for point in points if len(point) == 2 and all(np.isfinite(point))]


def automatic_view_limits(results):
    """Return padded plot limits from all solved safe-region points."""
    points = []
    for result in results:
        points.extend(finite_result_points(result))
    if not points:
        return VIEW_XLIM, VIEW_YLIM, False

    coordinates = np.asarray(points, dtype=float)
    limits = []
    for axis in range(2):
        lower = float(coordinates[:, axis].min())
        upper = float(coordinates[:, axis].max())
        span = max(upper - lower, MIN_VIEW_SPAN)
        center = 0.5 * (lower + upper)
        padding = VIEW_PADDING_RATIO * span
        limits.append((center - 0.5 * span - padding, center + 0.5 * span + padding))
    return limits[0], limits[1], True


def view_limits(results):
    """Return either automatic or configured plot limits."""
    if AUTO_VIEW_LIMITS:
        return automatic_view_limits(results)
    return VIEW_XLIM, VIEW_YLIM, False


def render_results(results, labels):
    """Render the disturbance comparison plot."""
    plt.rcParams["svg.fonttype"] = "none"
    xlim, ylim, auto_limits = view_limits(results)
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        show_points=False,
        show_center=True,
        xlim=xlim,
        ylim=ylim,
    )
    figure.set_size_inches(10, 6)
    axes.set_xlabel("Loaded block Fx")
    axes.set_ylabel("Loaded block Fz")
    axes.set_title("Arch safe-load regions with block-1/2 disturbance uncertainty")
    axes.text(
        0.02,
        0.02,
        (
            "Disturbance on every non-support, non-loaded block: 4 coherent + {} random samples, bound = r Wi\n"
            "Loaded block uses four right exposed radial-face points; force bound = 1e6 placeholder"
        ).format(RANDOM_DISTURBANCE_SAMPLES),
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    if auto_limits:
        axes.text(
            0.98,
            0.02,
            "auto view limits",
            transform=axes.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
        )
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    assembly = build_assembly()
    results, labels = solve_disturbance_cases(assembly)
    figure = render_results(results, labels)
    save_svg(figure, OUTPUT_SVG)

    if SHOW_COMPAS_VIEW2_ASSEMBLY:
        plt.show(block=False)
        show_compas_view2_assembly(assembly)
    elif plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
