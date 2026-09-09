"""Compare arch safe-load regions under gravity/material uncertainty."""

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

HEIGHT = 5
SPAN = 10
THICKNESS = 0.8
DEPTH = 0.5
NUM_BLOCKS = 15
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 36
APPLICATION_FORCE_BOUND = 1e6
UNCERTAINTY_RATIOS = (0.0, 0.01, 0.02, 0.03)
RANDOM_UNCERTAINTY_SAMPLES = 60
RANDOM_UNCERTAINTY_SEED = 19201
VIEW_XLIM = (-2.0, 0.0)
VIEW_YLIM = (0.0, 1.0)
AUTO_VIEW_LIMITS = True
VIEW_PADDING_RATIO = 0.18
MIN_VIEW_SPAN = 0.25
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
    """Create arch blocks ordered left to right."""
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


def loaded_node(assembly):
    """Return the node that carries the visible safe load."""
    return max(assembly.graph.nodes())


def uncertainty_nodes(assembly, load_node):
    """Return nodes that carry material/gravity uncertainty."""
    return [
        node
        for node in assembly.graph.nodes()
        if node != load_node and not assembly.graph.node_attribute(node, "is_support")
    ]


def block_x_span(block):
    """Return the global x-span of one block mesh."""
    x_values = [point_coordinates(block.vertex_coordinates(vertex))[0] for vertex in block.vertices()]
    return max(x_values) - min(x_values)


def node_weight(assembly, node):
    """Return nominal block weight magnitude for one node."""
    block = assembly.graph.node_attribute(node, "block")
    return block.volume() * DENSITY


def vertical_bounds(assembly, nodes, ratio):
    """Return vertical material uncertainty bounds ``r Wi``."""
    return [ratio * node_weight(assembly, node) for node in nodes]


def moment_bounds(assembly, nodes, ratio):
    """Return CoG-offset moment uncertainty bounds ``Wi * r * x_span_i``."""
    bounds = []
    for node in nodes:
        block = assembly.graph.node_attribute(node, "block")
        bounds.append(node_weight(assembly, node) * ratio * block_x_span(block))
    return bounds


def material_uncertainty_model(assembly, nodes, ratio, include_moment):
    """Return uncertainty DOFs and component bounds for one material model."""
    vertical = vertical_bounds(assembly, nodes, ratio)
    moments = moment_bounds(assembly, nodes, ratio) if include_moment else []
    dofs = []
    bounds = []
    for index, node in enumerate(nodes):
        dofs.append((node, "fz"))
        bounds.append(vertical[index])
        if include_moment:
            dofs.append((node, "my"))
            bounds.append(moments[index])
    return dofs, bounds, vertical, moments


def coherent_vertices(bounds, components_per_node):
    """Return coherent sign vertices for all uncertainty components."""
    if not bounds:
        return []
    if components_per_node == 1:
        return [[sign * bound for bound in bounds] for sign in (-1.0, 1.0)]
    if components_per_node == 2:
        vertices = []
        for fz_sign, moment_sign in product((-1.0, 1.0), repeat=2):
            vertex = []
            for index, bound in enumerate(bounds):
                sign = fz_sign if index % 2 == 0 else moment_sign
                vertex.append(sign * bound)
            vertices.append(vertex)
        return vertices
    raise ValueError("components_per_node must be 1 or 2.")


def random_vertices(bounds, ratio, model_offset, sample_count=None):
    """Return reproducible random samples inside the independent uncertainty box."""
    if sample_count is None:
        sample_count = RANDOM_UNCERTAINTY_SAMPLES
    if sample_count <= 0 or not bounds:
        return []
    rng = np.random.default_rng(RANDOM_UNCERTAINTY_SEED + model_offset + int(round(1000.0 * ratio)))
    lower = -np.asarray(bounds, dtype=float)
    upper = np.asarray(bounds, dtype=float)
    return rng.uniform(lower, upper, size=(sample_count, len(bounds))).tolist()


def uncertainty_vertices(bounds, ratio, components_per_node, model_offset):
    """Return coherent sign scenarios plus random independent samples."""
    return coherent_vertices(bounds, components_per_node) + random_vertices(bounds, ratio, model_offset)


def ratio_label(ratio):
    """Return a compact label for one uncertainty ratio."""
    return "{:.0f}%".format(100.0 * ratio)


def is_empty_safe_set_error(error):
    """Return whether a solver error is the expected empty-safe-set signal."""
    return EMPTY_SAFE_SET_TEXT in str(error)


def report_case(model_name, ratio, nodes, vertical, moments, scenario_count, result=None, error=None):
    """Print one compact material-uncertainty result row."""
    max_vertical = max(vertical) if vertical else 0.0
    sum_vertical = sum(vertical)
    max_moment = max(moments) if moments else 0.0
    sum_moment = sum(moments)
    prefix = (
        "{:<18} | {:<3} | scenarios={} | nodes={} | max_dFz={:.6g} | sum_dFz={:.6g} | max_dMy={:.6g} | sum_dMy={:.6g}"
    ).format(
        model_name,
        ratio_label(ratio),
        scenario_count,
        list(nodes),
        max_vertical,
        sum_vertical,
        max_moment,
        sum_moment,
    )
    if error is not None:
        print("{} | empty safe-load set ({})".format(prefix, error))
        return
    bounded_directions = result.statuses.count("optimal")
    print(
        "{} | bounded={} | center={} | bounded_dirs={}/{} | outer_polygon_empty={}".format(
            prefix,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


def solve_model(assembly, model_name, include_moment, model_offset):
    """Solve all uncertainty ratios for one gravity/material model."""
    if len(list(assembly.graph.nodes())) < 4:
        raise ValueError(
            "Example 19-2-1 requires at least three free/load blocks: nodes 0, 1, 2, and a loaded end node."
        )

    load_node = loaded_node(assembly)
    load_block = assembly.graph.node_attribute(load_node, "block")
    nodes = uncertainty_nodes(assembly, load_node)
    load_dofs = [(load_node, "fx"), (load_node, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {load_node: exposed_radial_side_vertices(load_block)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("")
    print("{} material uncertainty".format(model_name))
    print("visible load node: {}".format(load_node))
    print("uncertain nodes: {}".format(list(nodes)))
    print("case | material-uncertainty summary")

    results = []
    labels = []
    components_per_node = 2 if include_moment else 1
    for ratio in UNCERTAINTY_RATIOS:
        dofs, bounds, vertical, moments = material_uncertainty_model(assembly, nodes, ratio, include_moment)
        if ratio == 0.0:
            scenario_count = 1
            result = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, **common_options)
        else:
            vertices = uncertainty_vertices(bounds, ratio, components_per_node, model_offset)
            scenario_count = len(vertices)
            try:
                result = rbe_uncertainty_disturb_support_dual(
                    assembly,
                    load_dofs,
                    uncertainty_vertices=vertices,
                    uncertainty_load_dofs=dofs,
                    **common_options,
                )
            except ValueError as error:
                if not is_empty_safe_set_error(error):
                    raise
                report_case(model_name, ratio, nodes, vertical, moments, scenario_count, error=error)
                continue
        report_case(model_name, ratio, nodes, vertical, moments, scenario_count, result=result)
        results.append(result)
        labels.append(ratio_label(ratio))
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


def automatic_view_limits(result_groups):
    """Return padded plot limits from all solved safe-region points."""
    points = []
    for results, _ in result_groups:
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


def view_limits(result_groups):
    """Return either automatic or configured plot limits."""
    if AUTO_VIEW_LIMITS:
        return automatic_view_limits(result_groups)
    return VIEW_XLIM, VIEW_YLIM, False


def render_results(weight_results, weight_labels, moment_results, moment_labels):
    """Render weight-only and weight-plus-moment uncertainty comparisons."""
    plt.rcParams["svg.fonttype"] = "none"
    result_groups = [(weight_results, weight_labels), (moment_results, moment_labels)]
    xlim, ylim, auto_limits = view_limits(result_groups)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    panels = (
        (axes[0], weight_results, weight_labels, "Weight uncertainty only"),
        (axes[1], moment_results, moment_labels, "Weight + CoG moment uncertainty"),
    )
    for axis, results, labels, title in panels:
        plot_rbe_robust_results(
            results,
            labels=labels,
            ax=axis,
            show_points=False,
            show_center=True,
            xlim=xlim,
            ylim=ylim,
        )
        axis.set_xlabel("Loaded block Fx")
        axis.set_ylabel("Loaded block Fz")
        axis.set_title(title)
        legend = axis.get_legend()
        if legend:
            legend.set_draggable(True)
        axis.text(
            0.02,
            0.02,
            (
                "Uncertain nodes exclude supports and loaded block\nRandom samples = {}; force bound = 1e6 placeholder"
            ).format(RANDOM_UNCERTAINTY_SAMPLES),
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
        )
    if auto_limits:
        axes[1].text(
            0.98,
            0.02,
            "auto view limits",
            transform=axes[1].transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
        )
    figure.suptitle("Example 19-2-1 gravity/material uncertainty safe-load regions")
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    assembly = build_assembly()
    weight_results, weight_labels = solve_model(assembly, "weight only", include_moment=False, model_offset=0)
    moment_results, moment_labels = solve_model(assembly, "weight + CoG My", include_moment=True, model_offset=10_000)
    figure = render_results(weight_results, weight_labels, moment_results, moment_labels)
    save_svg(figure, OUTPUT_SVG)

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
