"""Compare full-arch robust safe-load regions with a fixed foundation block."""

import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compas.geometry import Box
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_robust_support_dual
from compas_cra.geometry import Arch
from compas_cra.viewers import cra_view_ex
from compas_cra.viewers.cra_view import _load_compas_view2

HEIGHT = 5
SPAN = 10
THICKNESS = 1
DEPTH = 1
FOUNDATION_HEIGHT = THICKNESS
FOUNDATION_NODE = "foundation"
NUM_BLOCKS_RANGE = range(15, 31)
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 36
APPLICATION_FORCE_BOUND = 1e6
VIEW_XLIM = (-1.5, 0.5)
VIEW_YLIM = (1.8, 2.2)
AUTO_VIEW_LIMITS = True
VIEW_PADDING_RATIO = 0.18
MIN_VIEW_SPAN = 0.25
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
METRICS_OUTPUT_SVG = OUTPUT_SVG.with_name("{}_metrics.svg".format(OUTPUT_SVG.stem))
OVERLAY_OUTPUT_SVG = OUTPUT_SVG.with_name("{}_overlay.svg".format(OUTPUT_SVG.stem))
DISPLAY_LOAD_DOFS = (("last", "fx"), ("last", "fz"))
EMPTY_SAFE_SET_TEXT = "safe load set is empty"
SHOW_COMPAS_VIEW2_ASSEMBLY = plt.get_backend().lower() != "agg"
VIEWER_COLUMNS = 4


def sampled_colormap_colors(name, count):
    """Return distinct colors sampled across a Matplotlib colormap."""
    if count <= 0:
        return []
    colormap = plt.get_cmap(name)
    if count == 1:
        return [colormap(0.5)]
    return [colormap(0.05 + 0.90 * index / (count - 1)) for index in range(count)]


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


def foundation_block():
    """Return a constant foundation block under the left springing face."""
    center = Point(-SPAN / 2.0 - THICKNESS / 2.0, DEPTH / 2.0, -FOUNDATION_HEIGHT / 2.0)
    frame = Frame(center, [1, 0, 0], [0, 1, 0])
    return Block.from_shape(Box(THICKNESS, DEPTH, FOUNDATION_HEIGHT, frame=frame))


def build_full_arch(num_blocks):
    """Build a full left-to-right arch assembly with a fixed foundation."""
    arch = Arch(
        height=HEIGHT,
        span=SPAN,
        thickness=THICKNESS,
        depth=DEPTH,
        num_blocks=num_blocks,
        extra_support=False,
    )
    assembly = CRA_Assembly()
    assembly.add_block(foundation_block(), node=FOUNDATION_NODE)
    for node, mesh in enumerate(reversed(arch.blocks())):
        assembly.add_block(mesh.copy(cls=Block), node=node)
    assembly.set_boundary_conditions([FOUNDATION_NODE])
    assembly_interfaces_numpy(assembly, nmax=10, amin=1e-2, tmax=1e-2)
    validate_foundation_contact(assembly)
    return assembly


def validate_foundation_contact(assembly):
    """Verify that the fixed foundation only contacts the first arch block."""
    foundation_edges = []
    for edge in assembly.graph.edges(False):
        if FOUNDATION_NODE not in edge:
            continue
        interfaces = assembly.graph.edge_attribute(edge, "interfaces") or []
        if interfaces:
            foundation_edges.append(edge)

    expected_edges = {(FOUNDATION_NODE, 0), (0, FOUNDATION_NODE)}
    if not foundation_edges:
        raise ValueError("The foundation block does not contact arch node 0.")
    unexpected_edges = [edge for edge in foundation_edges if edge not in expected_edges]
    if unexpected_edges:
        raise ValueError("The foundation block has unexpected contact edges: {}.".format(unexpected_edges))


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
    """Return a readable scale for viewer offsets and load-direction arrows."""
    lower, upper = assembly_bounds(assembly)
    spans = [upper[axis] - lower[axis] for axis in range(3)]
    return max(spans + [0.25])


def translated_display_assembly(assembly, offset):
    """Build a translated display copy of an assembly."""
    transformation = Translation.from_vector(offset)
    display = CRA_Assembly()
    support_nodes = []
    for node in assembly.graph.nodes():
        block = assembly.graph.node_attribute(node, "block")
        display.add_block(block.copy(cls=Block).transformed(transformation), node=node)
        if assembly.graph.node_attribute(node, "is_support"):
            support_nodes.append(node)
    display.set_boundary_conditions(support_nodes)
    assembly_interfaces_numpy(display, nmax=10, amin=1e-2, tmax=1e-2)
    return display


def load_application_centroid(assembly, load_node):
    """Return the centroid of the right exposed radial-face load application points."""
    points = exposed_radial_side_vertices(assembly.graph.node_attribute(load_node, "block"))
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


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


def show_compas_view2_discretizations():
    """Show all discretized full arches with foundations in one compas_view2 window."""
    view2_app, _, _ = _load_compas_view2()
    viewer = view2_app.App(
        title="Example 19-5-1 fixed-foundation arch discretization assemblies",
        width=1600,
        height=900,
        viewmode="shaded",
        show_grid=True,
    )
    reference = build_full_arch(max(NUM_BLOCKS_RANGE))
    lower, upper = assembly_bounds(reference)
    x_step = upper[0] - lower[0] + 1.0
    y_step = max(upper[1] - lower[1] + 1.0, 1.5)

    for index, num_blocks in enumerate(NUM_BLOCKS_RANGE):
        row, column = divmod(index, VIEWER_COLUMNS)
        assembly = build_full_arch(num_blocks)
        load_node = num_blocks - 1
        origin = load_application_centroid(assembly, load_node)
        assembly_lower, _ = assembly_bounds(assembly)
        offset = [
            column * x_step - assembly_lower[0],
            row * y_step - assembly_lower[1],
            0,
        ]
        display = translated_display_assembly(assembly, offset)
        cra_view_ex(
            viewer,
            display,
            blocks=True,
            interfaces=True,
            forces=False,
            forcesdirect=False,
            forcesline=False,
            weights=False,
            displacements=False,
        )
        display_origin = [origin[axis] + offset[axis] for axis in range(3)]
        add_load_direction_arrows(viewer, display, display_origin)

    print("compas_view2 discretization view: red arrow = +Fx, blue arrow = +Fz")
    viewer.run()


def polygon_area(polygon):
    """Return polygon area with the shoelace formula."""
    if len(polygon) < 3:
        return math.nan
    coordinates = np.asarray(polygon, dtype=float)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bounds(polygon):
    """Return ``Fx`` and ``Fz`` bounds for one polygon."""
    if len(polygon) < 3:
        return math.nan, math.nan, math.nan, math.nan
    coordinates = np.asarray(polygon, dtype=float)
    return (
        float(coordinates[:, 0].min()),
        float(coordinates[:, 0].max()),
        float(coordinates[:, 1].min()),
        float(coordinates[:, 1].max()),
    )


def metric_row(num_blocks, result=None, error=None):
    """Return one metrics dictionary for console and plot output."""
    if result is None:
        return {
            "num_blocks": num_blocks,
            "status": "empty" if error else "failed",
            "bounded": False,
            "bounded_directions": 0,
            "center": [],
            "area": math.nan,
            "fx_min": math.nan,
            "fx_max": math.nan,
            "fz_min": math.nan,
            "fz_max": math.nan,
        }

    if result.is_bounded and len(result.outer_polygon) >= 3:
        area = polygon_area(result.outer_polygon)
        fx_min, fx_max, fz_min, fz_max = polygon_bounds(result.outer_polygon)
    else:
        area = math.nan
        fx_min, fx_max, fz_min, fz_max = math.nan, math.nan, math.nan, math.nan

    return {
        "num_blocks": num_blocks,
        "status": "ok",
        "bounded": result.is_bounded,
        "bounded_directions": result.statuses.count("optimal"),
        "center": result.feasible_center,
        "area": area,
        "fx_min": fx_min,
        "fx_max": fx_max,
        "fz_min": fz_min,
        "fz_max": fz_max,
    }


def solve_discretization_cases():
    """Solve fixed-foundation full-arch safe-load regions for each block count."""
    results = []
    labels = []
    metrics = []
    print("blocks | status | bounded | bounded_dirs | area | Fx_min | Fx_max | Fz_min | Fz_max")
    for num_blocks in NUM_BLOCKS_RANGE:
        assembly = build_full_arch(num_blocks)
        load_node = num_blocks - 1
        block = assembly.graph.node_attribute(load_node, "block")
        load_dofs = [(load_node, "fx"), (load_node, "fz")]
        try:
            result = rbe_robust_support_dual(
                assembly,
                load_dofs,
                mu=MU,
                density=DENSITY,
                load_application_points={load_node: exposed_radial_side_vertices(block)},
                application_force_bound=APPLICATION_FORCE_BOUND,
                num_directions=NUM_DIRECTIONS,
            )
        except ValueError as error:
            if EMPTY_SAFE_SET_TEXT not in str(error):
                raise
            row = metric_row(num_blocks, error=error)
            metrics.append(row)
            print_metric_row(row)
            continue

        row = metric_row(num_blocks, result=result)
        metrics.append(row)
        results.append(replace(result, load_dofs=DISPLAY_LOAD_DOFS))
        labels.append("{} blocks".format(num_blocks))
        print_metric_row(row)
    return results, labels, metrics


def format_metric(value):
    """Format a metric value while preserving ``nan`` text."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return "{:.6g}".format(value)


def print_metric_row(row):
    """Print one compact metrics table row."""
    print(
        "{num_blocks:>6} | {status:<6} | {bounded!s:<7} | {bounded_directions:>13} | "
        "{area:>10} | {fx_min:>10} | {fx_max:>10} | {fz_min:>10} | {fz_max:>10}".format(
            num_blocks=row["num_blocks"],
            status=row["status"],
            bounded=row["bounded"],
            bounded_directions=row["bounded_directions"],
            area=format_metric(row["area"]),
            fx_min=format_metric(row["fx_min"]),
            fx_max=format_metric(row["fx_max"]),
            fz_min=format_metric(row["fz_min"]),
            fz_max=format_metric(row["fz_max"]),
        )
    )


def finite_result_points(result):
    """Collect finite visible load points from one robust result."""
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


def render_regions(results, labels):
    """Render all solved safe-load regions in one comparison plot."""
    plt.rcParams["svg.fonttype"] = "none"
    xlim, ylim, auto_limits = view_limits(results)
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        colors=sampled_colormap_colors("turbo", len(results)),
        show_points=False,
        show_center=False,
        xlim=xlim,
        ylim=ylim,
    )
    figure.set_size_inches(10.5, 6.2)
    axes.set_xlabel("Last-block load Fx")
    axes.set_ylabel("Last-block load Fz")
    axes.set_title("Full-arch robust safe-load regions by block discretization")
    axes.text(
        0.02,
        0.02,
        "Full arch; fixed foundation; arch block 0 is free\n"
        "Four right exposed radial-face load points; force bound = 1e6 placeholder",
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
    legend = axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    legend.set_draggable(True)
    figure.tight_layout()
    return figure


def render_visual_overlay(results, labels):
    """Render a clean overlay of all solved safe-load regions."""
    plt.rcParams["svg.fonttype"] = "none"
    colors = sampled_colormap_colors("turbo", len(results))
    xlim, ylim, auto_limits = view_limits(results)
    figure, axes = plt.subplots(figsize=(10.5, 6.2))
    for result, label, color in zip(results, labels, colors):
        if len(result.outer_polygon) < 3:
            continue
        polygon = np.asarray(result.outer_polygon, dtype=float)
        closed_polygon = np.vstack([polygon, polygon[0]])
        axes.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.08)
        axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], color=color, linewidth=1.3, label=label)

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Last-block load Fx")
    axes.set_ylabel("Last-block load Fz")
    axes.set_title("Full-arch safe-load regions with fixed foundation: visual overlay")
    axes.grid(True, alpha=0.3)
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
    legend = axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    legend.set_draggable(True)
    figure.tight_layout()
    return figure


def metric_array(metrics, key):
    """Return one metric series as a NumPy array."""
    return np.asarray([row[key] for row in metrics], dtype=float)


def render_metrics(metrics):
    """Render area and visible-load bounds versus block count."""
    plt.rcParams["svg.fonttype"] = "none"
    block_counts = metric_array(metrics, "num_blocks")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(block_counts, metric_array(metrics, "area"), marker="o", color="#0072B2")
    axes[0].set_title("Outer-polygon area")
    axes[0].set_xlabel("number of arch blocks")
    axes[0].set_ylabel("area")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(block_counts, metric_array(metrics, "fx_min"), marker="o", label="Fx min")
    axes[1].plot(block_counts, metric_array(metrics, "fx_max"), marker="o", label="Fx max")
    axes[1].plot(block_counts, metric_array(metrics, "fz_min"), marker="o", label="Fz min")
    axes[1].plot(block_counts, metric_array(metrics, "fz_max"), marker="o", label="Fz max")
    axes[1].set_title("Visible polygon bounds")
    axes[1].set_xlabel("number of arch blocks")
    axes[1].set_ylabel("load value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("Full-arch fixed-foundation discretization metrics")
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    solved_results, solved_labels, solved_metrics = solve_discretization_cases()
    if solved_results:
        region_figure = render_regions(solved_results, solved_labels)
        save_svg(region_figure, OUTPUT_SVG)
        overlay_figure = render_visual_overlay(solved_results, solved_labels)
        save_svg(overlay_figure, OVERLAY_OUTPUT_SVG)
    else:
        region_figure = None
        overlay_figure = None
        print("no solved regions to plot")

    metrics_figure = render_metrics(solved_metrics)
    save_svg(metrics_figure, METRICS_OUTPUT_SVG)

    if SHOW_COMPAS_VIEW2_ASSEMBLY:
        plt.show(block=False)
        show_compas_view2_discretizations()
    elif plt.get_backend().lower() != "agg":
        plt.show()
    else:
        if region_figure is not None:
            plt.close(region_figure)
        if overlay_figure is not None:
            plt.close(overlay_figure)
        plt.close(metrics_figure)
