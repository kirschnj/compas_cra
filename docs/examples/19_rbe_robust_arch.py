"""Track robust RBE safe-load regions during left-to-right arch construction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_robust_support_primal
from compas_cra.geometry import Arch
from compas_cra.viewers import cra_view_ex
from compas_cra.viewers.cra_view import _load_compas_view2

HEIGHT = 5
SPAN = 10
THICKNESS = 0.5
THICKNESS_VALUES = (0.3, 0.5, 0.8)
DEPTH = 0.5
NUM_BLOCKS = 20
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 36
APPLICATION_FORCE_BOUND = 1e6
SHOW_COMPAS_VIEW2_ASSEMBLY = plt.get_backend().lower() != "agg"
VIEW_XLIM = (-3.0, 0.5)
VIEW_YLIM = (-3.2, 2.4)
MAX_ZOOM_SPAN = 10.0
MAX_ZOOM_COORDINATE = 100.0
MIN_ZOOM_SPAN = 0.25
OUTPUT_SVG = Path(__file__).with_suffix(".svg")


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


def construction_blocks(thickness):
    """Create arch blocks in left-to-right construction order."""
    arch = Arch(
        height=HEIGHT,
        span=SPAN,
        thickness=thickness,
        depth=DEPTH,
        num_blocks=NUM_BLOCKS,
        extra_support=False,
    )
    return list(reversed(arch.blocks()))


def construction_stage(blocks, stage):
    """Create a stage assembly containing blocks ``0..stage - 1``."""
    assembly = CRA_Assembly()
    for node, mesh in enumerate(blocks[:stage]):
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
    """Return the centroid of the four right exposed radial-face load application points."""
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


def show_compas_view2_construction(blocks):
    """Show the default-thickness construction stages in one compas_view2 window."""
    view2_app, _, _ = _load_compas_view2()
    viewer = view2_app.App(
        title="Example 19 robust RBE construction assemblies",
        width=1600,
        height=900,
        viewmode="shaded",
        show_grid=True,
    )
    full_assembly = construction_stage(blocks, NUM_BLOCKS)
    lower, upper = assembly_bounds(full_assembly)
    x_step = upper[0] - lower[0] + 1.0
    y_step = max(upper[1] - lower[1] + 1.0, 1.5)
    columns = 5

    for index, stage in enumerate(range(2, NUM_BLOCKS + 1)):
        row, column = divmod(index, columns)
        assembly = construction_stage(blocks, stage)
        stage_lower, _ = assembly_bounds(assembly)
        offset = [
            column * x_step - stage_lower[0],
            row * y_step - stage_lower[1],
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
        origin = load_application_centroid(assembly, stage - 1)
        display_origin = [origin[axis] + offset[axis] for axis in range(3)]
        add_load_direction_arrows(viewer, display, display_origin)

    print("compas_view2 construction view: red arrow = +Fx, blue arrow = +Fz")
    viewer.run()


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


def stage_view_limits(result):
    """Return readable plot limits and whether the stage is viewport-clipped."""
    points = finite_result_points(result)
    if not points:
        return VIEW_XLIM, VIEW_YLIM, True

    coordinates = np.asarray(points, dtype=float)
    spans = coordinates.max(axis=0) - coordinates.min(axis=0)
    max_coordinate = float(np.max(np.abs(coordinates)))
    if max_coordinate > MAX_ZOOM_COORDINATE or float(max(spans)) > MAX_ZOOM_SPAN:
        return VIEW_XLIM, VIEW_YLIM, True

    limits = []
    for axis in range(2):
        lower = float(coordinates[:, axis].min())
        upper = float(coordinates[:, axis].max())
        span = max(upper - lower, MIN_ZOOM_SPAN)
        center = 0.5 * (lower + upper)
        padding = 0.2 * span
        limits.append((center - 0.5 * span - padding, center + 0.5 * span + padding))
    return limits[0], limits[1], False


def thickness_svg_path(thickness):
    """Return the SVG path for one thickness-specific construction plot."""
    suffix = "{:.1f}".format(thickness).replace(".", "_")
    return OUTPUT_SVG.with_name("{}_thickness_{}.svg".format(OUTPUT_SVG.stem, suffix))


def report_result(thickness, stage, result):
    """Print a compact robust-analysis summary for one construction stage."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "thickness {} stage {}: bounded={}, feasible_center={}, bounded_directions={}/{}".format(
            thickness,
            stage,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
        )
    )


def report_empty_stage(thickness, stage):
    """Print a compact summary for a stage with an empty safe-load set."""
    print("thickness {} stage {}: empty safe load set".format(thickness, stage))


def plot_empty_stage(axis, stage):
    """Mark one construction stage as infeasible in the stage grid."""
    axis.set_xlim(VIEW_XLIM)
    axis.set_ylim(VIEW_YLIM)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.set_title("stage {}".format(stage))
    axis.text(
        0.5,
        0.5,
        "empty\nsafe set",
        transform=axis.transAxes,
        ha="center",
        va="center",
    )


def render_thickness(thickness):
    """Render construction safe-load regions for one arch thickness."""
    blocks = construction_blocks(thickness)
    stages = list(range(2, NUM_BLOCKS + 1))
    figure, axes_grid = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes_grid.ravel()

    for axis, stage in zip(axes, stages):
        assembly = construction_stage(blocks, stage)
        load_node = stage - 1
        block = assembly.graph.node_attribute(load_node, "block")
        try:
            result = rbe_robust_support_primal(
                assembly,
                [(load_node, "fx"), (load_node, "fz")],
                mu=MU,
                density=DENSITY,
                load_application_points={load_node: exposed_radial_side_vertices(block)},
                application_force_bound=APPLICATION_FORCE_BOUND,
                num_directions=NUM_DIRECTIONS,
            )
        except ValueError as error:
            if "safe load set is empty" not in str(error):
                raise
            report_empty_stage(thickness, stage)
            plot_empty_stage(axis, stage)
            continue

        report_result(thickness, stage, result)
        xlim, ylim, is_clipped = stage_view_limits(result)
        plot_rbe_robust_results(
            result,
            labels=["stage {}".format(stage)],
            ax=axis,
            show_points=False,
            show_center=True,
            xlim=xlim,
            ylim=ylim,
        )
        axis.set_title("stage {}".format(stage))
        if is_clipped:
            axis.text(
                0.04,
                0.94,
                "clipped",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
        legend = axis.get_legend()
        if legend:
            legend.remove()

    for axis in axes[len(stages) :]:
        axis.axis("off")
        axis.text(
            0.05,
            0.65,
            "thickness = {}\nFour right exposed radial-face points\nforce bound = 1e6\nplaceholder only".format(
                thickness
            ),
            transform=axis.transAxes,
            va="top",
        )

    figure.suptitle("Arch construction robust safe-load regions, thickness = {}".format(thickness))
    figure.supxlabel("Current last-block load Fx")
    figure.supylabel("Current last-block load Fz")
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    figures = []
    for thickness in THICKNESS_VALUES:
        figure = render_thickness(thickness)
        figures.append(figure)
        save_svg(figure, thickness_svg_path(thickness))
        if thickness == THICKNESS:
            save_svg(figure, OUTPUT_SVG)

    if SHOW_COMPAS_VIEW2_ASSEMBLY:
        plt.show(block=False)
        show_compas_view2_construction(construction_blocks(THICKNESS))
    elif plt.get_backend().lower() != "agg":
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
