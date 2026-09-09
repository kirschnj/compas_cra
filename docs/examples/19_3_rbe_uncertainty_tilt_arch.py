"""Compare four-block arch safe-load regions under foundation tilt uncertainty."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_dual
from compas_cra.geometry import Arch
from compas_cra.viewers import cra_view_ex
from compas_cra.viewers.cra_view import _load_compas_view2

HEIGHT = 2
SPAN = 4
THICKNESS = 0.5
DEPTH = 0.5
NUM_BLOCKS = 4
MU = 0.7
DENSITY = 1.0
NUM_DIRECTIONS = 72
APPLICATION_FORCE_BOUND = 1e6
TILT_DEGREES = (0.0, 5.0, 10.0, 15.0, 20.0)
SINGLE_TILT_DEGREES = tuple(range(-20, 21, 5))
TILT_ANGLE_COUNT = 2
VIEW_XLIM = (-2.0, 0.0)
VIEW_YLIM = (0.0, 1.0)
SHOW_COMPAS_VIEW2_ASSEMBLY = plt.get_backend().lower() != "agg"
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
SINGLE_ANGLE_OUTPUT_SVG = OUTPUT_SVG.with_name("{}_single_angles.svg".format(OUTPUT_SVG.stem))
DEGREE = "\N{DEGREE SIGN}"
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"
EMPTY_SAFE_SET_TEXT = "safe load set is empty"


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
    """Build the fixed four-block arch assembly."""
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
    """Show the four-block arch assembly in a compas_view2 window."""
    view2_app, _, _ = _load_compas_view2()
    viewer = view2_app.App(
        title="Example 19-3 tilt arch assembly",
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
    add_load_direction_arrows(viewer, assembly, load_application_centroid(assembly, 3))
    print("compas_view2 arch view: red arrow = +Fx, blue arrow = +Fz")
    viewer.run()


def tilt_label(degrees):
    """Return a compact label for one symmetric tilt interval."""
    if degrees == 0.0:
        return "0{}".format(DEGREE)
    value = "{:.1f}".format(degrees).rstrip("0").rstrip(".")
    return "{}{}{}".format(PLUS_MINUS, value, DEGREE)


def single_tilt_label(degrees):
    """Return a compact label for one deterministic tilt angle."""
    value = "{:+.0f}".format(degrees)
    if value == "+0":
        value = "0"
    return "{}{}".format(value, DEGREE)


def report_result(degrees, scenario_count, result):
    """Print a compact tilt-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{:<7} | scenarios={} | bounded={} | center={} | bounded_dirs={}/{} | outer_polygon_empty={}".format(
            tilt_label(degrees),
            scenario_count,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


def report_empty_result(degrees, scenario_count, error):
    """Print a compact empty-set summary."""
    print("{:<7} | scenarios={} | empty safe-load set ({})".format(tilt_label(degrees), scenario_count, error))


def report_single_angle_result(degrees, result):
    """Print a compact deterministic tilt-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{:<5} | bounded={} | center={} | bounded_dirs={}/{} | outer_polygon_empty={}".format(
            single_tilt_label(degrees),
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


def report_empty_single_angle_result(degrees, error):
    """Print a compact empty-set summary for one deterministic tilt case."""
    print("{:<5} | empty safe-load set ({})".format(single_tilt_label(degrees), error))


def is_empty_safe_set_error(error):
    """Return whether a solver error is the expected empty-safe-set signal."""
    return EMPTY_SAFE_SET_TEXT in str(error)


def solve_tilt_cases(assembly):
    """Solve baseline and tilt-robust block-3 safe-load regions."""
    block_3 = assembly.graph.node_attribute(3, "block")
    load_dofs = [(3, "fx"), (3, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {3: exposed_radial_side_vertices(block_3)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("hidden hand-force component bound: {:.6g} (placeholder)".format(APPLICATION_FORCE_BOUND))
    print("foundation tilt keeps structural matrices fixed and reports loads in world Fx/Fz")
    print("case    | tilt scenario summary")

    results = []
    labels = []
    for degrees in TILT_DEGREES:
        if degrees == 0.0:
            scenario_count = 1
            try:
                result = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, **common_options)
            except ValueError as error:
                if not is_empty_safe_set_error(error):
                    raise
                report_empty_result(degrees, scenario_count, error)
                continue
        else:
            angle = math.radians(degrees)
            scenario_count = TILT_ANGLE_COUNT
            try:
                result = rbe_uncertainty_disturb_support_dual(
                    assembly,
                    load_dofs,
                    tilt_angle_bounds=(-angle, angle),
                    tilt_angle_count=TILT_ANGLE_COUNT,
                    **common_options,
                )
            except ValueError as error:
                if not is_empty_safe_set_error(error):
                    raise
                report_empty_result(degrees, scenario_count, error)
                continue
        report_result(degrees, scenario_count, result)
        results.append(result)
        labels.append(tilt_label(degrees))
    return results, labels


def solve_single_angle_cases(assembly):
    """Solve deterministic block-3 safe-load regions for exact tilt angles."""
    block_3 = assembly.graph.node_attribute(3, "block")
    load_dofs = [(3, "fx"), (3, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {3: exposed_radial_side_vertices(block_3)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("")
    print("deterministic single-angle tilt cases")
    print("case  | safe-load summary")

    results = []
    labels = []
    for degrees in SINGLE_TILT_DEGREES:
        angle = math.radians(degrees)
        try:
            result = rbe_uncertainty_disturb_support_dual(
                assembly,
                load_dofs,
                tilt_angles=[angle],
                **common_options,
            )
        except ValueError as error:
            if not is_empty_safe_set_error(error):
                raise
            report_empty_single_angle_result(degrees, error)
            continue
        report_single_angle_result(degrees, result)
        results.append(result)
        labels.append(single_tilt_label(degrees))
    return results, labels


def render_results(results, labels):
    """Render the four-block tilt comparison plot."""
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        colors=sampled_colormap_colors("viridis", len(results)),
        show_points=False,
        show_center=True,
        xlim=VIEW_XLIM,
        ylim=VIEW_YLIM,
    )
    figure.set_size_inches(10, 6)
    axes.set_xlabel("World block 3 load Fx")
    axes.set_ylabel("World block 3 load Fz")
    axes.set_title("Four-block arch safe-load regions with foundation tilt uncertainty")
    axes.text(
        0.02,
        0.02,
        "Tilt rotates gravity, then maps the safe region back to world Fx/Fz\n"
        "Empty tilt cases are omitted from the curves and listed in the console\n"
        "Block 3 load uses four right exposed radial-face points; force bound = 1e6 placeholder",
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    figure.tight_layout()
    return figure


def render_single_angle_results(results, labels):
    """Render the deterministic single-angle tilt comparison plot."""
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        colors=sampled_colormap_colors("turbo", len(results)),
        show_points=False,
        show_center=True,
        xlim=VIEW_XLIM,
        ylim=VIEW_YLIM,
    )
    figure.set_size_inches(11, 6)
    axes.set_xlabel("World block 3 load Fx")
    axes.set_ylabel("World block 3 load Fz")
    axes.set_title("Four-block arch safe-load regions for deterministic single tilt angles")
    axes.text(
        0.02,
        0.02,
        "Each curve uses one exact tilt and is reported in world Fx/Fz\n"
        "Empty tilt cases are omitted from the curves and listed in the console\n"
        "Block 3 load uses four right exposed radial-face points; force bound = 1e6 placeholder",
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    figure.tight_layout()
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


if __name__ == "__main__":
    assembly = build_assembly()
    results, labels = solve_tilt_cases(assembly)
    figure = render_results(results, labels)
    save_svg(figure, OUTPUT_SVG)

    single_results, single_labels = solve_single_angle_cases(assembly)
    single_figure = render_single_angle_results(single_results, single_labels)
    save_svg(single_figure, SINGLE_ANGLE_OUTPUT_SVG)

    if SHOW_COMPAS_VIEW2_ASSEMBLY:
        plt.show(block=False)
        show_compas_view2_assembly(assembly)
    elif plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
        plt.close(single_figure)
