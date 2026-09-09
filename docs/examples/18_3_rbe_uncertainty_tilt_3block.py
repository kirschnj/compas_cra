"""Compare three-block safe-load regions under foundation tilt uncertainty."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_uncertainty_disturb_support_dual

MU = 0.8
DENSITY = 1.0
NUM_DIRECTIONS = 144
APPLICATION_FORCE_BOUND = 1e6
TILT_DEGREES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
SINGLE_TILT_DEGREES = tuple(range(-60, 61, 10))
TILT_ANGLE_COUNT = 10
VIEW_XLIM = (-1.0, 0.1)
VIEW_YLIM = (-0.5, 1.0)
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


class Arch(object):
    """Create the three-block geometry used for the robust RBE comparison."""

    def __init__(self, b0_width, b1_height, b1_base, b2_base, b2_top, alpha, beta, gamma, thickness):
        super().__init__()
        self.b0_width = b0_width
        self.b1_height = b1_height
        self.b1_base = b1_base
        self.b2_base = b2_base
        self.b2_top = b2_top
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.thickness = thickness

    def assembly(self):
        """Create the assembly with block 0 fixed."""
        assembly = CRA_Assembly()
        for mesh in self.blocks():
            assembly.add_block(mesh.copy(cls=Block))
        assembly.graph.node_attribute(0, "is_support", True)
        return assembly

    def blocks(self):
        """Create the three block meshes."""
        blocks = []
        faces = [
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [3, 7, 4, 0],
            [6, 2, 1, 5],
            [7, 3, 2, 6],
            [5, 1, 0, 4],
        ]
        translation = Translation.from_vector([0, self.thickness, 0])

        b0_1 = Point(0, 0, 0)
        b0_2 = Point(-self.b1_height * math.tan(self.alpha - math.pi / 2), 0, self.b1_height)
        b0_0 = Point(
            self.b0_width * math.cos(self.alpha + math.pi / 2),
            0,
            self.b0_width * math.sin(self.alpha + math.pi / 2),
        )
        b0_3 = Point((b0_0 + b0_2 - b0_1).x, (b0_0 + b0_2 - b0_1).y, (b0_0 + b0_2 - b0_1).z)
        b0_front = [b0_0, b0_1, b0_2, b0_3]
        b0_back = [vertex.transformed(translation) for vertex in b0_front]
        blocks.append(Mesh.from_vertices_and_faces(b0_front + b0_back, faces))

        b1_0 = b0_1
        b1_3 = b0_2
        b1_1 = Point(self.b1_base, 0, 0)
        b1_2 = Point(
            self.b1_base + self.b1_height * math.tan(self.beta - math.pi / 2),
            0,
            self.b1_height,
        )
        b1_front = [b1_0, b1_1, b1_2, b1_3]
        b1_back = [vertex.transformed(translation) for vertex in b1_front]
        blocks.append(Mesh.from_vertices_and_faces(b1_front + b1_back, faces))

        b2_0 = b1_1
        b2_3 = b1_2
        b2_1 = Point(
            self.b1_base + self.b2_base * math.cos(math.pi - self.beta - self.gamma),
            0,
            self.b2_base * math.sin(math.pi - self.beta - self.gamma),
        )
        b2_direction = (b2_1 - b2_0).unitized()
        b2_2 = Point(
            (b2_direction * self.b2_top + b2_3).x,
            (b2_direction * self.b2_top + b2_3).y,
            (b2_direction * self.b2_top + b2_3).z,
        )
        b2_front = [b2_0, b2_1, b2_2, b2_3]
        b2_back = [vertex.transformed(translation) for vertex in b2_front]
        blocks.append(Mesh.from_vertices_and_faces(b2_front + b2_back, faces))

        return blocks


def point_coordinates(point):
    """Return point coordinates as a plain list."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return [point.x, point.y, point.z]
    return list(point)


def rightmost_side_vertices(block):
    """Return the four vertices of the block face with maximum x-coordinate."""
    face_coordinates = []
    for face in block.faces():
        coordinates = [point_coordinates(block.vertex_coordinates(vertex)) for vertex in block.face_vertices(face)]
        face_coordinates.append(coordinates)
    return max(face_coordinates, key=lambda coordinates: sum(point[0] for point in coordinates) / len(coordinates))


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
    """Print a compact empty-set summary for one interval tilt case."""
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


def build_assembly():
    """Build the example-18 three-block assembly."""
    geometry = Arch(
        b0_width=0.5,
        b1_height=0.5,
        b1_base=0.5,
        b2_base=0.6,
        b2_top=0.9,
        alpha=math.pi / 2,
        beta=2 * math.pi / 3,
        gamma=2 * math.pi / 3,
        thickness=1,
    )
    assembly = geometry.assembly()
    assembly_interfaces_numpy(assembly)
    return assembly


def common_solver_options(assembly):
    """Return solver options shared by interval and single-angle tilt cases."""
    block_2 = assembly.graph.node_attribute(2, "block")
    return {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {2: rightmost_side_vertices(block_2)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }


def solve_tilt_cases(assembly):
    """Solve baseline and tilt-robust block-2 safe-load regions."""
    load_dofs = [(2, "fx"), (2, "fz")]
    common_options = common_solver_options(assembly)

    print("hidden hand-force component bound: {:.6g} (placeholder)".format(APPLICATION_FORCE_BOUND))
    print("foundation tilt keeps structural matrices fixed and reports loads in world Fx/Fz")
    print("symmetric interval tilt cases")
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
    """Solve deterministic block-2 safe-load regions for exact tilt angles."""
    load_dofs = [(2, "fx"), (2, "fz")]
    common_options = common_solver_options(assembly)

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
    """Render the three-block tilt comparison plot."""
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
    axes.set_xlabel("World block 2 load Fx")
    axes.set_ylabel("World block 2 load Fz")
    axes.set_title("Three-block safe-load regions with foundation tilt uncertainty")
    axes.text(
        0.02,
        0.02,
        "Tilt rotates gravity, then maps the safe region back to world Fx/Fz\n"
        "Empty tilt cases are omitted from the curves and listed in the console\n"
        "Block 2 load uses four rightmost-side points; force bound = 1e6 placeholder",
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
    axes.set_xlabel("World block 2 load Fx")
    axes.set_ylabel("World block 2 load Fz")
    axes.set_title("Three-block safe-load regions for deterministic single tilt angles")
    axes.text(
        0.02,
        0.02,
        "Each curve uses one exact tilt and is reported in world Fx/Fz\n"
        "Empty tilt cases are omitted from the curves and listed in the console\n"
        "Block 2 load uses four rightmost-side points; force bound = 1e6 placeholder",
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

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
        plt.close(single_figure)
