"""Compare three-block safe-load regions under block-1 disturbance uncertainty."""

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
NUM_DIRECTIONS = 72
APPLICATION_FORCE_BOUND = 1e6
DISTURBANCE_RATIOS = (0.0, 0.05, 0.10, 0.15, 0.20)
VIEW_XLIM = (-1.0, 0.1)
VIEW_YLIM = (-0.5, 1.0)
OUTPUT_SVG = Path(__file__).with_suffix(".svg")


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


def disturbance_vertices(magnitude):
    """Return the four vertices of a block-1 Fx/Fz disturbance box."""
    return [
        [-magnitude, -magnitude],
        [-magnitude, magnitude],
        [magnitude, -magnitude],
        [magnitude, magnitude],
    ]


def disturbance_label(ratio):
    """Return a compact label for one disturbance ratio."""
    return "{:.0f}% disturbance".format(100.0 * ratio)


def report_result(ratio, magnitude, result):
    """Print a compact disturbance-analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{:<15} | bound={:.6g} | bounded={} | center={} | bounded_dirs={}/{} | outer_polygon_empty={}".format(
            disturbance_label(ratio),
            magnitude,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
            not bool(result.outer_polygon),
        )
    )


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


def solve_disturbance_cases(assembly):
    """Solve the baseline and disturbance-robust block-2 safe-load regions."""
    block_1 = assembly.graph.node_attribute(1, "block")
    block_2 = assembly.graph.node_attribute(2, "block")
    block_1_weight = block_1.volume() * DENSITY
    load_dofs = [(2, "fx"), (2, "fz")]
    common_options = {
        "mu": MU,
        "density": DENSITY,
        "load_application_points": {2: rightmost_side_vertices(block_2)},
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": NUM_DIRECTIONS,
    }

    print("block 1 weight: {:.6g}".format(block_1_weight))
    print("hidden hand-force component bound: {:.6g} (placeholder)".format(APPLICATION_FORCE_BOUND))
    print("block 1 disturbance box: Fx,Fz independently in [-r W1, r W1]")
    print("case            | disturbance bound | safe-load summary")

    results = []
    labels = []
    for ratio in DISTURBANCE_RATIOS:
        magnitude = ratio * block_1_weight
        if ratio == 0.0:
            result = rbe_uncertainty_disturb_support_dual(assembly, load_dofs, **common_options)
        else:
            result = rbe_uncertainty_disturb_support_dual(
                assembly,
                load_dofs,
                uncertainty_vertices=disturbance_vertices(magnitude),
                uncertainty_load_dofs=[(1, "fx"), (1, "fz")],
                **common_options,
            )
        report_result(ratio, magnitude, result)
        results.append(result)
        labels.append(disturbance_label(ratio))
    return results, labels


def render_results(results, labels):
    """Render the disturbance comparison plot."""
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        show_points=False,
        show_center=True,
        xlim=VIEW_XLIM,
        ylim=VIEW_YLIM,
    )
    figure.set_size_inches(10, 6)
    axes.set_xlabel("Block 2 load Fx")
    axes.set_ylabel("Block 2 load Fz")
    axes.set_title("Three-block safe-load regions with block-1 disturbance uncertainty")
    axes.text(
        0.02,
        0.02,
        "Block 1 disturbance: Fx,Fz in +/- r W1\nBlock 2 load uses four rightmost-side points",
        transform=axes.transAxes,
        fontsize=8,
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

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
