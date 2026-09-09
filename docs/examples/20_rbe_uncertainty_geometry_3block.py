"""Compare finite geometry-uncertainty safe-load regions for a three-block assembly."""

import math

import matplotlib.pyplot as plt
from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Translation
from compas_assembly.datastructures import Block

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium import plot_rbe_robust_results
from compas_cra.equilibrium import rbe_uncertainty_geometry_support_dual

XLIM = (-1.0, 0.1)
ZLIM = (-0.5, 1.0)
APPLICATION_FORCE_BOUND = 1e6
GEOMETRY_SAMPLE_COUNT = 60


class Arch(object):
    """Create the three-block geometry used for the geometry-uncertainty comparison."""

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


def build_assembly():
    """Build the example-18 three-block assembly and detect contact interfaces."""
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


def sampled_scenario_count(kwargs, bounds_name, count_name):
    """Return the generated finite sample count for one bounded uncertainty family."""
    bounds = kwargs.get(bounds_name)
    if bounds is None:
        return 1
    if isinstance(bounds, (int, float)):
        values = [float(bounds), float(bounds)]
    else:
        values = [float(value) for value in bounds]
    if not any(value > 0.0 for value in values):
        return 1
    return int(kwargs.get(count_name, GEOMETRY_SAMPLE_COUNT))


def scenario_count(kwargs):
    """Return the finite geometry scenario count represented by solver keyword arguments."""
    count = 1
    if kwargs.get("interface_scale_factors") is not None:
        count *= len(kwargs["interface_scale_factors"])
    if kwargs.get("point_offset_bounds") is not None:
        count *= sampled_scenario_count(kwargs, "point_offset_bounds", "point_offset_sample_count")
    if kwargs.get("normal_tilt_vectors") is not None:
        count *= len(kwargs["normal_tilt_vectors"])
    if kwargs.get("normal_tilt_bounds") is not None:
        count *= sampled_scenario_count(kwargs, "normal_tilt_bounds", "normal_tilt_sample_count")
    if kwargs.get("contact_failure_scenarios") is not None:
        count *= len(kwargs["contact_failure_scenarios"])
    return count


def report_result(group, label, result, count):
    """Print a compact geometry-uncertainty analysis summary."""
    bounded_directions = result.statuses.count("optimal")
    print(
        "{} | {}: scenarios={}, bounded={}, feasible_center={}, bounded_directions={}/{}".format(
            group,
            label,
            count,
            result.is_bounded,
            result.feasible_center,
            bounded_directions,
            len(result.directions),
        )
    )


def solve_cases(assembly, load_dofs, common_options, group, labels_and_kwargs):
    """Solve one comparison group."""
    results = []
    labels = []
    for label, kwargs in labels_and_kwargs:
        result = rbe_uncertainty_geometry_support_dual(assembly, load_dofs, **common_options, **kwargs)
        report_result(group, label, result, scenario_count(kwargs))
        results.append(result)
        labels.append(label)
    return results, labels


def save_comparison(results, labels, title, filename):
    """Save one safe-load comparison figure."""
    figure, axes = plot_rbe_robust_results(
        results,
        labels=labels,
        show_points=False,
        show_center=False,
        xlim=XLIM,
        ylim=ZLIM,
    )
    figure.set_size_inches(9, 5.5)
    axes.set_xlabel("Block 2 load Fx")
    axes.set_ylabel("Block 2 load Fz")
    axes.set_title(title)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    figure.tight_layout()
    figure.savefig(filename, bbox_inches="tight")
    return figure


if __name__ == "__main__":
    assembly = build_assembly()

    load_dofs = [(2, "fx"), (2, "fz")]
    load_application_points = {
        2: rightmost_side_vertices(assembly.graph.node_attribute(2, "block")),
    }
    common_options = {
        "mu": 0.8,
        "density": 1.0,
        "load_application_points": load_application_points,
        "application_force_bound": APPLICATION_FORCE_BOUND,
        "num_directions": 72,
    }
    comparison_groups = [
        (
            "region shrink",
            "Three-block RBE safe-load region: global interface shrink",
            "docs/examples/20_rbe_uncertainty_geometry_3block_region_shrink.svg",
            [
                ("scale 1.0", {"interface_scale_factors": [1.0]}),
                ("scale 0.9", {"interface_scale_factors": [0.9]}),
                ("scale 0.8", {"interface_scale_factors": [0.8]}),
                ("scale 0.7", {"interface_scale_factors": [0.7]}),
            ],
        ),
        (
            "point offset",
            "Three-block RBE safe-load region: in-plane contact-point offset box",
            "docs/examples/20_rbe_uncertainty_geometry_3block_point_offsets.svg",
            [
                ("nominal", {}),
                (
                    "+/-0.05",
                    {
                        "point_offset_bounds": [0.05, 0.05],
                        "point_offset_sample_count": GEOMETRY_SAMPLE_COUNT,
                        "point_offset_seed": 20,
                    },
                ),
                (
                    "+/-0.10",
                    {
                        "point_offset_bounds": [0.10, 0.10],
                        "point_offset_sample_count": GEOMETRY_SAMPLE_COUNT,
                        "point_offset_seed": 20,
                    },
                ),
                (
                    "+/-0.15",
                    {
                        "point_offset_bounds": [0.15, 0.15],
                        "point_offset_sample_count": GEOMETRY_SAMPLE_COUNT,
                        "point_offset_seed": 20,
                    },
                ),
            ],
        ),
        (
            "normal tilt",
            "Three-block RBE safe-load region: normal tilt about local y",
            "docs/examples/20_rbe_uncertainty_geometry_3block_normal_tilts.svg",
            [
                ("nominal", {}),
                (
                    "+/-5 deg",
                    {
                        "normal_tilt_bounds": [0.0, math.radians(5.0)],
                        "normal_tilt_sample_count": GEOMETRY_SAMPLE_COUNT,
                        "normal_tilt_seed": 30,
                    },
                ),
                (
                    "+/-10 deg",
                    {
                        "normal_tilt_bounds": [0.0, math.radians(10.0)],
                        "normal_tilt_sample_count": GEOMETRY_SAMPLE_COUNT,
                        "normal_tilt_seed": 30,
                    },
                ),
                (
                    "+/-20 deg",
                    {
                        "normal_tilt_bounds": [0.0, math.radians(20.0)],
                        "normal_tilt_sample_count": GEOMETRY_SAMPLE_COUNT,
                        "normal_tilt_seed": 30,
                    },
                ),
            ],
        ),
    ]

    print("interface scale factors are global: one ratio is applied to every detected interface")
    print("point_offset_bounds sample independent in-plane offsets per interface point; dw=0")
    print("normal_tilt_bounds sample independent tilts per interface; this example varies only local y")
    print(
        "sampled geometry uncertainty uses N={} deterministic scenarios including nominal".format(
            GEOMETRY_SAMPLE_COUNT
        )
    )
    print("safe regions are robust intersections over generated geometry scenarios")
    print("hidden hand-force component bound: {} (placeholder)".format(APPLICATION_FORCE_BOUND))

    figures = []
    for group, title, filename, labels_and_kwargs in comparison_groups:
        results, labels = solve_cases(assembly, load_dofs, common_options, group, labels_and_kwargs)
        figures.append(save_comparison(results, labels, title, filename))
        print("saved {}".format(filename))

    if plt.get_backend().lower() != "agg":
        plt.show()
