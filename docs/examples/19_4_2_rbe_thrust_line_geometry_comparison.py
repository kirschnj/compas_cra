"""Compare joint-admissible thrust-line families for four arch geometries."""

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from compas.datastructures import Mesh
from compas_assembly.datastructures import Block
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from scipy.optimize import brentq
from scipy.optimize import linprog
from scipy.optimize import root
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.spatial import ConvexHull

from compas_cra.algorithms import assembly_interfaces_numpy
from compas_cra.datastructures import CRA_Assembly
from compas_cra.equilibrium.cra_helper import equilibrium_setup
from compas_cra.equilibrium.cra_helper import external_force_setup
from compas_cra.equilibrium.cra_helper import friction_setup
from compas_cra.equilibrium.cra_helper import num_vertices

CENTERLINE_SPAN = 11.0
CENTERLINE_HEIGHT = 5.5
THICKNESS = 1.0
DEPTH = 1.0
NUM_BLOCKS = 20
FRICTION_COEFFICIENT = 0.7
DENSITY = 1.0
CONTAINMENT_TOLERANCE = 1e-9
NUM_SUPPORT_DIRECTIONS = 360
NUM_CONTOUR_DIRECTIONS = 360
PLOT_SAMPLE_STRIDE = 8
PROFILE_SAMPLE_COUNT = 20001
EXTENSION_RATIO = 0.10
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
OVERLAY_SVG = Path(__file__).with_name("19_4_2_rbe_thrust_line_geometry_comparison_overlay.svg")


@dataclass(frozen=True)
class ProfileSpec:
    """Definition and display metadata for one centerline profile."""

    key: str
    label: str
    design_gravity_degrees: float
    color: str


@dataclass
class CenterlineProfile:
    """A parametric arch centerline and its solved constants."""

    spec: ProfileSpec
    parameter_start: float
    parameter_end: float
    apex_parameter: float
    point_function: Callable[[float], np.ndarray]
    derivative_function: Callable[[float], np.ndarray]
    catenary_a: Optional[float] = None
    catenary_u0: Optional[float] = None
    catenary_c: Optional[float] = None

    def point(self, parameter):
        """Return one centerline point as ``[x, z]``."""
        return np.asarray(self.point_function(float(parameter)), dtype=float)

    def derivative(self, parameter):
        """Return the centerline derivative with respect to its parameter."""
        return np.asarray(self.derivative_function(float(parameter)), dtype=float)


@dataclass
class GeometryData:
    """Two-dimensional geometry and self-weight data for one arch."""

    profile: CenterlineProfile
    centers: np.ndarray
    weights: np.ndarray
    block_polygons: list
    block_halfspaces: list
    interfaces: list
    left_ground: np.ndarray
    right_ground: np.ndarray
    joint_parameters: np.ndarray
    segment_arclengths: np.ndarray
    centerline_points: np.ndarray


@dataclass
class ForceDiagram:
    """Weighted force polygon for one right-end load."""

    pole: np.ndarray
    nodes: np.ndarray
    directions: np.ndarray


@dataclass
class ThrustTrace:
    """One right-to-left form diagram reconstructed from a weighted force polygon."""

    load: np.ndarray
    right_insertion_x: float
    right_load_point: np.ndarray
    kinks: np.ndarray
    crossings: np.ndarray
    left_support_extension: np.ndarray
    directions: np.ndarray
    block_indices: np.ndarray
    interface_indices: np.ndarray


@dataclass
class InsertionInterval:
    """Admissible interval for the moment-exempt right insertion coordinate."""

    lower: float
    upper: float

    @property
    def feasible(self):
        """Return whether the interval is nonempty."""
        return self.lower <= self.upper

    @property
    def midpoint(self):
        """Return the midpoint of a feasible interval."""
        if not self.feasible:
            raise ValueError("Cannot take the midpoint of an empty insertion interval.")
        return 0.5 * (self.lower + self.upper)


@dataclass
class JointCheck:
    """Finite-joint, compression, and friction diagnostics."""

    interface_index: int
    parameter: float
    overrun: float
    normal_component: float
    friction_utilization: float

    @property
    def valid(self):
        """Return whether the pressure point and resultant are admissible."""
        return (
            self.overrun <= CONTAINMENT_TOLERANCE
            and self.normal_component > CONTAINMENT_TOLERANCE
            and self.friction_utilization <= 1.0 + CONTAINMENT_TOLERANCE
        )


@dataclass
class FamilySample:
    """One color-linked sample on a joint-admissible load contour."""

    boundary_parameter: float
    ray_angle: float
    rbe_radius: float
    admissible_radius: float
    load: np.ndarray
    interval: InsertionInterval
    trace: ThrustTrace


@dataclass
class StabilityCheck:
    """Result of the two-fixed-terminal self-weight feasibility check."""

    feasible: bool
    status: int
    message: str
    equilibrium_residual: float


@dataclass
class GeometryResult:
    """All solved and plotted information for one profile."""

    spec: ProfileSpec
    assembly: CRA_Assembly
    geometry: GeometryData
    self_weight: StabilityCheck
    rbe_boundary: Optional[np.ndarray]
    rbe_center: Optional[np.ndarray]
    family: list
    right_load_status: str


PROFILE_SPECS = (
    ProfileSpec("circular", "Circular reference", 0.0, "#1F77B4"),
    ProfileSpec("catenary", "Symmetric catenary", 0.0, "#009E73"),
    ProfileSpec("asymmetric_negative", "Asymmetric catenary, theta_g = -15 deg", -15.0, "#D55E00"),
    ProfileSpec("asymmetric_positive", "Asymmetric catenary, theta_g = +15 deg", 15.0, "#CC79A7"),
)


def load_sibling_module(filename, module_name):
    """Load one example module located beside this script."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError("Could not load sibling example {}.".format(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def load_example_19_4():
    """Return the full-arch RBE helper example."""
    return load_sibling_module("19_4_rbe_boundary_failure_modes_arch.py", "_compas_cra_example_19_4")


def load_example_19_4_1():
    """Return the preceding thrust-line helper example."""
    return load_sibling_module("19_4_1_rbe_thrust_line_family_arch.py", "_compas_cra_example_19_4_1")


def symmetric_catenary_parameter():
    """Solve the catenary parameter for the common endpoints and rise."""
    half_span = 0.5 * CENTERLINE_SPAN
    return brentq(
        lambda value: value * (math.cosh(half_span / value) - 1.0) - CENTERLINE_HEIGHT,
        0.1,
        100.0,
    )


def build_centerline_profile(spec):
    """Return the solved parametric centerline for one profile specification."""
    half_span = 0.5 * CENTERLINE_SPAN
    if spec.key == "circular":
        radius = CENTERLINE_HEIGHT

        def point(parameter):
            return [radius * math.sin(parameter), radius * math.cos(parameter)]

        def derivative(parameter):
            return [radius * math.cos(parameter), -radius * math.sin(parameter)]

        return CenterlineProfile(
            spec=spec,
            parameter_start=-0.5 * math.pi,
            parameter_end=0.5 * math.pi,
            apex_parameter=0.0,
            point_function=point,
            derivative_function=derivative,
        )

    symmetric_a = symmetric_catenary_parameter()
    if spec.key == "catenary":

        def point(parameter):
            return [parameter, CENTERLINE_HEIGHT - symmetric_a * (math.cosh(parameter / symmetric_a) - 1.0)]

        def derivative(parameter):
            return [1.0, -math.sinh(parameter / symmetric_a)]

        return CenterlineProfile(
            spec=spec,
            parameter_start=-half_span,
            parameter_end=half_span,
            apex_parameter=0.0,
            point_function=point,
            derivative_function=derivative,
            catenary_a=symmetric_a,
            catenary_u0=0.0,
            catenary_c=CENTERLINE_HEIGHT + symmetric_a,
        )

    theta = math.radians(spec.design_gravity_degrees)
    parameter_start = -half_span * math.cos(theta)
    parameter_end = half_span * math.cos(theta)
    left_v = half_span * math.sin(theta)
    right_v = -half_span * math.sin(theta)

    def equations(values):
        catenary_a = math.exp(values[0])
        catenary_u0 = values[1]
        catenary_c = values[2]
        apex_parameter = catenary_u0 + catenary_a * math.asinh(math.tan(theta))
        apex_v = catenary_c - catenary_a * math.cosh((apex_parameter - catenary_u0) / catenary_a)
        apex_z = apex_parameter * math.sin(theta) + apex_v * math.cos(theta)
        return [
            catenary_c - catenary_a * math.cosh((parameter_start - catenary_u0) / catenary_a) - left_v,
            catenary_c - catenary_a * math.cosh((parameter_end - catenary_u0) / catenary_a) - right_v,
            apex_z - CENTERLINE_HEIGHT,
        ]

    solution = root(
        equations,
        [math.log(symmetric_a), 0.0, CENTERLINE_HEIGHT + symmetric_a],
    )
    if not solution.success or np.linalg.norm(equations(solution.x), ord=np.inf) > 1e-10:
        raise RuntimeError("Could not solve asymmetric catenary {}: {}".format(spec.label, solution.message))
    catenary_a = math.exp(solution.x[0])
    catenary_u0 = float(solution.x[1])
    catenary_c = float(solution.x[2])
    apex_parameter = catenary_u0 + catenary_a * math.asinh(math.tan(theta))

    def point(parameter):
        value_v = catenary_c - catenary_a * math.cosh((parameter - catenary_u0) / catenary_a)
        return [
            parameter * math.cos(theta) - value_v * math.sin(theta),
            parameter * math.sin(theta) + value_v * math.cos(theta),
        ]

    def derivative(parameter):
        derivative_v = -math.sinh((parameter - catenary_u0) / catenary_a)
        return [
            math.cos(theta) - derivative_v * math.sin(theta),
            math.sin(theta) + derivative_v * math.cos(theta),
        ]

    return CenterlineProfile(
        spec=spec,
        parameter_start=parameter_start,
        parameter_end=parameter_end,
        apex_parameter=apex_parameter,
        point_function=point,
        derivative_function=derivative,
        catenary_a=catenary_a,
        catenary_u0=catenary_u0,
        catenary_c=catenary_c,
    )


def unit_normal(profile, parameter):
    """Return the normal pointing toward the arch extrados."""
    tangent = profile.derivative(parameter)
    tangent /= np.linalg.norm(tangent)
    return np.asarray([-tangent[1], tangent[0]], dtype=float)


def offset_point(profile, parameter, signed_distance):
    """Return a point offset orthogonally from the centerline."""
    return profile.point(parameter) + signed_distance * unit_normal(profile, parameter)


def equal_arclength_parameters(profile, count=NUM_BLOCKS):
    """Return parameters at equal centerline arc-length intervals."""
    samples = np.linspace(profile.parameter_start, profile.parameter_end, PROFILE_SAMPLE_COUNT)
    points = np.asarray([profile.point(parameter) for parameter in samples], dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    targets = np.linspace(0.0, cumulative[-1], count + 1)
    parameters = np.interp(targets, cumulative, samples)
    return parameters, np.diff(targets)


def ground_intersection_parameter(profile, signed_distance, lower, upper):
    """Intersect an offset profile with the horizontal ground line."""

    def function(parameter):
        return offset_point(profile, parameter, signed_distance)[1]

    if abs(function(lower)) <= 1e-12:
        return float(lower)
    if abs(function(upper)) <= 1e-12:
        return float(upper)
    if function(lower) * function(upper) > 0.0:
        raise ValueError("The 10% extended profile does not bracket an offset-ground intersection.")
    return float(brentq(function, lower, upper))


def mesh_from_xz_polygon(polygon):
    """Extrude a convex x-z polygon through the common arch depth."""
    polygon = [np.asarray(point, dtype=float) for point in polygon]
    signed_area = 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )
    if signed_area > 0.0:
        polygon.reverse()
    count = len(polygon)
    vertices = [[point[0], 0.0, point[1]] for point in polygon]
    vertices += [[point[0], DEPTH, point[1]] for point in polygon]
    faces = [list(reversed(range(count))), list(range(count, 2 * count))]
    faces += [[index, (index + 1) % count, (index + 1) % count + count, index + count] for index in range(count)]
    return Mesh.from_vertices_and_faces(vertices, faces)


def custom_profile_blocks(profile):
    """Build convex voussoirs from normal joints and clipped terminal faces."""
    joint_parameters, segment_arclengths = equal_arclength_parameters(profile)
    parameter_span = profile.parameter_end - profile.parameter_start
    extended_start = profile.parameter_start - EXTENSION_RATIO * parameter_span
    extended_end = profile.parameter_end + EXTENSION_RATIO * parameter_span
    half_thickness = 0.5 * THICKNESS

    left_outer = ground_intersection_parameter(profile, half_thickness, extended_start, profile.parameter_start)
    left_inner = ground_intersection_parameter(profile, -half_thickness, profile.parameter_start, joint_parameters[1])
    right_inner = ground_intersection_parameter(profile, -half_thickness, joint_parameters[-2], profile.parameter_end)
    right_outer = ground_intersection_parameter(profile, half_thickness, profile.parameter_end, extended_end)

    blocks = []
    for index in range(NUM_BLOCKS):
        if index == 0:
            polygon = [
                offset_point(profile, left_outer, half_thickness),
                offset_point(profile, joint_parameters[1], half_thickness),
                offset_point(profile, joint_parameters[1], -half_thickness),
                offset_point(profile, left_inner, -half_thickness),
            ]
        elif index == NUM_BLOCKS - 1:
            polygon = [
                offset_point(profile, joint_parameters[-2], half_thickness),
                offset_point(profile, right_outer, half_thickness),
                offset_point(profile, right_inner, -half_thickness),
                offset_point(profile, joint_parameters[-2], -half_thickness),
            ]
        else:
            polygon = [
                offset_point(profile, joint_parameters[index], half_thickness),
                offset_point(profile, joint_parameters[index + 1], half_thickness),
                offset_point(profile, joint_parameters[index + 1], -half_thickness),
                offset_point(profile, joint_parameters[index], -half_thickness),
            ]
        for point in polygon:
            if abs(point[1]) <= 1e-10:
                point[1] = 0.0
        hull = ConvexHull(np.asarray(polygon, dtype=float))
        if len(hull.vertices) != len(polygon):
            raise ValueError("Generated block {} of {} is not strictly convex.".format(index, profile.spec.label))
        block = mesh_from_xz_polygon(polygon).copy(cls=Block)
        if block.volume() <= 0.0:
            raise ValueError("Generated block {} of {} has non-positive volume.".format(index, profile.spec.label))
        blocks.append(block)
    return blocks, joint_parameters, segment_arclengths


def circular_reference_blocks(base):
    """Return copies of the exact blocks used by examples 19-4 and 19-4-1."""
    assembly = base.build_full_arch()
    blocks = [assembly.node_block(node).copy(cls=Block) for node in assembly.graph.nodes()]
    profile = build_centerline_profile(PROFILE_SPECS[0])
    joint_parameters = np.linspace(profile.parameter_start, profile.parameter_end, NUM_BLOCKS + 1)
    segment_arclengths = np.full(NUM_BLOCKS, CENTERLINE_HEIGHT * math.pi / NUM_BLOCKS)
    return blocks, joint_parameters, segment_arclengths


def assemble_blocks(blocks, supports):
    """Create an assembly and discover the 19 block-to-block interfaces."""
    assembly = CRA_Assembly()
    for node, block in enumerate(blocks):
        assembly.add_block(block.copy(cls=Block), node=node)
    assembly.set_boundary_conditions(list(supports))
    assembly_interfaces_numpy(assembly, nmax=10, amin=1e-2, tmax=1e-2)
    interfaces = [edge for edge in assembly.graph.edges(False) if assembly.graph.edge_attribute(edge, "interfaces")]
    if len(interfaces) != NUM_BLOCKS - 1:
        raise ValueError("Expected 19 block-to-block interfaces, found {}.".format(len(interfaces)))
    return assembly


def point_xz(point):
    """Return one point as a two-component x-z array."""
    if hasattr(point, "x") and hasattr(point, "z"):
        return np.asarray([point.x, point.z], dtype=float)
    return np.asarray([point[0], point[2]], dtype=float)


def unique_points(points, tolerance=1e-9):
    """Return tolerance-deduplicated NumPy points."""
    unique = []
    for point in points:
        candidate = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(candidate - existing) <= tolerance for existing in unique):
            unique.append(candidate)
    return unique


def terminal_ground_segment(block):
    """Return the two x-z endpoints of a terminal block's ground face."""
    points = unique_points(
        point_xz(block.vertex_coordinates(vertex))
        for vertex in block.vertices()
        if abs(block.vertex_coordinates(vertex)[2]) <= 1e-9
    )
    if len(points) != 2:
        raise ValueError("Expected exactly two x-z endpoints on a terminal ground face.")
    return np.asarray(sorted(points, key=lambda point: point[0]), dtype=float)


def extract_geometry(profile, assembly, joint_parameters, segment_arclengths):
    """Extract polygons, interfaces, centers, weights, and reference curves."""
    nodes = list(assembly.graph.nodes())
    centers = []
    weights = []
    polygons = []
    halfspaces = []
    for node in nodes:
        block = assembly.node_block(node)
        centers.append(point_xz(block.center()))
        weights.append(block.volume() * DENSITY)
        points = unique_points(point_xz(block.vertex_coordinates(vertex)) for vertex in block.vertices())
        hull = ConvexHull(np.asarray(points, dtype=float))
        polygons.append(np.asarray(points, dtype=float)[hull.vertices])
        halfspaces.append(hull.equations.copy())

    interfaces = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        points = []
        for interface in assembly.graph.edge_attribute((left, right), "interfaces") or []:
            points.extend(point_xz(point) for point in interface.points)
        unique = unique_points(points)
        if len(unique) != 2:
            raise ValueError("Expected one two-point x-z interface for edge {}.".format((left, right)))
        interfaces.append(np.asarray(unique, dtype=float))

    parameters = np.linspace(profile.parameter_start, profile.parameter_end, 401)
    return GeometryData(
        profile=profile,
        centers=np.asarray(centers, dtype=float),
        weights=np.asarray(weights, dtype=float),
        block_polygons=polygons,
        block_halfspaces=halfspaces,
        interfaces=interfaces,
        left_ground=terminal_ground_segment(assembly.node_block(nodes[0])),
        right_ground=terminal_ground_segment(assembly.node_block(nodes[-1])),
        joint_parameters=np.asarray(joint_parameters, dtype=float),
        segment_arclengths=np.asarray(segment_arclengths, dtype=float),
        centerline_points=np.asarray([profile.point(parameter) for parameter in parameters], dtype=float),
    )


def ground_application_points(block):
    """Return front and back vertices of the right terminal ground face."""
    points = []
    for vertex in block.vertices():
        point = np.asarray(block.vertex_coordinates(vertex), dtype=float)
        if abs(point[2]) <= CONTAINMENT_TOLERANCE:
            points.append(point)
    points = unique_points(points)
    if len(points) != 4:
        raise ValueError("Expected four three-dimensional load points on the terminal ground face.")
    return points


def self_weight_feasibility(assembly, base):
    """Check vertical self-weight with both moment-capable terminal blocks fixed."""
    equilibrium = equilibrium_setup(assembly, penalty=False).tocsr()
    friction = friction_setup(assembly, FRICTION_COEFFICIENT, penalty=False).tocsr()
    vertex_count = num_vertices(assembly)
    force_count = 3 * vertex_count
    normal_nonnegative = csr_matrix(
        (-np.ones(vertex_count), (np.arange(vertex_count), 3 * np.arange(vertex_count))),
        shape=(vertex_count, force_count),
    )
    inequalities = vstack([friction, normal_nonnegative], format="csr")
    baseline = np.asarray(external_force_setup(assembly, DENSITY, None), dtype=float).ravel()

    contact_pairs = base.contact_front_back_pairs(assembly)
    tie_rows = base.variable_tie_rows(force_count, contact_pairs, 3, range(3))
    if tie_rows.shape[0]:
        equilibrium = vstack([equilibrium, tie_rows], format="csr")
        baseline = np.concatenate([baseline, np.zeros(tie_rows.shape[0])])

    result = linprog(
        np.zeros(equilibrium.shape[1]),
        A_ub=inequalities,
        b_ub=np.zeros(inequalities.shape[0]),
        A_eq=equilibrium,
        b_eq=-baseline,
        bounds=[(None, None)] * equilibrium.shape[1],
        method="highs",
    )
    residual = math.inf
    if result.status == 0:
        residual = float(np.linalg.norm(equilibrium.dot(result.x) + baseline, ord=np.inf))
    return StabilityCheck(result.status == 0, result.status, result.message, residual)


def force_diagram(load, weights):
    """Return a right-load force polygon for a right-to-left construction.

    ``load`` is the actual force applied to block 19 in repository coordinates.
    ``weights`` are ordered from block 19 back to block 1; block 0 is fixed and
    therefore excluded from this free-body chain.
    """
    load = np.asarray(load, dtype=float)
    weights = np.asarray(weights, dtype=float)
    pole = load.copy()
    nodes = np.zeros((len(weights) + 1, 2), dtype=float)
    nodes[1:, 1] = np.cumsum(weights)
    directions = nodes - pole
    if np.any(np.abs(directions[:, 0]) <= 1e-12):
        raise ValueError("Graphic statics requires a nonzero horizontal right-load component.")
    return ForceDiagram(pole, nodes, directions)


def cross_2d(first, second):
    """Return the scalar two-dimensional cross product."""
    return first[0] * second[1] - first[1] * second[0]


def point_on_line_at_x(point, direction, x_coordinate):
    """Intersect a directed line with ``x = x_coordinate``."""
    if abs(direction[0]) <= 1e-12:
        raise ValueError("Cannot intersect two vertical lines.")
    return point + direction * ((x_coordinate - point[0]) / direction[0])


def point_on_line_at_z(point, direction, z_coordinate):
    """Intersect a directed line with ``z = z_coordinate``."""
    if abs(direction[1]) <= 1e-12:
        raise ValueError("Cannot intersect a horizontal line with the ground.")
    return point + direction * ((z_coordinate - point[1]) / direction[1])


def line_intersection(point, direction, line_points):
    """Intersect a directed line with the infinite line through two points."""
    line_direction = line_points[1] - line_points[0]
    denominator = cross_2d(direction, line_direction)
    if abs(denominator) <= 1e-12:
        raise ValueError("Thrust segment is parallel to an assembly interface.")
    distance = cross_2d(line_points[0] - point, line_direction) / denominator
    return point + distance * direction


def trace_thrust_line(load, insertion_x, geometry):
    """Trace from the loaded block 19 to the fixed block 0."""
    block_indices = np.arange(len(geometry.centers) - 1, 0, -1, dtype=int)
    interface_indices = np.arange(len(geometry.interfaces) - 1, -1, -1, dtype=int)
    diagram = force_diagram(load, geometry.weights[block_indices])
    point = np.asarray([insertion_x, 0.0], dtype=float)
    kinks = []
    for step, block_index in enumerate(block_indices):
        point = point_on_line_at_x(point, diagram.directions[step], geometry.centers[block_index, 0])
        kinks.append(point)
    kinks = np.asarray(kinks, dtype=float)
    crossings = np.asarray(
        [
            line_intersection(kinks[step], diagram.directions[step + 1], geometry.interfaces[interface_index])
            for step, interface_index in enumerate(interface_indices)
        ],
        dtype=float,
    )
    left_support_extension = point_on_line_at_z(kinks[-1], diagram.directions[-1], 0.0)
    return ThrustTrace(
        load=np.asarray(load, dtype=float),
        right_insertion_x=float(insertion_x),
        right_load_point=np.asarray([insertion_x, 0.0], dtype=float),
        kinks=kinks,
        crossings=crossings,
        left_support_extension=left_support_extension,
        directions=diagram.directions,
        block_indices=block_indices,
        interface_indices=interface_indices,
    )


def interface_parameter(point, interface):
    """Return the affine coordinate of a point along a finite joint."""
    direction = interface[1] - interface[0]
    return float(np.dot(point - interface[0], direction) / np.dot(direction, direction))


def insertion_interval(load, geometry):
    """Return all right insertions whose resultants cross every finite joint."""
    try:
        zero_trace = trace_thrust_line(load, 0.0, geometry)
        unit_trace = trace_thrust_line(load, 1.0, geometry)
    except ValueError:
        return InsertionInterval(1.0, 0.0)
    lower = -math.inf
    upper = math.inf
    for interface_index, zero_point, unit_point in zip(
        zero_trace.interface_indices, zero_trace.crossings, unit_trace.crossings
    ):
        interface = geometry.interfaces[interface_index]
        zero_parameter = interface_parameter(zero_point, interface)
        coefficient = interface_parameter(unit_point, interface) - zero_parameter
        if abs(coefficient) <= 1e-12:
            if zero_parameter < -CONTAINMENT_TOLERANCE or zero_parameter > 1.0 + CONTAINMENT_TOLERANCE:
                return InsertionInterval(1.0, 0.0)
            continue
        candidates = sorted(
            [
                (-CONTAINMENT_TOLERANCE - zero_parameter) / coefficient,
                (1.0 + CONTAINMENT_TOLERANCE - zero_parameter) / coefficient,
            ]
        )
        lower = max(lower, candidates[0])
        upper = min(upper, candidates[1])
    return InsertionInterval(lower, upper)


def joint_checks(trace, geometry):
    """Check finite pressure points, signed compression, and Coulomb friction."""
    checks = []
    for interface_index, point, force in zip(trace.interface_indices, trace.crossings, trace.directions[1:]):
        interface = geometry.interfaces[interface_index]
        parameter = interface_parameter(point, interface)
        overrun = max(0.0, -parameter, parameter - 1.0)
        tangent = interface[1] - interface[0]
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        if normal[0] < 0.0:
            normal *= -1.0
        normal_component = float(np.dot(force, normal))
        tangent_component = abs(float(np.dot(force, tangent)))
        friction_utilization = math.inf
        if normal_component > CONTAINMENT_TOLERANCE:
            friction_utilization = tangent_component / (FRICTION_COEFFICIENT * normal_component)
        checks.append(JointCheck(int(interface_index), parameter, overrun, normal_component, friction_utilization))
    return checks


def trace_is_joint_admissible(trace, geometry):
    """Return whether every physical block-to-block resultant is valid."""
    return all(check.valid for check in joint_checks(trace, geometry))


def load_is_joint_admissible(load, geometry):
    """Return whether a right-end load admits at least one valid right insertion."""
    interval = insertion_interval(load, geometry)
    if not interval.feasible:
        return False
    return trace_is_joint_admissible(trace_thrust_line(load, interval.midpoint, geometry), geometry)


def polygon_centroid(polygon):
    """Return the area centroid of a counterclockwise or clockwise polygon."""
    polygon = np.asarray(polygon, dtype=float)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    area_twice = float(np.sum(cross))
    if abs(area_twice) <= 1e-12:
        return np.mean(polygon, axis=0)
    return np.asarray(
        [
            np.sum((polygon[:, 0] + following[:, 0]) * cross),
            np.sum((polygon[:, 1] + following[:, 1]) * cross),
        ],
        dtype=float,
    ) / (3.0 * area_twice)


def point_polygon_margin(point, halfspaces):
    """Return positive distance to the nearest normalized convex halfspace."""
    return -float(np.max(halfspaces[:, :2].dot(point) + halfspaces[:, 2]))


def find_joint_admissible_center(rbe_boundary, solver_center, geometry):
    """Find a deterministic interior seed for the center-connected component."""
    hull = ConvexHull(np.asarray(rbe_boundary, dtype=float))
    candidates = [
        polygon_centroid(rbe_boundary),
        np.mean(rbe_boundary, axis=0),
        np.asarray(solver_center, dtype=float),
    ]
    for candidate in candidates:
        if point_polygon_margin(candidate, hull.equations) >= -CONTAINMENT_TOLERANCE:
            if load_is_joint_admissible(candidate, geometry):
                return candidate

    lower = np.min(rbe_boundary, axis=0)
    upper = np.max(rbe_boundary, axis=0)
    scale = max(float(np.linalg.norm(upper - lower)), 1.0)
    feasible = []
    for force_x in np.linspace(lower[0], upper[0], 41):
        for force_z in np.linspace(lower[1], upper[1], 41):
            candidate = np.asarray([force_x, force_z], dtype=float)
            margin = point_polygon_margin(candidate, hull.equations)
            if margin < -CONTAINMENT_TOLERANCE or not load_is_joint_admissible(candidate, geometry):
                continue
            interval = insertion_interval(candidate, geometry)
            feasible.append((margin / scale + (interval.upper - interval.lower) / scale, candidate))
    if not feasible:
        return None
    return max(feasible, key=lambda item: item[0])[1]


def rbe_ray_radius(center, direction, halfspaces):
    """Return the forward distance from a center to a convex RBE boundary."""
    residuals = halfspaces[:, :2].dot(center) + halfspaces[:, 2]
    denominators = halfspaces[:, :2].dot(direction)
    forward = denominators > 1e-12
    if np.any(residuals > CONTAINMENT_TOLERANCE) or not np.any(forward):
        raise ValueError("Could not intersect a ray from the RBE center.")
    return max(float(np.min(-residuals[forward] / denominators[forward])), 0.0)


def assign_boundary_parameters(samples):
    """Assign normalized arclength parameters around an ordered contour."""
    edge_lengths = np.asarray(
        [
            np.linalg.norm(samples[(index + 1) % len(samples)].load - sample.load)
            for index, sample in enumerate(samples)
        ],
        dtype=float,
    )
    perimeter = float(np.sum(edge_lengths))
    travelled = 0.0
    for index, sample in enumerate(samples):
        sample.boundary_parameter = travelled / perimeter
        travelled += edge_lengths[index]


def build_admissible_contour(rbe_boundary, center, geometry, num_directions=NUM_CONTOUR_DIRECTIONS):
    """Reconstruct every radial sample directly on the RBE boundary."""
    if not load_is_joint_admissible(center, geometry):
        raise ValueError("The selected family center is not joint-admissible.")
    halfspaces = ConvexHull(np.asarray(rbe_boundary, dtype=float)).equations
    samples = []
    for angle in np.linspace(0.0, 2.0 * math.pi, num_directions, endpoint=False):
        direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        rbe_radius = rbe_ray_radius(center, direction, halfspaces)
        admissible_radius = rbe_radius
        load = center + rbe_radius * direction
        interval = insertion_interval(load, geometry)
        if not interval.feasible:
            raise ValueError("An RBE-boundary load has no finite-joint insertion interval.")
        trace = trace_thrust_line(load, interval.midpoint, geometry)
        if not trace_is_joint_admissible(trace, geometry):
            raise ValueError("An RBE-boundary trace is not joint-admissible.")
        samples.append(FamilySample(0.0, float(angle), rbe_radius, admissible_radius, load, interval, trace))
    assign_boundary_parameters(samples)
    return samples


def analyze_geometry(spec, support_directions=NUM_SUPPORT_DIRECTIONS, contour_directions=NUM_CONTOUR_DIRECTIONS):
    """Generate, check, and solve one geometry comparison row."""
    base = load_example_19_4()
    family_base = load_example_19_4_1()
    profile = build_centerline_profile(spec)
    if spec.key == "circular":
        blocks, joint_parameters, segment_arclengths = circular_reference_blocks(base)
    else:
        blocks, joint_parameters, segment_arclengths = custom_profile_blocks(profile)

    one_sided = assemble_blocks(blocks, [0])
    two_sided = assemble_blocks(blocks, [0, NUM_BLOCKS - 1])
    geometry = extract_geometry(profile, one_sided, joint_parameters, segment_arclengths)
    self_weight = self_weight_feasibility(two_sided, base)

    load_node = NUM_BLOCKS - 1
    application_points = {load_node: ground_application_points(one_sided.node_block(load_node))}
    problem = base.hidden_load_problem(
        one_sided,
        load_node,
        [(load_node, "fx"), (load_node, "fz")],
        application_points,
        penalty=False,
    )
    try:
        analysis = base._analyze_load_set(problem, {})
    except ValueError as error:
        return GeometryResult(spec, one_sided, geometry, self_weight, None, None, [], str(error))
    if not analysis.is_bounded:
        return GeometryResult(
            spec,
            one_sided,
            geometry,
            self_weight,
            None,
            None,
            [],
            "RBE load set is unbounded and has no closed family plot",
        )

    rbe_boundary = family_base.solve_primal_rbe_boundary(base, problem, num_directions=support_directions)
    center = find_joint_admissible_center(rbe_boundary, analysis.feasible_center, geometry)
    if center is None:
        return GeometryResult(
            spec,
            one_sided,
            geometry,
            self_weight,
            rbe_boundary,
            None,
            [],
            "no joint-admissible right-load solution space",
        )
    family = build_admissible_contour(rbe_boundary, center, geometry, contour_directions)
    return GeometryResult(spec, one_sided, geometry, self_weight, rbe_boundary, center, family, "available")


def analyze_geometries(support_directions=NUM_SUPPORT_DIRECTIONS, contour_directions=NUM_CONTOUR_DIRECTIONS):
    """Solve all four comparison geometries."""
    return [
        analyze_geometry(spec, support_directions=support_directions, contour_directions=contour_directions)
        for spec in PROFILE_SPECS
    ]


def closed_points(points):
    """Return points with the first row appended."""
    return np.vstack([points, points[0]])


def right_load_coordinates(points):
    """Return actual repository right-end load coordinates without reflection."""
    points = np.asarray(points, dtype=float)
    return points.copy()


def trace_polyline(trace):
    """Return the thin force-action polyline through all CoG concurrency points."""
    return np.vstack([trace.right_load_point, trace.kinks, trace.left_support_extension])


def pressure_polyline(trace):
    """Return the illustrative path through consecutive joint pressure points."""
    return np.vstack([trace.right_load_point, trace.crossings, trace.left_support_extension])


def polygon_area(points):
    """Return the unsigned area of an ordered two-dimensional polygon."""
    points = np.asarray(points, dtype=float)
    following = np.roll(points, -1, axis=0)
    return 0.5 * abs(float(np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])))


def representative_sample(result):
    """Return the minimum-Fx family sample used by the force-diagram panel."""
    if not result.family:
        return None
    return min(result.family, key=lambda sample: sample.load[0])


def family_color(sample, colormap):
    """Return the shared contour/form color of one family sample."""
    return colormap(sample.boundary_parameter)


def common_load_limits(results):
    """Return shared actual right-load coordinate limits for load plots."""
    points = []
    for result in results:
        if result.rbe_boundary is not None:
            points.extend(right_load_coordinates(result.rbe_boundary))
        if result.family:
            points.extend(right_load_coordinates([sample.load for sample in result.family]))
    points = np.asarray(points, dtype=float)
    spans = np.maximum(np.ptp(points, axis=0), 0.5)
    return points.min(axis=0) - 0.12 * spans, points.max(axis=0) + 0.12 * spans


def draw_load_panel(axes, result, colormap, limits):
    """Draw one right-end RBE load region and pressure-path boundary."""
    if result.rbe_boundary is not None:
        rbe = right_load_coordinates(result.rbe_boundary)
        axes.add_patch(PolygonPatch(rbe, closed=True, facecolor="0.88", edgecolor="0.45", linewidth=1.0))
    if result.family:
        contour = right_load_coordinates([sample.load for sample in result.family])
        segments = [[contour[index], contour[(index + 1) % len(contour)]] for index in range(len(contour))]
        axes.add_collection(
            LineCollection(
                segments,
                colors=[family_color(sample, colormap) for sample in result.family],
                linewidths=2.4,
            )
        )
        center = right_load_coordinates([result.rbe_center])[0]
        axes.plot(center[0], center[1], marker="+", color="black", markersize=8, markeredgewidth=1.5)
        representative = representative_sample(result)
        representative_point = right_load_coordinates([representative.load])[0]
        axes.plot(
            representative_point[0],
            representative_point[1],
            marker="o",
            markerfacecolor=family_color(representative, colormap),
            markeredgecolor="black",
            markersize=7,
            zorder=5,
        )
        max_friction = max(
            check.friction_utilization
            for sample in result.family
            for check in joint_checks(sample.trace, result.geometry)
        )
        text = (
            "RBE load-region area = {:.3f}\n"
            "pressure-path load-region area = {:.3f}\n"
            "maximum friction utilization = {:.3f}"
        ).format(
            polygon_area(rbe),
            polygon_area(contour),
            max_friction,
        )
        axes.text(
            0.03,
            0.04,
            text,
            transform=axes.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    else:
        axes.text(
            0.5,
            0.5,
            result.right_load_status,
            transform=axes.transAxes,
            ha="center",
            va="center",
            color="#B22222",
            fontsize=10,
            wrap=True,
        )
    axes.axhline(0.0, color="0.85", linewidth=0.7)
    axes.axvline(0.0, color="0.85", linewidth=0.7)
    axes.grid(True, color="0.9", linewidth=0.6)
    axes.set_xlim(limits[0][0], limits[1][0])
    axes.set_ylim(limits[0][1], limits[1][1])
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("right-end load Fx")
    axes.set_ylabel("right-end load Fz")


def draw_arch_panel(axes, result, colormap):
    """Draw one block geometry and its color-linked thrust-line family."""
    geometry = result.geometry
    for polygon in geometry.block_polygons:
        axes.add_patch(
            PolygonPatch(polygon, closed=True, facecolor="#EFA3A3", edgecolor="#9E1B1B", linewidth=0.55, alpha=0.72)
        )
    axes.plot(
        geometry.centerline_points[:, 0],
        geometry.centerline_points[:, 1],
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    for sample in result.family[::PLOT_SAMPLE_STRIDE]:
        color = family_color(sample, colormap)
        action = trace_polyline(sample.trace)
        pressure = pressure_polyline(sample.trace)
        axes.plot(action[:, 0], action[:, 1], color=color, linewidth=0.35, alpha=0.28)
        if len(pressure) > 3:
            axes.plot(pressure[1:-1, 0], pressure[1:-1, 1], color=color, linewidth=0.85, alpha=0.78)
        axes.plot(pressure[:2, 0], pressure[:2, 1], color=color, linewidth=0.55, linestyle="--", alpha=0.55)
        axes.plot(pressure[-2:, 0], pressure[-2:, 1], color=color, linewidth=0.55, linestyle="--", alpha=0.55)

    representative = representative_sample(result)
    if representative is not None:
        color = family_color(representative, colormap)
        action = trace_polyline(representative.trace)
        pressure = pressure_polyline(representative.trace)
        axes.plot(action[:, 0], action[:, 1], color=color, linewidth=1.0, alpha=0.9)
        axes.plot(pressure[1:-1, 0], pressure[1:-1, 1], color=color, linewidth=2.2, alpha=1.0)
        axes.plot(pressure[:2, 0], pressure[:2, 1], color=color, linewidth=1.0, linestyle="--", alpha=0.9)
        axes.plot(pressure[-2:, 0], pressure[-2:, 1], color=color, linewidth=1.0, linestyle="--", alpha=0.9)
        arrow_target = representative.trace.right_load_point + 0.35 * (
            representative.trace.kinks[0] - representative.trace.right_load_point
        )
        axes.annotate(
            "",
            xy=arrow_target,
            xytext=representative.trace.right_load_point,
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.8},
            zorder=8,
        )
        axes.plot(
            representative.trace.right_load_point[0],
            representative.trace.right_load_point[1],
            marker="*",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            zorder=9,
        )
        axes.plot(
            representative.trace.kinks[:, 0],
            representative.trace.kinks[:, 1],
            linestyle="none",
            marker="o",
            markersize=2.8,
            markerfacecolor="#56B4E9",
            markeredgecolor="none",
            zorder=5,
        )
        axes.plot(
            representative.trace.crossings[:, 0],
            representative.trace.crossings[:, 1],
            linestyle="none",
            marker="D",
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.35,
            zorder=6,
        )
    axes.axhline(0.0, color="0.25", linewidth=0.8)
    axes.set_xlim(-6.7, 6.7)
    axes.set_ylim(-0.6, 6.4)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("z")
    axes.grid(True, color="0.92", linewidth=0.5)


def draw_force_panel(axes, result, colormap):
    """Draw the representative weighted force polygon and its direction rays."""
    representative = representative_sample(result)
    if representative is None:
        axes.axis("off")
        axes.text(0.5, 0.5, "No representative force diagram", transform=axes.transAxes, ha="center", va="center")
        return
    free_weights = result.geometry.weights[:0:-1]
    diagram = force_diagram(representative.load, free_weights)
    color = family_color(representative, colormap)
    axes.plot(diagram.nodes[:, 0], diagram.nodes[:, 1], color="#B22222", linewidth=1.4, marker="o", markersize=2.8)
    for node in diagram.nodes:
        axes.plot(
            [diagram.pole[0], node[0]],
            [diagram.pole[1], node[1]],
            color=color,
            linewidth=0.7,
            alpha=0.68,
        )
    axes.plot(
        diagram.pole[0],
        diagram.pole[1],
        marker="D",
        markersize=7,
        markerfacecolor=color,
        markeredgecolor="black",
    )
    axes.text(
        0.03,
        0.04,
        "right load = ({:.3f}, {:.3f})\nsum W(blocks 19..1) = {:.4f}\nW range = [{:.4f}, {:.4f}]".format(
            representative.load[0],
            representative.load[1],
            np.sum(free_weights),
            np.min(free_weights),
            np.max(free_weights),
        ),
        transform=axes.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    diagram_points = np.vstack([diagram.nodes, diagram.pole])
    lower = np.min(diagram_points, axis=0)
    upper = np.max(diagram_points, axis=0)
    padding = np.maximum(0.06 * (upper - lower), [0.35, 0.35])
    axes.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axes.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    axes.grid(True, color="0.92", linewidth=0.5)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Fx")
    axes.set_ylabel("Fz")


def plot_geometry_comparison(results):
    """Return the detailed load-region, arch-family, and force-diagram figure."""
    colormap = plt.get_cmap("turbo")
    limits = common_load_limits(results)
    figure, axes = plt.subplots(
        len(results),
        3,
        figsize=(18, 19),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.45, 0.62]},
    )
    figure.suptitle(
        "Example 19-4-2: geometry-dependent joint-pressure thrust-line families",
        fontsize=15,
        y=0.997,
    )
    for row, result in enumerate(results):
        draw_load_panel(axes[row, 0], result, colormap, limits)
        draw_arch_panel(axes[row, 1], result, colormap)
        draw_force_panel(axes[row, 2], result, colormap)
        stability = "stable" if result.self_weight.feasible else "UNSTABLE"
        axes[row, 0].set_title(
            "{}\nself-weight: {}; right-load family: {}".format(result.spec.label, stability, result.right_load_status)
        )
        axes[row, 1].set_title("45 right-to-left family members; representative pressure points shown")
        axes[row, 2].set_title("Representative minimum-Fx right-load force diagram")
    handles = [
        Line2D([0], [0], color="0.45", linewidth=7, alpha=0.25, label="compression-only RBE region"),
        Line2D([0], [0], color=colormap(0.35), linewidth=2.2, label="verified pressure-path boundary"),
        Line2D([0], [0], color="black", marker="*", linestyle="none", label="right-end construction start"),
        Line2D([0], [0], color="#56B4E9", marker="o", linestyle="none", label="CoG concurrency point"),
        Line2D([0], [0], color="black", marker="D", linestyle="none", label="finite-joint pressure point"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.985))
    return figure


def overlay_metrics(result):
    """Return concise numerical metrics for the comparison summary."""
    values = {
        "weight": float(np.sum(result.geometry.weights)),
        "rbe_area": math.nan,
        "pressure_path_area": math.nan,
        "max_friction": math.nan,
        "fx": (math.nan, math.nan),
        "fz": (math.nan, math.nan),
    }
    if result.rbe_boundary is not None:
        values["rbe_area"] = polygon_area(result.rbe_boundary)
    if result.family:
        right_loads = right_load_coordinates([sample.load for sample in result.family])
        values["pressure_path_area"] = polygon_area(right_loads)
        values["max_friction"] = max(
            check.friction_utilization
            for sample in result.family
            for check in joint_checks(sample.trace, result.geometry)
        )
        values["fx"] = (float(np.min(right_loads[:, 0])), float(np.max(right_loads[:, 0])))
        values["fz"] = (float(np.min(right_loads[:, 1])), float(np.max(right_loads[:, 1])))
    return values


def plot_comparison_overlay(results):
    """Return a compact geometry, load-contour, and metric overlay figure."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.25]})
    geometry_axes, load_axes, metrics_axes = axes
    load_limits = common_load_limits(results)
    for result in results:
        color = result.spec.color
        for polygon in result.geometry.block_polygons:
            geometry_axes.plot(*closed_points(polygon).T, color=color, linewidth=0.35, alpha=0.22)
        points = result.geometry.centerline_points
        geometry_axes.plot(points[:, 0], points[:, 1], color=color, linewidth=2.0, label=result.spec.label)
        if result.rbe_boundary is not None:
            rbe = right_load_coordinates(result.rbe_boundary)
            load_axes.plot(*closed_points(rbe).T, color=color, linewidth=0.9, linestyle="--", alpha=0.55)
        if result.family:
            contour = right_load_coordinates([sample.load for sample in result.family])
            load_axes.plot(*closed_points(contour).T, color=color, linewidth=2.0, label=result.spec.label)

    geometry_axes.axhline(0.0, color="0.25", linewidth=0.8)
    geometry_axes.set_title("Common centerline endpoints and rise")
    geometry_axes.set_xlabel("x")
    geometry_axes.set_ylabel("z")
    geometry_axes.set_aspect("equal", adjustable="box")
    geometry_axes.grid(True, color="0.92", linewidth=0.5)
    geometry_axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)

    load_axes.axhline(0.0, color="0.85", linewidth=0.7)
    load_axes.axvline(0.0, color="0.85", linewidth=0.7)
    load_axes.set_title("Right-end load contours\ndashed = RBE, solid = verified pressure path")
    load_axes.set_xlabel("right-end load Fx")
    load_axes.set_ylabel("right-end load Fz")
    load_axes.set_aspect("equal", adjustable="box")
    load_axes.set_xlim(load_limits[0][0], load_limits[1][0])
    load_axes.set_ylim(load_limits[0][1], load_limits[1][1])
    load_axes.grid(True, color="0.92", linewidth=0.5)
    load_axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)

    metrics_axes.axis("off")
    lines = [
        "Geometry comparison metrics",
        "",
        "All analyses use vertical actual gravity and mu = 0.7.",
        "Ranges use the actual force applied to right-end block 19.",
        "",
    ]
    for result in results:
        values = overlay_metrics(result)
        lines.extend(
            [
                result.spec.label,
                "  self-weight={}  right-load={}".format(
                    "stable" if result.self_weight.feasible else "unstable",
                    result.right_load_status,
                ),
                "  sum W={:.4f}  RBE load area={:.4f}  pressure-path load area={:.4f}".format(
                    values["weight"], values["rbe_area"], values["pressure_path_area"]
                ),
                "  maximum friction utilization={:.3f}".format(values["max_friction"]),
                "  Fx=[{:.3f}, {:.3f}]  Fz=[{:.3f}, {:.3f}]".format(
                    values["fx"][0], values["fx"][1], values["fz"][0], values["fz"][1]
                ),
                "",
            ]
        )
    metrics_axes.text(
        0.0, 1.0, "\n".join(lines), transform=metrics_axes.transAxes, va="top", family="monospace", fontsize=8.5
    )
    figure.tight_layout()
    return figure


def report_results(results):
    """Print solved profile, stability, load-region, and family diagnostics."""
    print("example 19-4-2 geometry comparison")
    print("actual gravity = (0, 0, -1); design-gravity angle only generates the asymmetric centerline")
    for result in results:
        profile = result.geometry.profile
        start = profile.point(profile.parameter_start)
        apex = profile.point(profile.apex_parameter)
        end = profile.point(profile.parameter_end)
        print("{}:".format(result.spec.label))
        print(
            "  centerline start=({:.6f}, {:.6f}), apex=({:.6f}, {:.6f}), end=({:.6f}, {:.6f})".format(
                start[0], start[1], apex[0], apex[1], end[0], end[1]
            )
        )
        if profile.catenary_a is not None:
            print(
                "  catenary a={:.9f}, u0={:.9f}, c={:.9f}".format(
                    profile.catenary_a, profile.catenary_u0, profile.catenary_c
                )
            )
        print(
            "  blocks={}, total weight={:.6f}, block-weight range=[{:.6f}, {:.6f}]".format(
                NUM_BLOCKS,
                np.sum(result.geometry.weights),
                np.min(result.geometry.weights),
                np.max(result.geometry.weights),
            )
        )
        print(
            "  two-fixed-terminal self-weight: {} (equilibrium residual {:.3e})".format(
                "stable" if result.self_weight.feasible else "UNSTABLE",
                result.self_weight.equilibrium_residual,
            )
        )
        print("  one-sided right-load family: {}".format(result.right_load_status))
        if result.family:
            metrics = overlay_metrics(result)
            print(
                "  RBE area={:.6f}, pressure-path load-region area={:.6f}, max friction={:.6f}, "
                "right-load Fx=[{:.6f}, {:.6f}], Fz=[{:.6f}, {:.6f}]".format(
                    metrics["rbe_area"],
                    metrics["pressure_path_area"],
                    metrics["max_friction"],
                    metrics["fx"][0],
                    metrics["fx"][1],
                    metrics["fz"][0],
                    metrics["fz"][1],
                )
            )


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


def main():
    """Solve four geometries and save the detailed and overlay SVGs."""
    results = analyze_geometries()
    report_results(results)
    detailed = plot_geometry_comparison(results)
    overlay = plot_comparison_overlay(results)
    save_svg(detailed, OUTPUT_SVG)
    save_svg(overlay, OVERLAY_SVG)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(detailed)
        plt.close(overlay)


if __name__ == "__main__":
    main()
