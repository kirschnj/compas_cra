"""Color-link a joint-admissible load contour to a family of 2D thrust lines."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from compas_cra.equilibrium.rbe_robust import _inner_polygon

NUM_SUPPORT_DIRECTIONS = 360
NUM_CONTOUR_DIRECTIONS = 360
PLOT_SAMPLE_STRIDE = 4
FRICTION_COEFFICIENT = 0.7
DENSITY = 1.0
THRUST_CENTER_FX = -4.0
CONTAINMENT_TOLERANCE = 1e-9
FIT_INSERTION_GUARD = 1e-9
REFERENCE_CASES = (
    ("R1", (-5.0, -8.6), -0.46),
    ("R2", (-3.5, -9.05), 0.35),
    ("R3", (-2.9, -8.6), 0.47),
    ("R4", (-3.5, -8.15), -0.26),
)
OUTPUT_SVG = Path(__file__).with_suffix(".svg")
DIAGNOSTIC_SVG = Path(__file__).with_name("19_4_1_rbe_thrust_line_supplied_cases.svg")
INSIDE_COLOR = "#009E73"
OUTSIDE_COLOR = "#D55E00"
POINT_COLOR = "#0072B2"
EXEMPT_COLOR = "#6B7280"
EXACT_COLOR = "#4B5563"


@dataclass
class ArchGeometry:
    """Two-dimensional geometry needed by the graphic-statics construction."""

    centers: np.ndarray
    weights: np.ndarray
    block_polygons: list
    block_halfspaces: list
    interfaces: list
    left_support_x: float


@dataclass
class ForceDiagram:
    """Force-polygon data for one load in repository coordinates."""

    pole: np.ndarray
    nodes: np.ndarray
    directions: np.ndarray


@dataclass
class ThrustTrace:
    """One form diagram constructed from a force diagram and insertion point."""

    load: np.ndarray
    insertion_x: float
    left_anchor: np.ndarray
    kinks: np.ndarray
    crossings: np.ndarray
    right_anchor: np.ndarray
    directions: np.ndarray


@dataclass
class InsertionInterval:
    """Admissible scalar interval for the left ground insertion point."""

    lower: float
    upper: float
    lower_source: Optional[tuple] = None
    upper_source: Optional[tuple] = None

    @property
    def feasible(self):
        """Return whether the interval is non-empty within numerical tolerance."""
        return self.lower <= self.upper

    @property
    def midpoint(self):
        """Return the midpoint of a feasible interval."""
        if not self.feasible:
            raise ValueError("Cannot take the midpoint of an empty insertion interval.")
        return 0.5 * (self.lower + self.upper)


@dataclass
class FamilySample:
    """One color-linked direction on the maximal joint-admissible contour."""

    boundary_parameter: float
    ray_angle: float
    rbe_radius: float
    admissible_radius: float
    admissible_load: np.ndarray
    interval: InsertionInterval
    trace: ThrustTrace


@dataclass
class JointCheck:
    """Finite-joint and friction diagnostics for one interface resultant."""

    interface_index: int
    parameter: float
    overrun: float
    normal_component: float
    friction_utilization: float
    point: np.ndarray

    @property
    def valid(self):
        """Return whether the resultant acts on the joint and inside its friction cone."""
        return (
            self.overrun <= CONTAINMENT_TOLERANCE
            and self.normal_component > CONTAINMENT_TOLERANCE
            and self.friction_utilization <= 1.0 + CONTAINMENT_TOLERANCE
        )


@dataclass
class SuppliedCaseDiagnostic:
    """Exact and nearest joint-admissible reconstructions of one supplied case."""

    label: str
    supplied_anchor_load: np.ndarray
    supplied_insertion_fraction: float
    supplied_trace: ThrustTrace
    supplied_checks: list
    fitted_anchor_load: np.ndarray
    fitted_insertion_fraction: float
    fitted_trace: ThrustTrace
    fitted_checks: list
    rbe_margin: float
    fitted_rbe_margin: float
    fitted_interval: InsertionInterval


def load_example_19_4():
    """Load the preceding example as a private sibling module."""
    module_name = "_compas_cra_example_19_4"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).with_name("19_4_rbe_boundary_failure_modes_arch.py")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError("Could not load example 19-4 from {}.".format(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def point_xz(point):
    """Return one point as an ``[x, z]`` NumPy array."""
    if hasattr(point, "x") and hasattr(point, "z"):
        return np.asarray([point.x, point.z], dtype=float)
    return np.asarray([point[0], point[2]], dtype=float)


def unique_rows(points, tolerance=1e-9):
    """Return tolerance-deduplicated two-dimensional points."""
    unique = []
    for point in points:
        candidate = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(candidate - existing) <= tolerance for existing in unique):
            unique.append(candidate)
    return unique


def build_arch_geometry(assembly):
    """Extract convex block polygons, mesh weights, CoGs, and interfaces."""
    nodes = list(assembly.graph.nodes())
    centers = []
    weights = []
    polygons = []
    halfspaces = []
    for node in nodes:
        block = assembly.graph.node_attribute(node, "block")
        centers.append(point_xz(block.center()))
        weights.append(block.volume() * DENSITY)
        points = unique_rows(point_xz(block.vertex_coordinates(vertex)) for vertex in block.vertices())
        hull = ConvexHull(np.asarray(points, dtype=float))
        polygons.append(np.asarray(points, dtype=float)[hull.vertices])
        halfspaces.append(hull.equations.copy())

    interfaces = []
    for left, right in zip(nodes[:-1], nodes[1:]):
        points = []
        for interface in assembly.graph.edge_attribute((left, right), "interfaces") or []:
            points.extend(point_xz(point) for point in interface.points)
        unique = unique_rows(points)
        if len(unique) != 2:
            raise ValueError("Expected one two-point x-z interface for edge {}.".format((left, right)))
        interfaces.append(np.asarray(unique, dtype=float))

    left_support_x = float(np.min(polygons[0][:, 0]))
    return ArchGeometry(
        centers=np.asarray(centers, dtype=float),
        weights=np.asarray(weights, dtype=float),
        block_polygons=polygons,
        block_halfspaces=halfspaces,
        interfaces=interfaces,
        left_support_x=left_support_x,
    )


def solve_primal_rbe_boundary(base, problem, num_directions=NUM_SUPPORT_DIRECTIONS):
    """Return feasible primal support points ordered as a polygon."""
    support_points = []
    for direction in base._directions(num_directions):
        objective = -np.asarray(problem.load_projection.transpose().dot(direction), dtype=float).ravel()
        result = linprog(
            objective,
            A_ub=problem.inequalities,
            b_ub=problem.inequality_rhs,
            A_eq=problem.equilibrium,
            b_eq=-problem.baseline_load,
            bounds=[(None, None)] * problem.equilibrium.shape[1],
            method="highs",
        )
        if result.status != 0:
            raise RuntimeError("Primal support solve failed: {}.".format(result.message))
        support_points.append(np.asarray(problem.load_projection.dot(result.x), dtype=float).tolist())

    polygon = _inner_polygon(support_points, base.SUPPORT_TOLERANCE)
    if len(polygon) < 3:
        raise ValueError("Example 19-4-1 requires a bounded primal safe-load polygon.")
    return np.asarray(polygon, dtype=float)


def force_diagram(load, weights):
    """Construct the force diagram that determines all form-diagram slopes.

    ``load`` uses the example-19 repository convention: ``Fz`` is positive
    upward on the held rightmost block. The displayed left-anchor reaction is
    ``(Fx, -Fz)``. Consequently that anchor vector is the pole, while the
    cumulative mesh-derived block weights form a vertical load line through
    the origin.
    """
    load = np.asarray(load, dtype=float)
    weights = np.asarray(weights, dtype=float)
    pole = np.asarray([load[0], -load[1]], dtype=float)
    nodes = np.zeros((len(weights) + 1, 2), dtype=float)
    nodes[1:, 1] = -np.cumsum(weights)
    directions = nodes - pole
    if np.any(np.abs(directions[:, 0]) <= 1e-12):
        raise ValueError("Graphic-statics construction requires a nonzero horizontal anchor component.")
    return ForceDiagram(pole=pole, nodes=nodes, directions=directions)


def cross_2d(first, second):
    """Return the scalar two-dimensional cross product."""
    return first[0] * second[1] - first[1] * second[0]


def point_on_line_at_x(point, direction, x_coordinate):
    """Intersect a directed line with ``x = x_coordinate``."""
    if abs(direction[0]) <= 1e-12:
        raise ValueError("Cannot intersect a vertical line with a vertical CoG load line.")
    return point + direction * ((x_coordinate - point[0]) / direction[0])


def point_on_line_at_z(point, direction, z_coordinate):
    """Intersect a directed line with ``z = z_coordinate``."""
    if abs(direction[1]) <= 1e-12:
        raise ValueError("Cannot intersect a horizontal line with the ground line.")
    return point + direction * ((z_coordinate - point[1]) / direction[1])


def line_intersection(point, direction, line_points):
    """Intersect a directed line with an infinite line through two points."""
    line_direction = line_points[1] - line_points[0]
    denominator = cross_2d(direction, line_direction)
    if abs(denominator) <= 1e-12:
        raise ValueError("Thrust segment is parallel to an assembly interface.")
    distance = cross_2d(line_points[0] - point, line_direction) / denominator
    return point + distance * direction


def trace_thrust_line(load, insertion_x, geometry):
    """Trace the form diagram using the slopes supplied by the force diagram."""
    diagram = force_diagram(load, geometry.weights)
    point = np.asarray([insertion_x, 0.0], dtype=float)
    kinks = []
    for index, center in enumerate(geometry.centers):
        point = point_on_line_at_x(point, diagram.directions[index], center[0])
        kinks.append(point)
    kinks = np.asarray(kinks, dtype=float)

    crossings = []
    for index, interface in enumerate(geometry.interfaces):
        crossings.append(line_intersection(kinks[index], diagram.directions[index + 1], interface))
    crossings = np.asarray(crossings, dtype=float)
    right_anchor = point_on_line_at_z(kinks[-1], diagram.directions[-1], 0.0)
    return ThrustTrace(
        load=np.asarray(load, dtype=float),
        insertion_x=float(insertion_x),
        left_anchor=np.asarray([insertion_x, 0.0], dtype=float),
        kinks=kinks,
        crossings=crossings,
        right_anchor=right_anchor,
        directions=diagram.directions,
    )


def insertion_interval(load, geometry):
    """Return insertions for which every resultant crosses its finite joint."""
    zero_trace = trace_thrust_line(load, 0.0, geometry)
    unit_trace = trace_thrust_line(load, 1.0, geometry)
    lower = -np.inf
    upper = np.inf
    lower_source = None
    upper_source = None

    for interface_index, (interface, zero_point, unit_point) in enumerate(
        zip(geometry.interfaces, zero_trace.crossings, unit_trace.crossings)
    ):
        zero_parameter = interface_parameter(zero_point, interface)
        coefficient = interface_parameter(unit_point, interface) - zero_parameter
        if abs(coefficient) <= 1e-12:
            if zero_parameter < -CONTAINMENT_TOLERANCE or zero_parameter > 1.0 + CONTAINMENT_TOLERANCE:
                source = (interface_index, "finite joint")
                return InsertionInterval(1.0, 0.0, source, source)
            continue

        bounds = (
            ((-CONTAINMENT_TOLERANCE - zero_parameter) / coefficient, (interface_index, "intrados endpoint")),
            (
                (1.0 + CONTAINMENT_TOLERANCE - zero_parameter) / coefficient,
                (interface_index, "extrados endpoint"),
            ),
        )
        (candidate_lower, source_lower), (candidate_upper, source_upper) = sorted(bounds, key=lambda item: item[0])
        if candidate_lower > lower:
            lower = candidate_lower
            lower_source = source_lower
        if candidate_upper < upper:
            upper = candidate_upper
            upper_source = source_upper

    return InsertionInterval(lower, upper, lower_source, upper_source)


def point_halfspace_violation(point, halfspaces):
    """Return the largest positive convex-polygon halfspace residual."""
    return max(float(equation[:2].dot(point) + equation[2]) for equation in halfspaces)


def interface_parameter(point, interface):
    """Return the affine coordinate of a point on an intrados-to-extrados joint."""
    direction = interface[1] - interface[0]
    return float(np.dot(point - interface[0], direction) / np.dot(direction, direction))


def joint_checks(trace, geometry):
    """Return finite-joint and friction checks for every interface resultant."""
    checks = []
    for interface_index, (point, interface, force) in enumerate(
        zip(trace.crossings, geometry.interfaces, trace.directions[1:-1])
    ):
        parameter = interface_parameter(point, interface)
        overrun = max(0.0, -parameter, parameter - 1.0)
        tangent = interface[1] - interface[0]
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        if normal[0] < 0.0:
            normal *= -1.0
        normal_component = float(np.dot(force, normal))
        tangent_component = abs(float(np.dot(force, tangent)))
        if normal_component <= CONTAINMENT_TOLERANCE:
            friction_utilization = np.inf
        else:
            friction_utilization = tangent_component / (FRICTION_COEFFICIENT * normal_component)
        checks.append(
            JointCheck(
                interface_index=interface_index,
                parameter=parameter,
                overrun=overrun,
                normal_component=normal_component,
                friction_utilization=friction_utilization,
                point=point,
            )
        )
    return checks


def joint_violations(trace, geometry):
    """Return only finite-joint or friction violations; CoG kinks are unconstrained."""
    return [check for check in joint_checks(trace, geometry) if not check.valid]


def maximum_joint_overrun(checks):
    """Return the largest distance beyond a unit-length joint segment."""
    return max((check.overrun for check in checks), default=0.0)


def maximum_friction_utilization(checks):
    """Return the largest Coulomb friction-cone utilization."""
    return max((check.friction_utilization for check in checks), default=0.0)


def trace_is_joint_admissible(trace, geometry):
    """Return whether all physical joint resultants are compression/friction admissible."""
    return not joint_violations(trace, geometry)


def point_convex_margin(point, polygon):
    """Return positive distance from a point to the nearest convex-polygon boundary."""
    equations = ConvexHull(np.asarray(polygon, dtype=float)).equations
    return -float(np.max(equations[:, :2].dot(point) + equations[:, 2]))


def pressure_path_segments(trace, block_count):
    """Return illustrative pressure-point connections for all convex blocks."""
    segments = [(0, trace.left_anchor, trace.crossings[0])]
    for block_index in range(1, block_count - 1):
        segments.append((block_index, trace.crossings[block_index - 1], trace.crossings[block_index]))
    segments.append((block_count - 1, trace.crossings[-1], trace.right_anchor))
    return segments


def rbe_ray_radius(center, direction, halfspaces):
    """Return the distance from an interior point to a convex RBE boundary."""
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("RBE ray direction must be nonzero.")
    direction = direction / length

    residuals = halfspaces[:, :2].dot(center) + halfspaces[:, 2]
    if np.any(residuals > CONTAINMENT_TOLERANCE):
        raise ValueError("The configured thrust-family center lies outside the RBE polygon.")
    denominators = halfspaces[:, :2].dot(direction)
    forward = denominators > 1e-12
    if not np.any(forward):
        raise ValueError("Could not intersect a forward ray with the bounded RBE polygon.")
    radii = -residuals[forward] / denominators[forward]
    radius = float(np.min(radii))
    if radius < -CONTAINMENT_TOLERANCE:
        raise ValueError("Computed a negative distance to the RBE polygon.")
    return max(radius, 0.0)


def load_is_joint_admissible(load, geometry):
    """Return whether a load admits an insertion with valid joint resultants."""
    interval = insertion_interval(load, geometry)
    if not interval.feasible:
        return False
    trace = trace_thrust_line(load, interval.midpoint, geometry)
    return trace_is_joint_admissible(trace, geometry)


def assign_boundary_parameters(samples):
    """Assign normalized admissible-contour arclength to ordered family samples."""
    edge_lengths = np.asarray(
        [
            np.linalg.norm(samples[(index + 1) % len(samples)].admissible_load - sample.admissible_load)
            for index, sample in enumerate(samples)
        ],
        dtype=float,
    )
    perimeter = float(np.sum(edge_lengths))
    if perimeter <= 1e-12:
        raise ValueError("The joint-admissible thrust contour has zero perimeter.")
    travelled = 0.0
    for index, sample in enumerate(samples):
        sample.boundary_parameter = travelled / perimeter
        travelled += edge_lengths[index]


def build_admissible_contour(rbe_boundary, geometry, num_directions=NUM_CONTOUR_DIRECTIONS):
    """Reconstruct every radial sample directly on the RBE boundary."""
    center = np.asarray([THRUST_CENTER_FX, float(np.mean(rbe_boundary[:, 1]))], dtype=float)
    if not load_is_joint_admissible(center, geometry):
        raise ValueError("The configured thrust-family center has no admissible insertion interval.")

    halfspaces = ConvexHull(np.asarray(rbe_boundary, dtype=float)).equations
    samples = []
    for angle in np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False):
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        rbe_radius = rbe_ray_radius(center, direction, halfspaces)
        admissible_radius = rbe_radius
        admissible_load = center + rbe_radius * direction
        interval = insertion_interval(admissible_load, geometry)
        if not interval.feasible:
            raise ValueError("An RBE-boundary load unexpectedly has an empty insertion interval.")
        trace = trace_thrust_line(admissible_load, interval.midpoint, geometry)
        violations = joint_violations(trace, geometry)
        if violations:
            raise ValueError("An RBE-boundary trace violates a physical joint: {}.".format(violations[0]))
        samples.append(
            FamilySample(
                boundary_parameter=0.0,
                ray_angle=float(angle),
                rbe_radius=rbe_radius,
                admissible_radius=admissible_radius,
                admissible_load=admissible_load,
                interval=interval,
                trace=trace,
            )
        )
    assign_boundary_parameters(samples)
    return center, samples


def plotted_family(family, stride=PLOT_SAMPLE_STRIDE):
    """Return an evenly distributed, readable subset of admissible thrust lines."""
    return family[::stride]


def projected_insertion(interval, requested_x):
    """Project an insertion coordinate into a feasible interval with a small guard."""
    if not interval.feasible:
        raise ValueError("Cannot project an insertion into an empty interval.")
    if interval.upper - interval.lower <= 2.0 * FIT_INSERTION_GUARD:
        return interval.midpoint
    return min(
        max(requested_x, interval.lower + FIT_INSERTION_GUARD),
        interval.upper - FIT_INSERTION_GUARD,
    )


def fit_anchor_load(label, supplied_anchor_load, geometry):
    """Keep a supplied load fixed when it admits a mesh-weight insertion."""
    supplied_anchor_load = np.asarray(supplied_anchor_load, dtype=float)
    repository_load = np.asarray([supplied_anchor_load[0], -supplied_anchor_load[1]], dtype=float)
    interval = insertion_interval(repository_load, geometry)
    if not interval.feasible:
        raise ValueError("Supplied case {} has no mesh-weight insertion interval.".format(label))
    trace = trace_thrust_line(repository_load, interval.midpoint, geometry)
    if maximum_friction_utilization(joint_checks(trace, geometry)) > 1.0 + CONTAINMENT_TOLERANCE:
        raise ValueError("Supplied case {} violates friction for every insertion.".format(label))
    return supplied_anchor_load.copy()


def supplied_case_diagnostics(geometry, block_thickness, rbe_boundary):
    """Reconstruct exact supplied cases and nearest joint-admissible fits."""
    diagnostics = []
    for label, supplied_anchor_load, supplied_insertion_fraction in REFERENCE_CASES:
        supplied_anchor_load = np.asarray(supplied_anchor_load, dtype=float)
        repository_load = np.asarray([supplied_anchor_load[0], -supplied_anchor_load[1]], dtype=float)
        insertion_x = geometry.left_support_x + supplied_insertion_fraction * block_thickness
        supplied_trace = trace_thrust_line(repository_load, insertion_x, geometry)
        supplied_checks = joint_checks(supplied_trace, geometry)
        rbe_margin = point_convex_margin(repository_load, rbe_boundary)
        if rbe_margin <= CONTAINMENT_TOLERANCE:
            raise ValueError("Supplied case {} lies outside the computed RBE region.".format(label))

        fitted_anchor_load = fit_anchor_load(label, supplied_anchor_load, geometry)
        fitted_repository_load = np.asarray([fitted_anchor_load[0], -fitted_anchor_load[1]], dtype=float)
        fitted_interval = insertion_interval(fitted_repository_load, geometry)
        requested_x = geometry.left_support_x + supplied_insertion_fraction * block_thickness
        fitted_insertion_x = projected_insertion(fitted_interval, requested_x)
        fitted_insertion_fraction = (fitted_insertion_x - geometry.left_support_x) / block_thickness
        fitted_trace = trace_thrust_line(fitted_repository_load, fitted_insertion_x, geometry)
        fitted_checks = joint_checks(fitted_trace, geometry)
        if any(not check.valid for check in fitted_checks):
            raise ValueError("Fitted case {} is not joint-admissible.".format(label))

        diagnostics.append(
            SuppliedCaseDiagnostic(
                label=label,
                supplied_anchor_load=supplied_anchor_load,
                supplied_insertion_fraction=supplied_insertion_fraction,
                supplied_trace=supplied_trace,
                supplied_checks=supplied_checks,
                fitted_anchor_load=fitted_anchor_load,
                fitted_insertion_fraction=fitted_insertion_fraction,
                fitted_trace=fitted_trace,
                fitted_checks=fitted_checks,
                rbe_margin=rbe_margin,
                fitted_rbe_margin=point_convex_margin(fitted_repository_load, rbe_boundary),
                fitted_interval=fitted_interval,
            )
        )
    return diagnostics


def nearest_admissible_sample(family, anchor_load):
    """Return the admissible-contour sample nearest an anchor-load point."""
    anchor_load = np.asarray(anchor_load, dtype=float)
    distances = [
        np.linalg.norm(np.asarray([sample.admissible_load[0], -sample.admissible_load[1]]) - anchor_load)
        for sample in family
    ]
    return family[int(np.argmin(distances))]


def closed_points(points):
    """Return a coordinate array with its first point appended."""
    return np.vstack([points, points[0]])


def anchor_plot_coordinates(loads):
    """Map repository loads to the left-anchor reaction convention."""
    loads = np.asarray(loads, dtype=float)
    return np.column_stack([loads[:, 0], -loads[:, 1]])


def trace_polyline(trace):
    """Return the plotted form-diagram polyline coordinates."""
    return np.vstack([trace.left_anchor, trace.kinks, trace.right_anchor])


def diagnostic_block_indices(geometry, block_indices=None):
    """Return normalized block indices for a full or cropped diagnostic plot."""
    if block_indices is None:
        return list(range(len(geometry.block_polygons)))
    return sorted(set(block_indices))


def plot_diagnostic_blocks(axes, geometry, block_indices, annotate_blocks=False):
    """Plot selected arch blocks, hatching the moment-exempt end blocks."""
    last_block = len(geometry.block_polygons) - 1
    for block_index in block_indices:
        polygon = geometry.block_polygons[block_index]
        end_block = block_index in (0, last_block)
        axes.add_patch(
            PolygonPatch(
                polygon,
                closed=True,
                facecolor="#F3C0C0" if end_block else "#E69F9F",
                edgecolor="#A00000",
                linewidth=0.8,
                alpha=0.44,
                hatch="//" if end_block else None,
            )
        )
        if annotate_blocks:
            center = geometry.centers[block_index]
            axes.text(center[0], center[1], str(block_index), color="#7F0000", fontsize=8, ha="center", va="center")


def block_action_segments(trace, block_index, block_count):
    """Return incoming and outgoing lines of action for one block."""
    left_point = trace.left_anchor if block_index == 0 else trace.crossings[block_index - 1]
    right_point = trace.right_anchor if block_index == block_count - 1 else trace.crossings[block_index]
    return ((left_point, trace.kinks[block_index]), (trace.kinks[block_index], right_point))


def plot_action_segments(axes, trace, geometry, block_indices, color, linewidth, alpha, linestyle="-"):
    """Plot the force-direction construction whose rays meet at CoG load lines."""
    block_count = len(geometry.block_polygons)
    for block_index in block_indices:
        for start, end in block_action_segments(trace, block_index, block_count):
            axes.plot(
                *np.vstack([start, end]).T,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=5,
            )


def plot_pressure_path(axes, trace, geometry, block_indices, color, linewidth, alpha, linestyle="-"):
    """Plot the pressure-point path; only end-block connections are moment-exempt."""
    last_block = len(geometry.block_polygons) - 1
    selected = set(block_indices)
    for block_index, start, end in pressure_path_segments(trace, len(geometry.block_polygons)):
        if block_index not in selected:
            continue
        end_block = block_index in (0, last_block)
        axes.plot(
            *np.vstack([start, end]).T,
            color=EXEMPT_COLOR if end_block else color,
            linewidth=linewidth if not end_block else max(1.0, 0.7 * linewidth),
            alpha=alpha,
            linestyle="--" if end_block else linestyle,
            zorder=7,
        )


def selected_interface_indices(block_indices, block_count):
    """Return the physical interfaces adjacent to selected blocks."""
    indices = set()
    for block_index in block_indices:
        if block_index > 0:
            indices.add(block_index - 1)
        if block_index < block_count - 1:
            indices.add(block_index)
    return sorted(indices)


def plot_trace_points(axes, trace, checks, geometry, block_indices, joint_color, alpha=1.0):
    """Plot neutral CoG concurrency points and physically checked joint points."""
    for block_index in block_indices:
        point = trace.kinks[block_index]
        axes.scatter(
            [point[0]],
            [point[1]],
            color=POINT_COLOR,
            edgecolors="none",
            s=20,
            alpha=alpha,
            zorder=9,
        )

    by_interface = {check.interface_index: check for check in checks}
    for interface_index in selected_interface_indices(block_indices, len(geometry.block_polygons)):
        check = by_interface[interface_index]
        axes.scatter(
            [check.point[0]],
            [check.point[1]],
            color=joint_color,
            edgecolors=OUTSIDE_COLOR if not check.valid else "none",
            linewidths=2.0 if not check.valid else 0.0,
            marker="D",
            s=34 if not check.valid else 22,
            alpha=alpha,
            zorder=10,
        )


def set_diagnostic_limits(axes, geometry, block_indices, traces, full_arch):
    """Set readable limits for a full arch or selected-block zoom."""
    if full_arch:
        anchors = [point for trace in traces for point in (trace.left_anchor, trace.right_anchor)]
        points = np.vstack([polygon for polygon in geometry.block_polygons] + anchors)
    else:
        points = np.vstack([geometry.block_polygons[index] for index in block_indices])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    spans = np.maximum(upper - lower, 0.25)
    padding = np.asarray([0.08 * spans[0], 0.12 * spans[1]])
    axes.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    axes.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))


def plot_diagnostic_case(axes, diagnostic, geometry, block_indices=None, title=None, summary=False):
    """Plot exact and fitted reconstructions with only physical joints judged."""
    indices = diagnostic_block_indices(geometry, block_indices)
    full_arch = block_indices is None
    plot_diagnostic_blocks(axes, geometry, indices, annotate_blocks=not full_arch)
    exact_matches_fit = (
        np.allclose(
            diagnostic.supplied_anchor_load,
            diagnostic.fitted_anchor_load,
            atol=1e-12,
            rtol=0.0,
        )
        and abs(diagnostic.supplied_insertion_fraction - diagnostic.fitted_insertion_fraction) <= 1e-12
    )
    if not exact_matches_fit:
        plot_action_segments(axes, diagnostic.supplied_trace, geometry, indices, EXACT_COLOR, 0.9, 0.55, ":")
        plot_pressure_path(axes, diagnostic.supplied_trace, geometry, indices, EXACT_COLOR, 1.1, 0.65, ":")
        plot_trace_points(
            axes,
            diagnostic.supplied_trace,
            diagnostic.supplied_checks,
            geometry,
            indices,
            EXACT_COLOR,
            alpha=0.75,
        )
    plot_action_segments(axes, diagnostic.fitted_trace, geometry, indices, INSIDE_COLOR, 1.0, 0.8)
    plot_pressure_path(axes, diagnostic.fitted_trace, geometry, indices, INSIDE_COLOR, 2.5, 0.95)
    plot_trace_points(
        axes,
        diagnostic.fitted_trace,
        diagnostic.fitted_checks,
        geometry,
        indices,
        INSIDE_COLOR,
    )
    axes.axhline(0.0, color="0.35", linewidth=0.6)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.18)
    axes.set_xlabel("x")
    axes.set_ylabel("z")
    set_diagnostic_limits(
        axes,
        geometry,
        indices,
        (diagnostic.supplied_trace, diagnostic.fitted_trace),
        full_arch,
    )

    if title is None:
        title = "{} supplied=({:.2f}, {:.2f}), insertion={:+.0%}".format(
            diagnostic.label,
            diagnostic.supplied_anchor_load[0],
            diagnostic.supplied_anchor_load[1],
            diagnostic.supplied_insertion_fraction,
        )
    axes.set_title(title, fontsize=10)
    if summary:
        load_delta = np.linalg.norm(diagnostic.fitted_anchor_load - diagnostic.supplied_anchor_load)
        insertion_delta = diagnostic.fitted_insertion_fraction - diagnostic.supplied_insertion_fraction
        exact_overrun = maximum_joint_overrun(diagnostic.supplied_checks)
        exact_friction = maximum_friction_utilization(diagnostic.supplied_checks)
        exact_status = "VALID" if all(check.valid for check in diagnostic.supplied_checks) else "rounded joint miss"
        axes.text(
            0.02,
            0.03,
            "RBE safe: margin {:.4f} | exact {}\n"
            "joint overrun {:.5f} | max friction {:.3f}\n"
            "fit ΔF={:.5f}, Δinsertion={:+.3%}".format(
                diagnostic.rbe_margin,
                exact_status,
                exact_overrun,
                exact_friction,
                load_delta,
                insertion_delta,
            ),
            transform=axes.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
            zorder=12,
        )


def plot_fit_summary(axes, diagnostics):
    """Summarize exact-to-fitted changes without hiding supplied values."""
    axes.axis("off")
    axes.set_title("Mesh-weight insertion corrections", fontsize=10)
    lines = ["case     fixed supplied load   fitted insertion    governing exact joint"]
    for diagnostic in diagnostics:
        misses = [check for check in diagnostic.supplied_checks if check.overrun > CONTAINMENT_TOLERANCE]
        if misses:
            governing = max(misses, key=lambda check: check.overrun)
            governing_text = "J{:02d}, overrun {:.5f}".format(
                governing.interface_index,
                governing.overrun,
            )
        else:
            governing_text = "none (exact valid)"
        lines.append(
            "{:<3}  ({:>8.5f}, {:>8.5f})   {:+8.3%}       {}".format(
                diagnostic.label,
                diagnostic.fitted_anchor_load[0],
                diagnostic.fitted_anchor_load[1],
                diagnostic.fitted_insertion_fraction,
                governing_text,
            )
        )
    axes.text(
        0.02,
        0.88,
        "\n".join(lines),
        transform=axes.transAxes,
        family="monospace",
        fontsize=8.5,
        va="top",
    )
    axes.text(
        0.02,
        0.20,
        "The supplied load is held fixed; only insertion is projected.\n"
        "CoG circles are unconstrained concurrency points.",
        transform=axes.transAxes,
        fontsize=9,
        va="top",
    )


def supplied_case_legend():
    """Return shared legend handles for the supplied-case diagnostic."""
    return [
        Line2D([0], [0], color=EXACT_COLOR, linewidth=1.2, linestyle=":", label="exact supplied construction"),
        Line2D([0], [0], color=INSIDE_COLOR, linewidth=1.0, label="fitted force-action segments"),
        Line2D([0], [0], color=INSIDE_COLOR, linewidth=2.5, label="joint-pressure path"),
        Line2D([0], [0], color=EXEMPT_COLOR, linewidth=1.4, linestyle="--", label="end-block connection (exempt)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POINT_COLOR, label="CoG concurrency point"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=INSIDE_COLOR, label="valid joint pressure point"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=EXACT_COLOR,
            markeredgecolor=OUTSIDE_COLOR,
            markeredgewidth=2.0,
            label="exact joint miss (red ring)",
        ),
    ]


def plot_supplied_case_diagnostics(diagnostics, geometry):
    """Render exact/fitted cases and finite-joint zooms."""
    plt.rcParams["svg.fonttype"] = "none"
    by_label = {diagnostic.label: diagnostic for diagnostic in diagnostics}
    figure = plt.figure(figsize=(18.0, 15.5))
    grid = figure.add_gridspec(4, 4, height_ratios=(1.0, 1.0, 0.62, 0.62), hspace=0.58, wspace=0.28)

    full_axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
    )
    for axes, diagnostic in zip(full_axes, diagnostics):
        plot_diagnostic_case(axes, diagnostic, geometry, summary=True)

    zooms = (
        (figure.add_subplot(grid[2, 0]), by_label["R1"], (0, 1), "R1: interface 0 exact overrun"),
        (figure.add_subplot(grid[2, 1]), by_label["R1"], (18, 19), "R1: interface 18 exact overrun"),
        (figure.add_subplot(grid[2, 2]), by_label["R2"], (10, 11), "R2 top: interface 10 exact overrun"),
        (figure.add_subplot(grid[2, 3]), by_label["R4"], (8, 9), "R4 bottom: interface 8 exact overrun"),
    )
    for axes, diagnostic, blocks, title in zooms:
        plot_diagnostic_case(axes, diagnostic, geometry, block_indices=blocks, title=title)

    plot_diagnostic_case(
        figure.add_subplot(grid[3, 0:2]),
        by_label["R3"],
        geometry,
        block_indices=(9, 10),
        title="R3: small interface 9 mesh-weight rounding miss",
    )
    plot_fit_summary(figure.add_subplot(grid[3, 2:4]), diagnostics)
    figure.suptitle("Supplied cases: finite-joint pressure validation and nearest admissible fits", y=0.985)
    figure.legend(handles=supplied_case_legend(), loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=4, fontsize=8)
    figure.subplots_adjust(top=0.885, bottom=0.055, left=0.055, right=0.985)
    return figure


def plot_load_panel(axes, rbe_boundary, family, plot_samples, diagnostics, representative, colormap):
    """Plot the RBE region, joint-admissible contour, and supplied/fitted cases."""
    rbe_anchor = anchor_plot_coordinates(rbe_boundary)
    admissible_anchor = anchor_plot_coordinates([sample.admissible_load for sample in family])
    colors = [colormap(sample.boundary_parameter) for sample in family]
    axes.fill(rbe_anchor[:, 0], rbe_anchor[:, 1], color="0.75", alpha=0.25, label="RBE feasible region")
    axes.plot(*closed_points(rbe_anchor).T, color="0.35", linewidth=1.2)

    segments = np.asarray(
        [
            [admissible_anchor[index], admissible_anchor[(index + 1) % len(admissible_anchor)]]
            for index in range(len(family))
        ]
    )
    axes.add_collection(LineCollection(segments, colors=colors, linewidths=2.0))
    plotted_anchor = anchor_plot_coordinates([sample.admissible_load for sample in plot_samples])
    axes.scatter(
        plotted_anchor[:, 0],
        plotted_anchor[:, 1],
        c=[colormap(sample.boundary_parameter) for sample in plot_samples],
        s=13,
        edgecolors="none",
        label="verified pressure-path boundary",
        zorder=4,
    )

    supplied = np.asarray([diagnostic.supplied_anchor_load for diagnostic in diagnostics], dtype=float)
    fitted = np.asarray([diagnostic.fitted_anchor_load for diagnostic in diagnostics], dtype=float)
    axes.scatter(supplied[:, 0], supplied[:, 1], color="black", marker="x", s=38, label="supplied loads")
    axes.scatter(
        fitted[:, 0],
        fitted[:, 1],
        facecolors="none",
        edgecolors=INSIDE_COLOR,
        marker="o",
        s=42,
        label="same loads with fitted insertions",
    )
    for diagnostic in diagnostics:
        axes.plot(
            [diagnostic.supplied_anchor_load[0], diagnostic.fitted_anchor_load[0]],
            [diagnostic.supplied_anchor_load[1], diagnostic.fitted_anchor_load[1]],
            color=EXACT_COLOR,
            linewidth=0.7,
            linestyle=":",
        )
        axes.annotate(
            "{} ({:+.0%})".format(diagnostic.label, diagnostic.supplied_insertion_fraction),
            xy=diagnostic.supplied_anchor_load,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    representative_anchor = np.asarray([representative.admissible_load[0], -representative.admissible_load[1]])
    axes.scatter(
        [representative_anchor[0]],
        [representative_anchor[1]],
        facecolors="none",
        edgecolors=[colormap(representative.boundary_parameter)],
        linewidths=1.8,
        s=75,
        label="representative admissible sample",
        zorder=6,
    )
    axes.set_xlabel("left-anchor Fx")
    axes.set_ylabel("left-anchor Fz (arch on anchor)")
    axes.set_title("RBE and coincident pressure-path boundaries")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.25)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8)


def plot_arch_panel(axes, geometry, plot_samples, representative, colormap):
    """Plot admissible force-action constructions and one pressure-point path."""
    for index, polygon in enumerate(geometry.block_polygons):
        end_block = index in (0, len(geometry.block_polygons) - 1)
        axes.add_patch(
            PolygonPatch(
                polygon,
                closed=True,
                facecolor="#E69F9F" if not end_block else "#F3C0C0",
                edgecolor="#A00000",
                linewidth=0.7,
                alpha=0.52,
                hatch="//" if end_block else None,
            )
        )

    for sample in plot_samples:
        points = trace_polyline(sample.trace)
        axes.plot(
            points[:, 0],
            points[:, 1],
            color=colormap(sample.boundary_parameter),
            linewidth=0.72,
            alpha=0.48,
        )

    representative_color = colormap(representative.boundary_parameter)
    representative_points = trace_polyline(representative.trace)
    axes.plot(
        representative_points[:, 0],
        representative_points[:, 1],
        color=representative_color,
        linewidth=1.7,
        label="representative force-action construction",
        zorder=6,
    )
    for block_index, start, end in pressure_path_segments(
        representative.trace,
        len(geometry.block_polygons),
    ):
        end_block = block_index in (0, len(geometry.block_polygons) - 1)
        axes.plot(
            *np.vstack([start, end]).T,
            color=EXEMPT_COLOR if end_block else representative_color,
            linewidth=1.1 if end_block else 2.2,
            linestyle="--" if end_block else "-",
            alpha=0.95,
            zorder=7,
        )
    axes.scatter(
        representative.trace.kinks[:, 0],
        representative.trace.kinks[:, 1],
        color="#28BCE0",
        s=15,
        edgecolors="none",
        zorder=7,
    )
    axes.scatter(
        representative.trace.crossings[:, 0],
        representative.trace.crossings[:, 1],
        color=representative_color,
        marker="D",
        s=14,
        edgecolors="none",
        zorder=8,
        label="joint pressure points",
    )
    axes.scatter(
        geometry.centers[:, 0],
        geometry.centers[:, 1],
        color="#8B0000",
        marker="x",
        s=18,
        zorder=4,
    )
    axes.scatter(
        [representative.trace.left_anchor[0]],
        [representative.trace.left_anchor[1]],
        color=representative_color,
        marker="D",
        s=30,
        zorder=7,
    )

    axes.axhline(0.0, color="0.35", linewidth=0.7)
    axes.set_xlabel("x")
    axes.set_ylabel("z")
    axes.set_title("Graphic-statics thrust-line family")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.18)
    axes.legend(loc="lower center", fontsize=8)


def plot_force_panel(axes, representative, insertion_fraction, color, weights):
    """Plot the force diagram for one retained joint-admissible sample."""
    diagram = force_diagram(representative.admissible_load, weights)
    axes.plot(diagram.nodes[:, 0], diagram.nodes[:, 1], color="#A00000", linewidth=1.0)
    for node in diagram.nodes:
        axes.plot(
            [diagram.pole[0], node[0]],
            [diagram.pole[1], node[1]],
            color=color,
            linewidth=0.55,
            alpha=0.75,
        )
    axes.scatter(diagram.nodes[:, 0], diagram.nodes[:, 1], color="#A00000", marker="x", s=15)
    axes.scatter([diagram.pole[0]], [diagram.pole[1]], color=color, marker="D", s=32, label="force pole")

    slope_indices = (0, 10, 20)
    slope_text = []
    for index in slope_indices:
        direction = diagram.directions[index]
        slope_text.append("m{}={:.3f}".format(index, direction[1] / direction[0]))
    axes.text(
        0.03,
        0.03,
        "anchor = ({:.3f}, {:.3f})\ninsertion = {:+.1%}\nmesh W range = [{:.6f}, {:.6f}]\n{}".format(
            representative.admissible_load[0],
            -representative.admissible_load[1],
            insertion_fraction,
            np.min(weights),
            np.max(weights),
            ", ".join(slope_text),
        ),
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    axes.set_xlabel("force Fx")
    axes.set_ylabel("force Fz")
    axes.set_title("Representative admissible force diagram")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.25)
    axes.legend(loc="upper left", fontsize=8)


def plot_thrust_family(rbe_boundary, geometry, family, diagnostics, representative):
    """Render the paired load, form, and force diagrams."""
    plt.rcParams["svg.fonttype"] = "none"
    figure = plt.figure(figsize=(17.0, 6.8))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.85, 0.88), wspace=0.28)
    load_axes = figure.add_subplot(grid[0, 0])
    arch_axes = figure.add_subplot(grid[0, 1])
    force_axes = figure.add_subplot(grid[0, 2])

    colormap = plt.get_cmap("turbo")
    plot_samples = plotted_family(family)
    representative_color = colormap(representative.boundary_parameter)
    insertion_fraction = representative.trace.insertion_x - geometry.left_support_x
    plot_load_panel(load_axes, rbe_boundary, family, plot_samples, diagnostics, representative, colormap)
    plot_arch_panel(arch_axes, geometry, plot_samples, representative, colormap)
    plot_force_panel(force_axes, representative, insertion_fraction, representative_color, geometry.weights)
    figure.suptitle("Example 19-4-1: mesh-weight pressure paths on the RBE boundary")
    figure.subplots_adjust(top=0.89, bottom=0.19, left=0.055, right=0.985)
    return figure


def save_svg(figure, path):
    """Save one Matplotlib figure as SVG and report the path."""
    figure.savefig(path, format="svg", bbox_inches="tight")
    print("saved SVG: {}".format(path))


def report_family(center, family, diagnostics, representative, geometry):
    """Print numerical diagnostics for the admissible family and supplied-case fits."""
    radial_ratios = np.asarray([sample.admissible_radius / sample.rbe_radius for sample in family], dtype=float)
    widths = np.asarray([sample.interval.upper - sample.interval.lower for sample in family], dtype=float)
    anchor_loads = anchor_plot_coordinates([sample.admissible_load for sample in family])
    print(
        "mesh-derived block weights: range [{:.9g}, {:.9g}], total {:.9g}".format(
            float(np.min(geometry.weights)),
            float(np.max(geometry.weights)),
            float(np.sum(geometry.weights)),
        )
    )
    print("joint-admissible center load: Fx={:.9g}, Fz={:.9g}".format(center[0], center[1]))
    print(
        "joint-admissible contour: {} directions, {} plotted lines, radial ratio range [{:.6g}, {:.6g}]".format(
            len(family),
            len(plotted_family(family)),
            float(np.min(radial_ratios)),
            float(np.max(radial_ratios)),
        )
    )
    print(
        "admissible anchor extents: Fx=[{:.9g}, {:.9g}], Fz=[{:.9g}, {:.9g}]".format(
            float(np.min(anchor_loads[:, 0])),
            float(np.max(anchor_loads[:, 0])),
            float(np.min(anchor_loads[:, 1])),
            float(np.max(anchor_loads[:, 1])),
        )
    )
    print(
        "admissible insertion-width range: [{:.6g}, {:.6g}]".format(
            float(np.min(widths)),
            float(np.max(widths)),
        )
    )
    print(
        "representative admissible sample: anchor=({:.6g}, {:.6g}), insertion={:+.3%}".format(
            representative.admissible_load[0],
            -representative.admissible_load[1],
            representative.trace.insertion_x - geometry.left_support_x,
        )
    )
    print("supplied cases and nearest joint-admissible fits:")
    for diagnostic in diagnostics:
        governing = max(diagnostic.supplied_checks, key=lambda check: check.overrun)
        supplied_valid = all(check.valid for check in diagnostic.supplied_checks)
        load_delta = np.linalg.norm(diagnostic.fitted_anchor_load - diagnostic.supplied_anchor_load)
        insertion_delta = diagnostic.fitted_insertion_fraction - diagnostic.supplied_insertion_fraction
        print(
            "  {} supplied=({:.6g}, {:.6g}), insertion={:+.3%}: RBE margin={:.6g}, "
            "joint status={}, max overrun={:.6g} at interface {}, max friction={:.6g}".format(
                diagnostic.label,
                diagnostic.supplied_anchor_load[0],
                diagnostic.supplied_anchor_load[1],
                diagnostic.supplied_insertion_fraction,
                diagnostic.rbe_margin,
                "valid" if supplied_valid else "rounded joint miss",
                governing.overrun,
                governing.interface_index,
                maximum_friction_utilization(diagnostic.supplied_checks),
            )
        )
        print(
            "    fitted=({:.9g}, {:.9g}), insertion={:+.6%}, dF={:.6g}, dinsertion={:+.6%}, "
            "max friction={:.6g}".format(
                diagnostic.fitted_anchor_load[0],
                diagnostic.fitted_anchor_load[1],
                diagnostic.fitted_insertion_fraction,
                load_delta,
                insertion_delta,
                maximum_friction_utilization(diagnostic.fitted_checks),
            )
        )


def main():
    """Solve the RBE boundary, construct the thrust family, and save the SVG."""
    base = load_example_19_4()
    assembly = base.build_full_arch()
    load_node, load_dofs, application_points = base.load_setup(assembly)
    problem = base.hidden_load_problem(assembly, load_node, load_dofs, application_points, penalty=False)
    geometry = build_arch_geometry(assembly)
    rbe_boundary = solve_primal_rbe_boundary(base, problem)
    center, family = build_admissible_contour(rbe_boundary, geometry)
    diagnostics = supplied_case_diagnostics(geometry, base.THICKNESS, rbe_boundary)
    representative = nearest_admissible_sample(family, diagnostics[0].fitted_anchor_load)
    report_family(center, family, diagnostics, representative, geometry)

    figure = plot_thrust_family(rbe_boundary, geometry, family, diagnostics, representative)
    diagnostic_figure = plot_supplied_case_diagnostics(diagnostics, geometry)
    save_svg(figure, OUTPUT_SVG)
    save_svg(diagnostic_figure, DIAGNOSTIC_SVG)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
        plt.close(diagnostic_figure)


if __name__ == "__main__":
    main()
