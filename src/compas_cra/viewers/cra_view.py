"""CRA view style using compas_view2"""

import os
import sys
import types
from itertools import tee
from math import sqrt

import numpy as np
from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import Rotation
from compas.geometry import Translation
from compas.geometry import is_coplanar

app = None
Collection = None
Arrow = None


class _CompasView2RobotModel:
    pass


class _CompasView2Geometry:
    @staticmethod
    def _get_item_meshes(item):
        raise NotImplementedError("Robot visualization is not supported by the CRA compas_view2 compatibility shim.")


def _flatten(items):
    for item in items:
        yield from item


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))


def _gif_from_images(files, gif_path, fps=10, delete_files=False):
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "Recording GIFs with compas_view2 requires imageio. Install it in the active environment."
        ) from e

    images = [imageio.imread(filename) for filename in files]
    duration = 1.0 / fps if fps else 0.1
    imageio.mimsave(gif_path, images, duration=duration)

    if delete_files:
        for filename in files:
            try:
                os.remove(filename)
            except OSError:
                pass


def _install_compas_view2_compatibility():
    """Install small COMPAS 1 compatibility shims required by compas_view2 0.11."""
    import compas.utilities as utilities

    if not hasattr(utilities, "flatten"):
        utilities.flatten = _flatten
    if not hasattr(utilities, "pairwise"):
        utilities.pairwise = _pairwise
    if not hasattr(utilities, "gif_from_images"):
        utilities.gif_from_images = _gif_from_images

    if "compas.robots" not in sys.modules:
        robots = types.ModuleType("compas.robots")
        robots.RobotModel = _CompasView2RobotModel
        robots.Geometry = _CompasView2Geometry
        sys.modules["compas.robots"] = robots


def _preload_qtpy():
    """Load QtPy before compas_view2 forces PySide2 in its package init."""
    if "qtpy" in sys.modules:
        return

    os.environ.setdefault("QT_API", "pyqt5")
    try:
        import qtpy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The compas_view2 viewer requires qtpy and a Qt binding. Install PyQt5 and qtpy in the active environment."
        ) from e


def _patch_compas_view2_app(view2_app):
    if getattr(view2_app.App, "_compas_cra_resize_patch", False):
        return

    def resize(self, width, height):
        width = int(width)
        height = int(height)
        self.window.resize(width, height)
        desktop = self._app.desktop()
        rect = desktop.availableGeometry()
        x = int(0.5 * (rect.width() - width))
        y = int(0.5 * (rect.height() - height))
        self.window.setGeometry(x, y, width, height)

    view2_app.App.resize = resize
    view2_app.App._compas_cra_resize_patch = True


def _patch_compas_view2_controller(view2_app):
    controller = view2_app.Controller
    if getattr(controller, "_compas_cra_wheel_patch", False):
        return

    def wheel_action(self, event):
        if hasattr(event, "angleDelta"):
            steps = event.angleDelta().y() / 120
        else:
            steps = event.delta() / 120

        self.app.view.camera.zoom(steps)
        self.app.view.update()

    controller.wheel_action = wheel_action
    controller._compas_cra_wheel_patch = True


def _patch_compas_view2_arrow(view2_arrow):
    if getattr(view2_arrow, "_compas_cra_arrow_patch", False):
        return

    def to_vertices_and_faces(self, u=4):
        if u < 3:
            raise ValueError("The value for u should be u > 3.")

        from compas.datastructures import Mesh
        from compas.geometry import Cone
        from compas.geometry import Cylinder
        from compas.geometry import Frame
        from compas.geometry import Plane

        head_position = self.position + self.direction * (1 - self.head_portion)
        head_frame = Frame.from_plane(Plane(head_position, self.direction))
        head = Cone(
            self.head_width * self.direction.length,
            self.direction.length * self.head_portion,
            frame=head_frame,
        )
        vertices, faces = head.to_vertices_and_faces(u=u)
        head_mesh = Mesh.from_vertices_and_faces(vertices, faces)

        body_center = self.position + self.direction * (1 - self.head_portion) / 2
        body_frame = Frame.from_plane(Plane(body_center, self.direction))
        body = Cylinder(
            self.body_width * self.direction.length,
            self.direction.length * (1 - self.head_portion),
            frame=body_frame,
        )
        vertices, faces = body.to_vertices_and_faces(u=u)
        body_mesh = Mesh.from_vertices_and_faces(vertices, faces)

        body_mesh.join(head_mesh)
        return body_mesh.to_vertices_and_faces()

    view2_arrow.to_vertices_and_faces = to_vertices_and_faces
    view2_arrow._compas_cra_arrow_patch = True


def _load_compas_view2():
    """Load compas_view2 with compatibility for the current COMPAS 2 environment."""
    global app
    global Collection
    global Arrow

    if app is not None and Collection is not None and Arrow is not None:
        return app, Collection, Arrow

    _preload_qtpy()
    _install_compas_view2_compatibility()

    import builtins

    robot_model_in_builtins = hasattr(builtins, "RobotModel")
    if not robot_model_in_builtins:
        builtins.RobotModel = _CompasView2RobotModel

    try:
        from compas_view2 import app as view2_app
        from compas_view2.collections import Collection as view2_collection
        from compas_view2.shapes import Arrow as view2_arrow
    except ModuleNotFoundError as e:
        if e.name == "freetype":
            raise ImportError(
                "The compas_view2 viewer imports freetype at startup. "
                "Install freetype-py in the active environment, for example: "
                "python -m pip install freetype-py"
            ) from e
        raise ImportError(
            "Could not import compas_view2. Install compas_view2 and its viewer runtime dependencies "
            "in the active environment."
        ) from e
    except ImportError as e:
        raise ImportError(
            "Could not import compas_view2 with the COMPAS 2 compatibility shims. "
            "Check that compas_view2, qtpy, PyQt5, PyOpenGL, matplotlib, and freetype-py are installed."
        ) from e
    finally:
        if not robot_model_in_builtins:
            del builtins.RobotModel

    _patch_compas_view2_app(view2_app)
    _patch_compas_view2_controller(view2_app)
    _patch_compas_view2_arrow(view2_arrow)

    app = view2_app
    Collection = view2_collection
    Arrow = view2_arrow
    return app, Collection, Arrow


def draw_blocks(assembly, viewer, edge=True, tol=0.0):
    _load_compas_view2()

    supports = []
    blocks = []
    supportedges = []
    blockedges = []
    for node in assembly.graph.nodes():
        block = assembly.graph.node_attribute(node, "block")
        if assembly.graph.node_attribute(node, "is_support"):
            supports.append(block)
        else:
            blocks.append(block)
        if not edge:
            continue
        for edge in block.edges():
            if tol != 0.0:
                fkeys = block.edge_faces(edge)
                ps = [
                    block.face_center(fkeys[0]),
                    block.face_center(fkeys[1]),
                    *block.edge_coordinates(edge),
                ]

                if is_coplanar(ps, tol=tol):
                    continue
            if assembly.graph.node_attribute(node, "is_support"):
                supportedges.append(Line(*block.edge_coordinates(edge)))
            else:
                blockedges.append(Line(*block.edge_coordinates(edge)))
    if len(blocks) != 0:
        viewer.add(
            Collection(blocks),
            show_faces=True,
            show_lines=False,
            opacity=0.6,
            facecolor=(0.9, 0.9, 0.9),
        )
    if len(supports) != 0:
        viewer.add(
            Collection(supports),
            show_faces=True,
            show_lines=False,
            opacity=0.5,
            facecolor=Color.from_hex("#f79d84"),
        )
    if len(blockedges) != 0:
        viewer.add(Collection(blockedges), linewidth=1.5)
    if len(supportedges) != 0:
        viewer.add(Collection(supportedges), linecolor=Color.from_hex("#f79d84"), linewidth=4)


def draw_interfaces(assembly, viewer):
    _load_compas_view2()

    interfaces = []
    faces = []
    for edge in assembly.graph.edges():
        interface = assembly.graph.edge_attribute(edge, "interface")
        if interface is not None:
            corners = np.array(interface.points)
            faces.append(Polyline(np.vstack((corners, corners[0]))))
            if assembly.graph.node_attribute(edge[0], "is_support") or assembly.graph.node_attribute(
                edge[1], "is_support"
            ):
                continue
            polygon = Polygon(interface.points)
            interfaces.append(Mesh.from_polygons([polygon]))
        if assembly.graph.edge_attribute(edge, "interfaces") is None:
            continue
        for subinterface in assembly.graph.edge_attribute(edge, "interfaces"):
            corners = np.array(subinterface.points)
            faces.append(Polyline(np.vstack((corners, corners[0]))))
            polygon = Polygon(subinterface.points)
            interfaces.append(Mesh.from_polygons([polygon]))

    if len(interfaces) != 0:
        viewer.add(
            Collection(interfaces),
            show_lines=False,
            show_points=False,
            facecolor=(0.8, 0.8, 0.8),
        )
    if len(faces) != 0:
        viewer.add(
            Collection(faces),
            linecolor=Color.from_hex("#fac05e"),
            linewidth=10,
            pointsize=10,
            show_points=True,
            pointcolor=(0, 0, 0),
        )


def draw_forces(assembly, viewer, scale=1.0, resultant=True, nodal=False):
    _load_compas_view2()

    locs = []
    res_np = []
    res_nn = []
    fnp = []
    fnn = []
    ft = []
    for edge in assembly.graph.edges():
        interface = assembly.graph.edge_attribute(edge, "interface")
        if interface is None:
            break
        forces = interface.forces
        if forces is None:
            continue
        corners = np.array(interface.points)
        frame = interface.frame
        w, u, v = frame.zaxis, frame.xaxis, frame.yaxis
        if nodal:
            for i, corner in enumerate(corners):
                pt = Point(*corner)
                force = forces[i]["c_np"] - forces[i]["c_nn"]
                p1 = pt + w * force * 0.5 * scale
                p2 = pt - w * force * 0.5 * scale
                if force >= 0:
                    fnn.append(Line(p1, p2))
                else:
                    fnp.append(Line(p1, p2))
                ft_uv = (u * forces[i]["c_u"] + v * forces[i]["c_v"]) * 0.5 * scale
                p1 = pt + ft_uv
                p2 = pt - ft_uv
                ft.append(Line(p1, p2))
        if resultant:
            sum_n = sum(force["c_np"] - force["c_nn"] for force in forces)
            sum_u = sum(force["c_u"] for force in forces)
            sum_v = sum(force["c_v"] for force in forces)
            if sum_n == 0:
                continue
            resultant_pos = np.average(
                np.array(corners),
                axis=0,
                weights=[force["c_np"] - force["c_nn"] for force in forces],
            )
            locs.append(Point(*resultant_pos))
            # resultant
            resultant_f = (w * sum_n + u * sum_u + v * sum_v) * 0.5 * scale
            p1 = resultant_pos + resultant_f
            p2 = resultant_pos - resultant_f
            if sum_n >= 0:
                res_np.append(Line(p1, p2))
            else:
                res_nn.append(Line(p1, p2))
    if len(locs) != 0:
        viewer.add(Collection(locs), size=12, color=Color.from_hex("#386641"))
    if len(res_np) != 0:
        viewer.add(Collection(res_np), linewidth=8, linecolor=(0, 0.3, 0))
    if len(res_nn) != 0:
        viewer.add(Collection(res_nn), linewidth=8, linecolor=(0.8, 0, 0))
    if len(fnn) != 0:
        viewer.add(Collection(fnn), linewidth=5, linecolor=Color.from_hex("#00468b"))
    if len(fnp) != 0:
        viewer.add(Collection(fnp), linewidth=5, linecolor=(1, 0, 0))
    if len(ft) != 0:
        viewer.add(Collection(ft), linewidth=5, linecolor=(1.0, 0.5, 0.0))


def draw_forcesline(assembly, viewer, scale=1.0, resultant=True, nodal=False):
    _load_compas_view2()

    locs = []
    res_np = []
    res_nn = []
    fnp = []
    fnn = []
    ft = []
    # total_reaction = 0
    for edge in assembly.graph.edges():
        for interface in assembly.graph.edge_attribute(edge, "interfaces"):
            forces = interface.forces
            if forces is None:
                continue
            corners = np.array(interface.points)
            frame = interface.frame
            w, u, v = frame.zaxis, frame.xaxis, frame.yaxis
            if nodal:
                for i, corner in enumerate(corners):
                    pt = Point(*corner)
                    force = forces[i]["c_np"] - forces[i]["c_nn"]
                    p1 = pt + w * force * 0.5 * scale
                    p2 = pt - w * force * 0.5 * scale
                    if force >= 0:
                        fnn.append(Line(p1, p2))
                    else:
                        fnp.append(Line(p1, p2))
                    ft_uv = (u * forces[i]["c_u"] + v * forces[i]["c_v"]) * 0.5 * scale
                    p1 = pt + ft_uv
                    p2 = pt - ft_uv
                    ft.append(Line(p1, p2))
            if resultant:
                is_tension = False
                for force in forces:
                    if force["c_np"] - force["c_nn"] <= -1e-5:
                        is_tension = True

                sum_n = sum(force["c_np"] - force["c_nn"] for force in forces)
                sum_u = sum(force["c_u"] for force in forces)
                sum_v = sum(force["c_v"] for force in forces)
                if sum_n == 0:
                    continue
                resultant_pos = np.average(
                    np.array(corners),
                    axis=0,
                    weights=[force["c_np"] - force["c_nn"] for force in forces],
                )
                locs.append(Point(*resultant_pos))
                # resultant
                resultant_f = (w * sum_n + u * sum_u + v * sum_v) * 0.5 * scale
                # print((w * sum_n + u * sum_u + v * sum_v).length * 100000, "edge: ", edge)

                # if assembly.graph.node_attribute(edge[0], "is_support") or assembly.graph.node_attribute(
                #     edge[1], "is_support"
                # ):
                #     print((w * sum_n + u * sum_u + v * sum_v).z)
                # total_reaction += abs((w * sum_n + u * sum_u + v * sum_v).z * 100000)

                p1 = resultant_pos + resultant_f
                p2 = resultant_pos - resultant_f

                if not is_tension:
                    res_np.append(Line(p1, p2))
                else:
                    res_nn.append(Line(p1, p2))
    if len(locs) != 0:
        viewer.add(Collection(locs), pointsize=12, pointcolor=Color.from_hex("#386641"))
    if len(res_np) != 0:
        viewer.add(Collection(res_np), linewidth=8, linecolor=(0, 0.3, 0))
    if len(res_nn) != 0:
        viewer.add(Collection(res_nn), linewidth=8, linecolor=(0.8, 0, 0))
    if len(fnn) != 0:
        viewer.add(Collection(fnn), linewidth=5, linecolor=Color.from_hex("#00468b"))
    if len(fnp) != 0:
        viewer.add(Collection(fnp), linewidth=5, linecolor=(1, 0, 0))
    if len(ft) != 0:
        viewer.add(Collection(ft), linewidth=5, linecolor=(1.0, 0.5, 0.0))
    # print("total reaction: ", total_reaction)


def draw_forcesdirect(assembly, viewer, scale=1.0, resultant=True, nodal=False):
    _load_compas_view2()

    locs = []
    res_np = []
    res_nn = []
    fnp = []
    fnn = []
    ft = []
    for edge in assembly.graph.edges():
        thres = 1e-6
        if assembly.graph.node_attribute(edge[0], "is_support") and not assembly.graph.node_attribute(
            edge[1], "is_support"
        ):
            flip = False
        else:
            flip = True
        if assembly.graph.edge_attribute(edge, "interfaces") is None:
            continue
        for interface in assembly.graph.edge_attribute(edge, "interfaces"):
            forces = interface.forces
            if forces is None:
                continue
            corners = np.array(interface.points)
            frame = interface.frame
            w, u, v = frame.zaxis, frame.xaxis, frame.yaxis
            if nodal:
                for i, corner in enumerate(corners):
                    pt = Point(*corner)
                    force = forces[i]["c_np"] - forces[i]["c_nn"]
                    if (w * force * scale).length == 0:
                        continue
                    if flip:
                        f = Arrow(
                            pt,
                            w * force * scale * -1,
                            head_portion=0.2,
                            head_width=0.07,
                            body_width=0.02,
                        )
                    else:
                        f = Arrow(
                            pt,
                            w * force * scale,
                            head_portion=0.2,
                            head_width=0.07,
                            body_width=0.02,
                        )
                    if force >= 0:
                        fnp.append(f)
                    else:
                        fnn.append(f)
                    ft_uv = (u * forces[i]["c_u"] + v * forces[i]["c_v"]) * scale
                    if ft_uv.length == 0:
                        continue
                    if flip:
                        f = Arrow(
                            pt,
                            ft_uv * -1,
                            head_portion=0.2,
                            head_width=0.07,
                            body_width=0.02,
                        )
                    else:
                        f = Arrow(
                            pt,
                            ft_uv,
                            head_portion=0.2,
                            head_width=0.07,
                            body_width=0.02,
                        )
                    ft.append(f)
            if resultant:
                is_tension = False

                for force in forces:
                    if force["c_np"] - force["c_nn"] <= -1e-5:
                        is_tension = True

                sum_n = sum(force["c_np"] - force["c_nn"] for force in forces)
                sum_u = sum(force["c_u"] for force in forces)
                sum_v = sum(force["c_v"] for force in forces)
                if abs(sum_n) <= thres:
                    resultant_pos = np.average(
                        np.array(corners),
                        axis=0,
                        weights=[sqrt(force["c_u"] ** 2 + force["c_v"] ** 2) for force in forces],
                    )
                    friction = True
                else:
                    resultant_pos = np.average(
                        np.array(corners),
                        axis=0,
                        weights=[force["c_np"] - force["c_nn"] for force in forces],
                    )
                    friction = False
                resultant_f = (w * sum_n + u * sum_u + v * sum_v) * scale
                if resultant_f.length >= thres:
                    locs.append(Point(*resultant_pos))
                if flip:
                    f = Arrow(
                        resultant_pos,
                        resultant_f * -1,
                        head_portion=0.2,
                        head_width=0.07,
                        body_width=0.02,
                    )
                else:
                    f = Arrow(
                        resultant_pos,
                        resultant_f,
                        head_portion=0.2,
                        head_width=0.07,
                        body_width=0.02,
                    )
                if friction:
                    viewer.add(f, facecolor=(1.0, 0.5, 0.0), show_lines=False)
                if not is_tension:
                    res_np.append(f)
                else:
                    res_nn.append(f)
    if len(locs) != 0:
        viewer.add(Collection(locs), size=12, color=Color.from_hex("#386641"))
    if len(res_np) != 0:
        viewer.add(Collection(res_np), facecolor=Color.from_hex("#386641"), show_lines=False)
    if len(res_nn) != 0:
        viewer.add(Collection(res_nn), facecolor=(0.8, 0, 0), show_lines=False)
    if len(fnp) != 0:
        viewer.add(
            Collection(fnp),
            facecolor=Color.from_hex("#00468b"),
            show_lines=False,
            opacity=0.5,
        )
    if len(fnn) != 0:
        viewer.add(Collection(fnn), facecolor=(1, 0, 0), show_lines=False, opacity=0.5)
    if len(ft) != 0:
        viewer.add(Collection(ft), facecolor=(1.0, 0.5, 0.0), show_lines=False, opacity=0.5)


def draw_displacements(assembly, viewer, dispscale=1.0, tol=0.0):
    _load_compas_view2()

    blocks = []
    nodes = []
    for node in assembly.graph.nodes():
        if assembly.graph.node_attribute(node, "is_support"):
            continue
        block = assembly.graph.node_attribute(node, "block")
        displacement = assembly.graph.node_attribute(node, "displacement")
        if displacement is None:
            continue
        displacement = np.array(displacement) * dispscale
        vec = (
            np.array([1, 0, 0]) * displacement[3]
            + np.array([0, 1, 0]) * displacement[4]
            + np.array([0, 0, 1]) * displacement[5]
        ).tolist()
        R = Rotation.from_axis_angle_vector(vec, point=block.center())
        T = Translation.from_vector(displacement[0:3])
        new_block = block.transformed(R).transformed(T)
        nodes.append(Point(*new_block.center()))
        for edge in block.edges():
            if tol != 0.0:
                fkeys = block.edge_faces(edge)
                ps = [
                    block.face_center(fkeys[0]),
                    block.face_center(fkeys[1]),
                    *block.edge_coordinates(edge),
                ]
                if is_coplanar(ps, tol=tol):
                    continue
            blocks.append(Line(*new_block.edge_coordinates(edge)))
    if len(blocks) != 0:
        viewer.add(Collection(blocks), linewidth=1, linecolor=(0.7, 0.7, 0.7))
    if len(nodes) != 0:
        viewer.add(Collection(nodes), pointcolor=(0.7, 0.7, 0.7))


def draw_weights(assembly, viewer, scale=1.0, density=1.0):
    _load_compas_view2()

    weights = []
    blocks = []
    supports = []
    # total_weights = 0
    for node in assembly.graph.nodes():
        block = assembly.graph.node_attribute(node, "block")
        if assembly.graph.node_attribute(node, "is_support"):
            supports.append(Point(*block.center()))
            continue
        d = block.attributes["density"] if "density" in block.attributes else density
        weights.append(
            Arrow(
                block.center(),
                [0, 0, -block.volume() * d * scale],
                head_portion=0.2,
                head_width=0.07,
                body_width=0.02,
            )
        )
        # print("self-weight", -block.volume() * density)
        # total_weights += block.volume() * 2500 * 9.8
        blocks.append(Point(*block.center()))

    # print("total self-weight: ", total_weights)

    if len(supports) != 0:
        viewer.add(Collection(supports), pointsize=20, pointcolor=Color.from_hex("#ee6352"))
    if len(blocks) != 0:
        viewer.add(Collection(blocks), pointsize=30, pointcolor=Color.from_hex("#3284a0"))
    if len(weights) != 0:
        viewer.add(Collection(weights), facecolor=Color.from_hex("#59cd90"), show_lines=False)


def cra_view(
    assembly,
    scale=1.0,
    density=1.0,
    dispscale=1.0,
    tol=1e-5,
    grid=False,
    resultant=True,
    nodal=False,
    edge=True,
    blocks=True,
    interfaces=True,
    forces=True,
    forcesdirect=True,
    forcesline=False,
    weights=True,
    displacements=True,
):
    """CRA Viewer, creating new viewer.

    Parameters
    ----------
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        The rigid block assembly.
    scale : float, optional
        Force scale.
    density : float, optional
        Density of the block material.
    dispscale : float, optional
        virtual displacement scale.
    tol : float, optional
        Tolerance value to consider faces to be planar.
    grid : bool, optional
        Show view grid.
    resultant : bool, optional
        Plot resultant forces.
    nodal : bool, optional
        Plot nodal forces.
    edge : bool, optional
        Plot block edges.
    blocks : bool, optional
        Plot block.
    interfaces : bool, optional
        Plot interfaces.
    forces : bool, optional
        Plot forces.
    forcesdirect : bool, optional
        Plot forces as vectors.
    forcesline : bool, optional
        Plot forces as lines.
    weights : bool, optional
        Plot block self weight as vectors.
    displacements : bool, optional
        Plot virtual displacements.

    Returns
    -------
    None
    """

    _load_compas_view2()
    viewer = app.App(width=1600, height=1000, viewmode="shaded", show_grid=grid)

    if blocks:
        draw_blocks(assembly, viewer, edge, tol)
    if interfaces:
        draw_interfaces(assembly, viewer)
    if forces:
        draw_forces(assembly, viewer, scale, resultant, nodal)
    if forcesdirect:
        draw_forcesdirect(assembly, viewer, scale, resultant, nodal)
    if forcesline:
        draw_forcesline(assembly, viewer, scale, resultant, nodal)
    if weights:
        draw_weights(assembly, viewer, scale, density)
    if displacements:
        draw_displacements(assembly, viewer, dispscale, tol)

    viewer.run()


def cra_view_ex(
    viewer,
    assembly,
    scale=1.0,
    density=1.0,
    dispscale=1.0,
    tol=1e-5,
    resultant=True,
    nodal=False,
    edge=True,
    blocks=True,
    interfaces=True,
    forces=True,
    forcesdirect=True,
    forcesline=False,
    weights=True,
    displacements=True,
):
    """CRA Viewer using existing view.

    Parameters
    ----------
    viewer : compas_view2.app.App
        External viewer object.
    assembly : :class:`~compas_assembly.datastructures.Assembly`
        The rigid block assembly.
    scale : float, optional
        Force scale.
    density : float, optional
        Density of the block material.
    dispscale : float, optional
        virtual displacement scale.
    tol : float, optional
        Tolerance value to consider faces to be planar.
    resultant : bool, optional
        Plot resultant forces.
    nodal : bool, optional
        Plot nodal forces.
    edge : bool, optional
        Plot block edges.
    blocks : bool, optional
        Plot block.
    interfaces : bool, optional
        Plot interfaces.
    forces : bool, optional
        Plot forces.
    forcesdirect : bool, optional
        Plot forces as vectors.
    forcesline : bool, optional
        Plot forces as lines.
    weights : bool, optional
        Plot block self weight as vectors.
    displacements : bool, optional
        Plot virtual displacements.

    Returns
    -------
    None
    """

    if blocks:
        draw_blocks(assembly, viewer, edge, tol)
    if interfaces:
        draw_interfaces(assembly, viewer)
    if forces:
        draw_forces(assembly, viewer, scale, resultant, nodal)
    if forcesdirect:
        draw_forcesdirect(assembly, viewer, scale, resultant, nodal)
    if forcesline:
        draw_forcesline(assembly, viewer, scale, resultant, nodal)
    if weights:
        draw_weights(assembly, viewer, scale, density)
    if displacements:
        draw_displacements(assembly, viewer, dispscale, tol)
