"""Utility functions shared across different types of environments."""

import abc

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from prpl_utils.utils import fig2data
from relational_structs import (
    Object,
    ObjectCentricState,
)
from tomsgeoms2d.structs import Circle, Geom2D, Lobject, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect as original_geom2ds_intersect
from tomsgeoms2d.utils import (
    line_segment_intersects_circle,
    line_segment_intersects_rectangle,
    line_segments_intersect,
)

from pr2s2r.prbench.envs.geom2d.object_types import (
    CircleType,
    CRVRobotType,
    DoubleRectType,
    LObjectType,
    RectangleType,
)
from pr2s2r.prbench.envs.geom2d.structs import (
    Body2D,
    MultiBody2D,
    SE2Pose,
    ZOrder,
    z_orders_may_collide,
)

PURPLE: tuple[float, float, float] = (128 / 255, 0 / 255, 128 / 255)
BLACK: tuple[float, float, float] = (0.1, 0.1, 0.1)
BROWN: tuple[float, float, float] = (0.4, 0.2, 0.1)
ORANGE: tuple[float, float, float] = (1.0, 165 / 255, 0.0)


class RobotActionSpace(Box):
    """A space for robot actions."""

    @abc.abstractmethod
    def create_markdown_description(self) -> str:
        """Create a markdown description of this space."""


def lobject_intersects_lobject(lobj1: Lobject, lobj2: Lobject) -> bool:
    """Checks if two L-objects intersect."""
    # Check if any line segment of lobj1 intersects any line segment of lobj2
    if any(
        line_segments_intersect(seg1, seg2)
        for seg1 in lobj1.line_segments
        for seg2 in lobj2.line_segments
    ):
        return True
    # Case 2: lobj1 inside lobj2
    if lobj2.contains_point(lobj1.x, lobj1.y):
        return True
    # Case 3: lobj2 inside lobj1
    if lobj1.contains_point(lobj2.x, lobj2.y):
        return True
    # Not intersecting
    return False


def lobject_intersects_rectangle(lobj: Lobject, rect: Rectangle) -> bool:
    """Checks if an L-object intersects with a rectangle."""
    # Check if any line segment of the L-object intersects the rectangle
    if any(line_segment_intersects_rectangle(seg, rect) for seg in lobj.line_segments):
        return True
    # Case 2: rectangle inside L-object
    if lobj.contains_point(rect.center[0], rect.center[1]):
        return True
    # Case 3: L-object inside rectangle
    if rect.contains_point(lobj.x, lobj.y):
        return True
    # Not intersecting
    return False


def lobject_intersects_circle(lobj: Lobject, circ: Circle) -> bool:
    """Checks if an L-object intersects with a circle."""
    # Check if any line segment of the L-object intersects the circle
    if any(line_segment_intersects_circle(seg, circ) for seg in lobj.line_segments):
        return True
    # Case 2: circle center inside L-object
    if lobj.contains_point(circ.x, circ.y):
        return True
    # Case 3: L-object center inside circle
    if circ.contains_point(lobj.x, lobj.y):
        return True
    # Not intersecting
    return False


def rectangle_intersects_lobject(rect: Rectangle, lobj: Lobject) -> bool:
    """Checks if a rectangle intersects with an L-object."""
    return lobject_intersects_rectangle(lobj, rect)


def circle_intersects_lobject(circ: Circle, lobj: Lobject) -> bool:
    """Checks if a circle intersects with an L-object."""
    return lobject_intersects_circle(lobj, circ)


def geom2ds_intersect(geom1: Geom2D, geom2: Geom2D) -> bool:
    """Check if two 2D bodies intersect, with support for L-objects."""
    # Handle L-object cases first
    if isinstance(geom1, Lobject) and isinstance(geom2, Lobject):
        return lobject_intersects_lobject(geom1, geom2)
    if isinstance(geom1, Lobject) and isinstance(geom2, Rectangle):
        return lobject_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Lobject):
        return rectangle_intersects_lobject(geom1, geom2)
    if isinstance(geom1, Lobject) and isinstance(geom2, Circle):
        return lobject_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Lobject):
        return circle_intersects_lobject(geom1, geom2)

    # For all other cases, use the original function
    return original_geom2ds_intersect(geom1, geom2)


def crv_robot_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d()."""
    assert obj.is_instance(CRVRobotType)
    bodies: list[Body2D] = []

    # Base.
    base_x = state.get(obj, "x")
    base_y = state.get(obj, "y")
    base_radius = state.get(obj, "base_radius")
    circ = Circle(
        x=base_x,
        y=base_y,
        radius=base_radius,
    )
    z_order = ZOrder.ALL
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    base = Body2D(circ, z_order, rendering_kwargs, name="base")
    bodies.append(base)

    # Gripper.
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_cx = base_x + np.cos(theta) * arm_joint
    gripper_cy = base_y + np.sin(theta) * arm_joint
    gripper_height = state.get(obj, "gripper_height")
    gripper_width = state.get(obj, "gripper_width")
    rect = Rectangle.from_center(
        center_x=gripper_cx,
        center_y=gripper_cy,
        height=gripper_height,
        width=gripper_width,
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    gripper = Body2D(rect, z_order, rendering_kwargs, name="gripper")
    bodies.append(gripper)

    # Arm.
    rect = Rectangle.from_center(
        center_x=(base_x + gripper_cx) / 2,
        center_y=(base_y + gripper_cy) / 2,
        height=np.sqrt((base_x - gripper_cx) ** 2 + (base_y - gripper_cy) ** 2),
        width=(0.5 * gripper_width),
        rotation_about_center=(theta + np.pi / 2),
    )
    z_order = ZOrder.SURFACE
    silver = (128 / 255, 128 / 255, 128 / 255)
    rendering_kwargs = {"facecolor": silver, "edgecolor": BLACK}
    arm = Body2D(rect, z_order, rendering_kwargs, name="arm")
    bodies.append(arm)

    # If the vacuum is on, add a suction area.
    if state.get(obj, "vacuum") > 0.5:
        suction_height = gripper_height
        suction_width = gripper_width
        suction_cx = base_x + np.cos(theta) * (
            arm_joint + gripper_width + suction_width / 2
        )
        suction_cy = base_y + np.sin(theta) * (
            arm_joint + gripper_width + suction_width / 2
        )
        rect = Rectangle.from_center(
            center_x=suction_cx,
            center_y=suction_cy,
            height=suction_height,
            width=suction_width,
            rotation_about_center=theta,
        )
        z_order = ZOrder.NONE  # NOTE: suction collides with nothing
        rendering_kwargs = {"facecolor": PURPLE}
        suction = Body2D(rect, z_order, rendering_kwargs, name="suction")
        bodies.append(suction)

    return MultiBody2D(obj.name, bodies)


def double_rectangle_object_to_part_geom(
    state: ObjectCentricState,
    double_rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract the second rectangle for a DoubleRectType object."""
    assert double_rect_obj.is_instance(DoubleRectType)
    multibody = object_to_multibody2d(double_rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 3
    # The second body is the "part" rectangle.
    assert "part" in multibody.bodies[2].name
    geom = multibody.bodies[2].geom
    assert isinstance(geom, Rectangle)
    return geom


def get_se2_pose(state: ObjectCentricState, obj: Object) -> SE2Pose:
    """Get the SE2Pose of an object in a given state."""
    return SE2Pose(
        x=state.get(obj, "x"),
        y=state.get(obj, "y"),
        theta=state.get(obj, "theta"),
    )


def object_to_multibody2d(
    obj: Object,
    state: ObjectCentricState,
    static_object_cache: dict[Object, MultiBody2D],
) -> MultiBody2D:
    """Create a Body2D instance for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return crv_robot_to_multibody2d(obj, state)
    is_static = state.get(obj, "static") > 0.5
    if is_static and obj in static_object_cache:
        return static_object_cache[obj]
    geom: Geom2D  # rectangle or circle
    if obj.is_instance(RectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        geom = Rectangle(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(CircleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        geom = Circle(x, y, radius)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(LObjectType):
        multibody = geom2d_lobject_to_multibody2d(obj, state)
    elif obj.is_instance(DoubleRectType):
        multibody = geom2d_double_rectangle_to_multibody2d(obj, state)
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = multibody
    return multibody


def rectangle_object_to_geom(
    state: ObjectCentricState,
    rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract a rectangle for an object."""
    assert rect_obj.is_instance(RectangleType)
    multibody = object_to_multibody2d(rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 1
    geom = multibody.bodies[0].geom
    assert isinstance(geom, Rectangle)
    return geom


def geom2d_lobject_to_multibody2d(
    obj: Object, state: ObjectCentricState
) -> MultiBody2D:
    """Helper to create a MultiBody2D for an LObjectType object."""
    assert obj.is_instance(LObjectType)
    # Get parameters
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    theta = state.get(obj, "theta")
    width = state.get(obj, "width")
    length_side1 = state.get(obj, "length_side1")
    length_side2 = state.get(obj, "length_side2")
    color = (
        state.get(obj, "color_r"),
        state.get(obj, "color_g"),
        state.get(obj, "color_b"),
    )
    z_order = ZOrder(int(state.get(obj, "z_order")))

    geom = Lobject(x, y, width, (length_side1, length_side2), theta)

    rendering_kwargs = {
        "facecolor": color,
        "edgecolor": BLACK,
    }
    body = Body2D(geom, z_order, rendering_kwargs, name="hook")

    return MultiBody2D(obj.name, [body])


def geom2d_double_rectangle_to_multibody2d(
    obj: Object, state: ObjectCentricState
) -> MultiBody2D:
    """Helper to create a MultiBody2D for a DoubleRectType object."""
    assert obj.is_instance(DoubleRectType)
    # Note: We need to assume the two rectangles are aligned now.
    # This means theta is the same, relative dy == 0, 0 <= dx < width0 - width1.
    # Such that we can create two obstacles from the base rectangle.
    bodies: list[Body2D] = []

    # First rectangle.
    x0 = state.get(obj, "x")
    y0 = state.get(obj, "y")
    theta0 = state.get(obj, "theta")
    height0 = state.get(obj, "height")
    width0 = state.get(obj, "width")
    pose0 = SE2Pose(x0, y0, theta0)
    # Second rectangle.
    x1 = state.get(obj, "x1")
    y1 = state.get(obj, "y1")
    theta1 = state.get(obj, "theta1")
    width1 = state.get(obj, "width1")
    height1 = state.get(obj, "height1")
    pose1 = SE2Pose(x1, y1, theta1)
    assert theta0 == theta1, f"Expected theta0 == theta1, got {theta0} != {theta1}"
    relative_pose = pose0.inverse * pose1
    assert relative_pose.y == 0.0, f"Expected relative y == 0, got {relative_pose.y}"
    assert relative_pose.x >= 0.0, f"Expected relative x >= 0, got {relative_pose.x}"
    assert relative_pose.x + width1 < width0, "Expected relative x + width1 < width0"
    right_bookend_width = width0 - width1 - relative_pose.x

    # Left bookend.
    geom0 = Rectangle(x0, y0, relative_pose.x, height0, theta0)
    z_order0 = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs0 = {
        "facecolor": (
            state.get(obj, "color_r"),
            state.get(obj, "color_g"),
            state.get(obj, "color_b"),
        ),
        "edgecolor": BLACK,
    }
    body0 = Body2D(geom0, z_order0, rendering_kwargs0, name=f"{obj.name}_base0")
    bodies.append(body0)
    # Right bookend.
    right_bookend_pose = pose0 * SE2Pose(relative_pose.x + width1, 0.0, theta0)
    geom0_ = Rectangle(
        right_bookend_pose.x, right_bookend_pose.y, right_bookend_width, height0, theta0
    )
    z_order0_ = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs0_ = {
        "facecolor": (
            state.get(obj, "color_r"),
            state.get(obj, "color_g"),
            state.get(obj, "color_b"),
        ),
        "edgecolor": BLACK,
    }
    body0_ = Body2D(geom0_, z_order0_, rendering_kwargs0_, name=f"{obj.name}_base1")
    bodies.append(body0_)

    # Second rectangle.
    x1 = state.get(obj, "x1")
    y1 = state.get(obj, "y1")
    width1 = state.get(obj, "width1")
    height1 = state.get(obj, "height1")
    theta1 = state.get(obj, "theta1")
    geom1 = Rectangle(x1, y1, width1, height1, theta1)
    z_order1 = ZOrder(int(state.get(obj, "z_order1")))
    rendering_kwargs1 = {
        "facecolor": (
            state.get(obj, "color_r1"),
            state.get(obj, "color_g1"),
            state.get(obj, "color_b1"),
        ),
        "edgecolor": BLACK,
        "alpha": 0.5,
    }
    body1 = Body2D(geom1, z_order1, rendering_kwargs1, name=f"{obj.name}_part")
    bodies.append(body1)

    return MultiBody2D(obj.name, bodies)


def sample_se2_pose(
    bounds: tuple[SE2Pose, SE2Pose], rng: np.random.Generator
) -> SE2Pose:
    """Sample a SE2Pose uniformly between the bounds."""
    lb, ub = bounds
    x = rng.uniform(lb.x, ub.x)
    y = rng.uniform(lb.y, ub.y)
    theta = rng.uniform(lb.theta, ub.theta)
    return SE2Pose(x, y, theta)


def state_2d_has_collision(
    state: ObjectCentricState,
    group1: set[Object],
    group2: set[Object],
    static_object_cache: dict[Object, MultiBody2D],
    ignore_z_orders: bool = False,
) -> bool:
    """Check for collisions between any objects in two groups."""
    # Create multibodies once.
    obj_to_multibody = {
        o: object_to_multibody2d(o, state, static_object_cache) for o in state
    }
    # Check pairwise collisions.
    for obj1 in group1:
        for obj2 in group2:
            obj1_static = (
                state.get(obj1, "static")
                if "static" in state.type_features[obj1.type]
                else False
            )
            obj2_static = (
                state.get(obj2, "static")
                if "static" in state.type_features[obj2.type]
                else False
            )
            if obj1 == obj2 or (obj1_static and obj2_static):
                # Skip self-collision and static-static collision.
                continue
            multibody1 = obj_to_multibody[obj1]
            multibody2 = obj_to_multibody[obj2]
            for body1 in multibody1.bodies:
                for body2 in multibody2.bodies:
                    if not (
                        ignore_z_orders
                        or z_orders_may_collide(body1.z_order, body2.z_order)
                    ):
                        continue
                    if geom2ds_intersect(body1.geom, body2.geom):
                        return True
    return False


def render_2dstate_on_ax(
    state: ObjectCentricState,
    ax: plt.Axes,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
) -> None:
    """Render a state on an existing plt.Axes."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    # Sort objects by ascending z order, with the robot first.
    def _render_order(obj: Object) -> int:
        if obj.is_instance(CRVRobotType):
            return -1
        return int(state.get(obj, "z_order"))

    for obj in sorted(state, key=_render_order):
        body = object_to_multibody2d(obj, state, static_object_body_cache)
        body.plot(ax)


def render_2dstate(
    state: ObjectCentricState,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    world_min_x: float = 0.0,
    world_max_x: float = 10.0,
    world_min_y: float = 0.0,
    world_max_y: float = 10.0,
    render_dpi: int = 150,
) -> NDArray[np.uint8]:
    """Render a state.

    Useful for viz and debugging.
    """
    if static_object_body_cache is None:
        static_object_body_cache = {}

    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=render_dpi)

    render_2dstate_on_ax(state, ax, static_object_body_cache)

    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()
    img = fig2data(fig)
    plt.close()
    return img
