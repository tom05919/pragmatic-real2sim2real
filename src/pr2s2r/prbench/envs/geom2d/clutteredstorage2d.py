"""Cluttered environment where blocks must be stored on a shelf."""

from dataclasses import dataclass

import numpy as np
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict
from tomsgeoms2d.structs import Rectangle

from pr2s2r.prbench.core import ConstantObjectPRBenchEnv, FinalConfigMeta
from pr2s2r.prbench.envs.geom2d.base_env import (
    Geom2DRobotEnvConfig,
    ObjectCentricGeom2DRobotEnv,
)
from pr2s2r.prbench.envs.geom2d.object_types import (
    CRVRobotType,
    DoubleRectType,
    Geom2DRobotEnvTypeFeatures,
    LObjectType,
    RectangleType,
)
from pr2s2r.prbench.envs.geom2d.structs import SE2Pose, ZOrder
from pr2s2r.prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    is_inside_shelf,
)
from pr2s2r.prbench.envs.utils import (
    BLACK,
    PURPLE,
    sample_se2_pose,
    state_2d_has_collision,
)

# NOTE: unlike some other environments, there are multiple target blocks here.
TargetBlockType = Type("target_block", parent=RectangleType)
Geom2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Geom2DRobotEnvTypeFeatures[RectangleType]
)
# There is only one target region (the shelf) and it is bookended by obstacles.
ShelfType = Type("shelf", parent=DoubleRectType)
Geom2DRobotEnvTypeFeatures[ShelfType] = list(Geom2DRobotEnvTypeFeatures[DoubleRectType])


@dataclass(frozen=True)
class ClutteredStorage2DEnvConfig(Geom2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for ClutteredStorage2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 5.0
    world_min_y: float = 0.0
    world_max_y: float = 3.0

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # Shelf hyperparameters:
    shelf_rgb: tuple[float, float, float] = PURPLE
    shelf_height: float = (world_max_y - world_min_y) / 8
    shelf_width_pad: float = shelf_height / 10
    shelf_y: float = world_max_y - shelf_height

    # LObject blocker hyperparameters:
    occupied_percentage: float = 0.8

    # Robot hyperparameters.
    robot_base_radius: float = 0.2
    # NOTE: extra long robot arm to make it easier to reach into shelf.
    robot_arm_length: float = 4 * robot_base_radius
    robot_gripper_height: float = 0.14
    robot_gripper_width: float = 0.02
    # NOTE: robot init y pose bounded above by shelf_y.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 4 * robot_base_radius,
            world_min_y + 4 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - 4 * robot_base_radius,
            shelf_y - 4 * robot_base_radius,
            np.pi,
        ),
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = (0.0, 0.3, 1.0)
    # N blocks are initialized in the shelf and N + 1 are initialized outside.
    target_block_in_shelf_rotation_bounds: tuple[float, float] = (
        -np.pi / 16,
        np.pi / 16,
    )
    target_block_out_of_shelf_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        robot_init_pose_bounds
    )
    target_block_shape: tuple[float, float] = (0.125, 0.3)

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20

    def get_shelf_width(self, num_init_shelf_blocks: int) -> float:
        """Calculate the shelf width as a function of number of blocks."""
        shelf_width = 0.5
        assert shelf_width <= self.world_max_x - self.world_min_x
        # Make sure that vertical stacking is possible.
        assert shelf_width > (num_init_shelf_blocks + 1) * self.target_block_shape[1]
        return shelf_width

    def get_shelf_init_pose_bounds(
        self, num_init_shelf_blocks: int
    ) -> tuple[SE2Pose, SE2Pose]:
        """Calculate the init pose bounds for the shelf based on block number."""
        shelf_width = self.get_shelf_width(num_init_shelf_blocks)
        return (
            SE2Pose(self.world_min_x, self.shelf_y, 0),
            SE2Pose(self.world_max_x - shelf_width, self.shelf_y, 0),
        )

    def get_target_block_in_shelf_center_positions(
        self, num_init_shelf_blocks: int, shelf_pose: SE2Pose
    ) -> list[tuple[float, float]]:
        """Get the init center positions for the target blocks in the shelf."""
        shelf_width = self.get_shelf_width(num_init_shelf_blocks)
        assert np.isclose(shelf_pose.theta, 0.0)
        total_half_pad = (self.shelf_width_pad + self.target_block_shape[0]) / 2
        min_x = shelf_pose.x + total_half_pad
        max_x = shelf_pose.x + shelf_width - total_half_pad
        xs = np.linspace(min_x, max_x, num=num_init_shelf_blocks, endpoint=True)
        # NOTE: there is an implicit assumption here that the shelf is not too
        # deep for the robot to reach in and grab the objects.
        y = shelf_pose.y + 2 * self.target_block_shape[1]
        return [(x, y) for x in xs]


class ObjectCentricClutteredStorage2DEnv(
    ObjectCentricGeom2DRobotEnv[ClutteredStorage2DEnvConfig]
):
    """Cluttered environment where blocks must be stored on a shelf.

    This is an object-centric environment. The vectorized version with Box spaces is
    defined below.
    """

    def __init__(
        self,
        num_blocks: int = 3,
        config: ClutteredStorage2DEnvConfig = ClutteredStorage2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        assert num_blocks % 2 == 1, "Number of blocks must be odd"
        self._num_init_shelf_blocks = num_blocks // 2
        self._num_init_outside_blocks = num_blocks - self._num_init_shelf_blocks

    def _sample_initial_state(self) -> ObjectCentricState:
        static_objects = set(self.initial_constant_state)
        # Sample robot pose.
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        # Sample shelf pose.
        shelf_init_pose_bounds = self.config.get_shelf_init_pose_bounds(
            self._num_init_shelf_blocks
        )
        shelf_pose = sample_se2_pose(shelf_init_pose_bounds, self.np_random)
        # Sample the target block rotations for those in the shelf.
        shelf_target_block_rotations = self.np_random.uniform(
            *self.config.target_block_in_shelf_rotation_bounds,
            size=self._num_init_shelf_blocks,
        ).tolist()
        state = self._create_initial_state(
            robot_pose,
            shelf_pose,
            shelf_target_block_rotations,  # type: ignore[arg-type]
        )
        robot = state.get_objects(CRVRobotType)[0]
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        # Sample target block poses for those outside the shelf.
        target_block_outside_poses: list[SE2Pose] = []
        for _ in range(self._num_init_outside_blocks):
            for _ in range(self.config.max_init_sampling_attempts):
                pose = sample_se2_pose(
                    self.config.target_block_out_of_shelf_pose_bounds, self.np_random
                )
                # Make sure in bounds.
                if not (
                    self.config.world_min_x < pose.x < self.config.world_max_x
                    and self.config.world_min_y < pose.y < self.config.world_max_y
                ):
                    continue
                # Check for collisions.
                state = self._create_initial_state(
                    robot_pose,
                    shelf_pose,
                    shelf_target_block_rotations,  # type: ignore[arg-type]
                    target_block_outside_poses=target_block_outside_poses + [pose],
                )
                obj_name_to_obj = {o.name: o for o in state}
                num_blocks = sum(b.startswith("block") for b in obj_name_to_obj)
                new_block = obj_name_to_obj[f"block{num_blocks-1}"]
                full_state = state.copy()
                full_state.data.update(self.initial_constant_state.data)
                if not state_2d_has_collision(
                    full_state, {new_block}, set(full_state), {}
                ):
                    break
            else:
                raise RuntimeError("Failed to sample obstruction pose.")
            # Update target blocks.
            target_block_outside_poses.append(pose)
        # The state should already be finalized.
        return state

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create room walls.
        assert isinstance(self.action_space, CRVRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        shelf_pose: SE2Pose,
        shelf_target_block_rotations: list[float],
        target_block_outside_poses: list[SE2Pose] | None = None,
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        assert self.initial_constant_state is not None
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", CRVRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,  # arm is fully retracted
            "arm_length": self.config.robot_arm_length,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": self.config.robot_gripper_height,
            "gripper_width": self.config.robot_gripper_width,
        }

        # Create the shelf.
        shelf = Object("shelf", ShelfType)
        shelf_width = self.config.get_shelf_width(self._num_init_shelf_blocks)
        init_state_dict[shelf] = {
            "x1": shelf_pose.x,
            "y1": shelf_pose.y,
            "theta1": shelf_pose.theta,
            "width1": shelf_width,
            "height1": self.config.shelf_height,
            "static": True,
            "color_r1": self.config.shelf_rgb[0],
            "color_g1": self.config.shelf_rgb[1],
            "color_b1": self.config.shelf_rgb[2],
            "z_order1": ZOrder.NONE.value,
            "x": self.config.world_min_x,
            "y": shelf_pose.y,
            "theta": shelf_pose.theta,
            "width": self.config.world_max_x - self.config.world_min_x,
            "height": self.config.shelf_height,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create L-shaped blocker inside the shelf opening.
        shelf_blocker = Object("shelf_blocker", LObjectType)
        blocker_x = shelf_pose.x + shelf_width
        # Place at the back of the shelf (minus the L's width to keep it inside)
        blocker_y = shelf_pose.y + self.config.shelf_height
        init_state_dict[shelf_blocker] = {
            "x": blocker_x,
            "y": blocker_y,
            "theta": 0.0,  # No rotation
            "width": 0.1,  # Thickness of L-shape arms
            "length_side1": shelf_width * self.config.occupied_percentage,
            "length_side2": self.config.shelf_height * self.config.occupied_percentage,
            "static": True,
            "color_r": 0.6,
            "color_g": 0.6,
            "color_b": 0.6,
            "z_order": ZOrder.SURFACE.value,
        }

        # Create the target blocks that are initially in the shelf. Evenly space
        # them horizontally and apply rotations.
        block_num = 0
        target_block_in_shelf_center_positions = (
            self.config.get_target_block_in_shelf_center_positions(
                self._num_init_shelf_blocks, shelf_pose
            )
        )
        for (center_x, center_y), rot in zip(
            target_block_in_shelf_center_positions,
            shelf_target_block_rotations,
            strict=True,
        ):
            block = Object(f"block{block_num}", TargetBlockType)
            block_num += 1
            rect = Rectangle.from_center(
                center_x,
                center_y,
                self.config.target_block_shape[0],
                self.config.target_block_shape[1],
                rot,
            )
            init_state_dict[block] = {
                "x": rect.x,
                "y": rect.y,
                "theta": rect.theta,
                "width": rect.width,
                "height": rect.height,
                "static": False,
                "color_r": self.config.target_block_rgb[0],
                "color_g": self.config.target_block_rgb[1],
                "color_b": self.config.target_block_rgb[2],
                "z_order": ZOrder.SURFACE.value,
            }

        # Create the target blocks that are initially outside the shelf.
        if target_block_outside_poses is not None:
            for pose in target_block_outside_poses:
                block = Object(f"block{block_num}", TargetBlockType)
                block_num += 1
                init_state_dict[block] = {
                    "x": pose.x,
                    "y": pose.y,
                    "theta": pose.theta,
                    "width": self.config.target_block_shape[0],
                    "height": self.config.target_block_shape[1],
                    "static": False,
                    "color_r": self.config.target_block_rgb[0],
                    "color_g": self.config.target_block_rgb[1],
                    "color_b": self.config.target_block_rgb[2],
                    "z_order": ZOrder.SURFACE.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Geom2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        assert self._current_state is not None
        shelf = self._current_state.get_objects(ShelfType)[0]
        blocks = self._current_state.get_objects(TargetBlockType)
        terminated = all(
            is_inside_shelf(
                self._current_state,
                block,
                shelf,
                self._static_object_body_cache,
            )
            for block in blocks
        )
        return -1.0, terminated


class ClutteredStorage2DEnv(ConstantObjectPRBenchEnv):
    """Cluttered storage 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricGeom2DRobotEnv:
        return ObjectCentricClutteredStorage2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "shelf", "shelf_blocker"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("block"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        num_blocks = len(self._constant_objects) - 3  # robot, shelf, shelf_blocker
        # pylint: disable=line-too-long
        return f"""A 2D environment where the goal is to put all blocks inside a shelf.

There are always {num_blocks} blocks in this environment.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when all blocks are inside the shelf.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return "Similar environments have been considered by many others, especially in the task and motion planning literature.\n"  # pylint: disable=line-too-long
