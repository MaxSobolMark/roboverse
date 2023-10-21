import numpy as np
import gym
import time
from gym.spaces import Box, Dict

from matplotlib import pyplot as plt
import roboverse.bullet as bullet
from roboverse.bullet import object_utils
from roboverse.envs import objects
from roboverse.assets.shapenet_object_lists import (
    BIN_SORT_OBJECTS,
    CONTAINER_CONFIGS_BIN_SORT,
    OBJECT_ORIENTATIONS,
    OBJECT_SCALINGS,
)


END_EFFECTOR_INDEX = 8
RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0.0, 0.0, 0.036, -0.036]
RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
GUESS = 3.14  # TODO(avi) This is a guess, need to verify what joint this is

JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
                     -0.037]
JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]
JOINT_RANGE = []
for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
    JOINT_RANGE.append(upper - lower)
GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]
GRIPPER_OPEN_STATE = [0.036, -0.036]
GRIPPER_CLOSED_STATE = [0.015, -0.015]
def bin_sort_hash(obj_name):
    if obj_name in BIN_SORT_OBJECTS:
        return BIN_SORT_OBJECTS.index(obj_name)
    else:
        print(obj_name)
        assert False
ACTION_DIM = 7


class SimEnv(gym.Env):
    def __init__(
        self,
        container1_name="bowl_small_pos1",
        container2_name="bowl_small_pos2",
        fixed_container_position=False,
        config_type="default",
        rand_obj=False,
        bin_obj=False,
        num_objects=2,
        obj_scale_default=0.75,
        obj_orientation_default=(0, 0, 1, 0),
        trunc=0,
        specific_task_id=True,
        desired_task_id=(25,),
        # For WidowEnv
        control_mode="discrete_gripper",
        observation_mode="pixels",
        observation_img_dim=100,
        transpose_image=False,
        object_names=("beer_bottle", "gatorade"),
        object_scales=(0.75, 0.75),
        object_orientations=((0, 0, 1, 0), (0, 0, 1, 0)),
        object_position_high=(0.7, 0.27, -0.30),
        object_position_low=(0.5, 0.18, -0.30),
        target_object="gatorade",
        load_tray=True,
        num_sim_steps=10,
        num_sim_steps_reset=50,
        num_sim_steps_discrete_action=75,
        reward_type="grasping",
        grasp_success_height_threshold=-0.25,
        grasp_success_object_gripper_threshold=0.1,
        xyz_action_scale=0.2,
        abc_action_scale=20.0,
        gripper_action_scale=20.0,
        ee_pos_low=(0.4, -0.2, -0.34),
        ee_pos_high=(0.8, 0.4, -0.1),
        camera_target_pos=(0.6, 0.2, -0.28),
        camera_distance=0.29,
        camera_roll=0.0,
        camera_pitch=-40,
        camera_yaw=180,
        gui=False,
        in_vr_replay=False,
        terminate_on_success=True,
        max_reward=1,
        objects_in_container=True,
        # For RobotEnv
        hz=5,
        DoF=6,
        # randomize arm position on reset
        randomize_ee_on_reset=True,
        # init joint angles
        neutral_joint_angles=None,
        # allows user to pause to reset reset of the environment
        pause_after_reset=False,
        # observation space configuration
        qpos=True,
        ee_pos=True,
        
        # multi-task setup
        use_task_id=False,
        task_dim=-1,
        default_task_ids=[0],
        backward_task_ids=None,
        # pass IP if not running on NUC
        ip_address=None,
        # for state only experiments
        goal_state=None,
        # specify path length if resetting after a fixed length
        max_length_reset=5000,
        max_path_length=40,
        # use local cameras, else use images from NUC
        local_cameras=False,
        ee_pos_bound=None,
        gripper_flip_value=False,
        dummy=False,
        name=None,
        running_reset_free=False,
        normalize_obs=False,
        task="forward",
        debug=False,
        dense_reward=False,
        if_rlpd=False,
        state_only=False,
        random_ori=False,
        random_z_offset=-0.25,
        min_distance_from_object=0.11,
        success_open_gripper=False,
        shadow=False,
        wrist_cam=False
    ):
        super().__init__()
        self.wrist_cam = wrist_cam
        print(normalize_obs, "normalize observation flag")
        self.shadow = shadow
        self.success_open_gripper = success_open_gripper
        self.random_ori = random_ori
        self.random_z_offset = random_z_offset
        self.state_only = state_only
        print("state only", self.state_only)
        self.dense_reward = dense_reward
        self._episode_count = 0
        self._max_path_length = max_path_length
        self._curr_path_length = 0
        self.n_life = 0
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._pause_after_reset = pause_after_reset
        self._normalize_obs = normalize_obs
        self.DoF = DoF
        self.name = name
        # Copied from widow250_binsort.py
        if specific_task_id:
            self.num_objects = len(desired_task_id)
        elif rand_obj:
            self.num_objects = num_objects
        else:
            self.num_objects = 2

        self.rand_obj = rand_obj
        self.specific_task_id = specific_task_id
        self.desired_task_id = desired_task_id
        self._switched_task_reset = False
        self.bin_obj = bin_obj
        self.trunc = max(min(trunc, len(BIN_SORT_OBJECTS)), self.num_objects)
        print("trunc", self.trunc)

        max_reward = self.num_objects
        object_scales = [obj_scale_default] * self.num_objects
        object_orientations = [obj_orientation_default] * self.num_objects

        if specific_task_id:
            object_names = tuple([BIN_SORT_OBJECTS[x] for x in desired_task_id])
        elif rand_obj:
            if self.trunc == 0:
                object_names = tuple(
                    np.random.choice(
                        BIN_SORT_OBJECTS, size=self.num_objects, replace=False
                    )
                )
            else:
                object_names = tuple(
                    np.random.choice(
                        BIN_SORT_OBJECTS[: self.trunc],
                        size=self.num_objects,
                        replace=False,
                    )
                )
        else:
            object_names = ("ball", "pepsi_bottle")

        self.container1_name = container1_name
        self.container2_name = container2_name

        ct = None
        if config_type == "default":
            ct = CONTAINER_CONFIGS_BIN_SORT
        else:
            assert False, "Invalid config type"

        container_config = ct[self.container1_name]
        print("Container config:", container_config)
        self.fixed_container_position = fixed_container_position
        if self.fixed_container_position:
            self.container_position_low = container_config["container_position_default"]
            self.container_position_high = container_config[
                "container_position_default"
            ]
        else:
            self.container_position_low = container_config["container_position_low"]
            self.container_position_high = container_config["container_position_high"]
        self.container_position_z = container_config["container_position_z"]
        self.container_orientation = container_config["container_orientation"]
        self.container_scale = container_config["container_scale"]
        self.min_distance_from_object = min_distance_from_object #container_config["min_distance_from_object"]

        container2_config = ct[self.container2_name]
        print("Container config:", container2_config)
        if self.fixed_container_position:
            self.container2_position_low = container2_config[
                "container_position_default"
            ]
            self.container2_position_high = container2_config[
                "container_position_default"
            ]
        else:
            self.container2_position_low = container2_config["container_position_low"]
            self.container2_position_high = container2_config["container_position_high"]
        self.container2_position_z = container2_config["container_position_z"]
        self.container2_orientation = container2_config["container_orientation"]
        self.container2_scale = container2_config["container_scale"]
        self.min_distance_from_object2 = min_distance_from_object#container2_config["min_distance_from_object"]

        self.place_success_height_threshold = container_config[
            "place_success_height_threshold"
        ]
        self.place_success_radius_threshold = container_config[
            "place_success_radius_threshold"
        ]

        target_object = np.random.choice(object_names)
        camera_distance = 0.4

        # Copied from Widow250.py
        print(object_names)

        self.render_enabled = True

        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.observation_img_dim = observation_img_dim
        self.transpose_image = transpose_image
        self.max_reward = max_reward
        self.objects_in_container = objects_in_container

        self.num_sim_steps = num_sim_steps
        self.num_sim_steps_reset = num_sim_steps_reset
        self.num_sim_steps_discrete_action = num_sim_steps_discrete_action

        self.reward_type = reward_type
        self.grasp_success_height_threshold = grasp_success_height_threshold
        self.grasp_success_object_gripper_threshold = (
            grasp_success_object_gripper_threshold
        )

        self.gui = gui
        self.debug = debug

        # TODO(avi): Add limits to ee orientation as well
        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        bullet.connect_headless(self.gui)
        # For reset-free tasks
        self._running_reset_free = running_reset_free
        self.task = task

        # object stuff
        assert target_object in object_names
        assert len(object_names) == len(object_scales)
        self.load_tray = load_tray
        self.num_objects = len(object_names)
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self._object_position_low_z = object_position_low[2]
        self.object_names = object_names
        self.target_object = target_object
        self.object_scales = dict()
        self.object_orientations = dict()
        for orientation, object_scale, object_name in zip(
            object_orientations, object_scales, self.object_names
        ):
            self.object_orientations[object_name] = orientation
            self.object_scales[object_name] = object_scale

        self.in_vr_replay = in_vr_replay
        self._load_meshes()

        self.movable_joints = bullet.get_movable_joints(self.robot_id)
        self.end_effector_index = END_EFFECTOR_INDEX
        self.reset_joint_values = RESET_JOINT_VALUES
        self.reset_joint_indices = RESET_JOINT_INDICES

        self.xyz_action_scale = xyz_action_scale
        self.abc_action_scale = abc_action_scale
        self.gripper_action_scale = gripper_action_scale

        self.camera_target_pos = camera_target_pos
        self.camera_distance = camera_distance
        self.camera_roll = camera_roll
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        view_matrix_args = dict(
            target_pos=self.camera_target_pos,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=self.camera_roll,
            up_axis_index=2,
        )
        self._view_matrix_obs = bullet.get_view_matrix(**view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_dim, self.observation_img_dim
        )

        # self._set_action_space()
        # self._set_observation_space()
        self.is_gripper_open = True  # TODO(avi): Clean this up

        self.terminate_on_success = terminate_on_success

        

        # From robot_env
        self.use_desired_pose = False
        self.max_lin_vel = 0.5
        self.max_rot_vel = 0.5
        self.DoF = DoF
        self.hz = hz

        self._max_path_length = max_path_length
        self._curr_path_length = 0
        
        self.max_length_reset = max_length_reset

        # resetting configuration
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._pause_after_reset = pause_after_reset
        self._robot_is_reset = False
        self._neutral_joint_angles = neutral_joint_angles
        self._gripper_flip_value = gripper_flip_value
        if DoF == 3:
            self._action_mode = '3trans'
        elif DoF == 4:
            self._action_mode = '3trans1rot'
        else:
            self._action_mode = '3trans3rot'

        # observation space config
        self._qpos = qpos
        self._ee_pos = ee_pos

        # multitask config
        self.use_task_id = use_task_id
        self.task_dim = task_dim

        # action space
        self.action_space = Box(
            np.array([-1] * (self.DoF + 1)), # dx_low, dy_low, dz_low, dgripper_low
            np.array([ 1] * (self.DoF + 1)), # dx_high, dy_high, dz_high, dgripper_high
        )
        if self.DoF == 3:
            # EE position (x, y, z) + gripper width
            self.ee_space = Box(
                np.array([*self.ee_pos_low, 0.0]),
                np.array([*self.ee_pos_high, 1.0])
            )
        elif self.DoF == 4:
            # EE position (x, y, z) + EE z rot + gripper width
            # self.ee_space = Box(
            #     np.array([0.25, -0.25, 0.0, -180.0, 0.00]),
            #     np.array([0.45, 0.25, 0.25, 180.0, 1.0])
            # )
            self.ee_space = Box(
                np.array([*self.ee_pos_low, -180.0, 0.00]),
                np.array([*self.ee_pos_high, 180.0, 1.0])
            )
        elif self.DoF == 6:
            self.ee_space = Box(
                np.array([*self.ee_pos_low, -180.0, -180.0, -180.0, 0.0]),
                np.array([*self.ee_pos_high, 180.0, 180.0, 180.0, 1.0])
            )
            self.object_space = Box(
                np.array(object_position_low),
                np.array(object_position_high)
            )
            # self.ee_space = Box(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER)
        else:
            raise ValueError(f'DoF cannot be {self.DoF}')
        if ee_pos_bound is not None:
            self.ee_pos_bound = ee_pos_bound
            self.ee_space = Box(
                np.concatenate([self.ee_pos_bound.min_pos, self.ee_space.low[3:]]),
                np.concatenate([self.ee_pos_bound.max_pos, self.ee_space.high[3:]])
            )
        else:
            self.ee_pos_bound = None

        self.ee_pos_init, self.ee_quat_init = None, None
        # self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
        #     self.robot_id, self.end_effector_index
        # )
        # joint limits + gripper
        # self._jointmin = np.array([-3.1416, -1.8850, -2.1468,
        #                            -3.1416, -1.7453, -3.1415, 0.0], dtype=np.float32)
        # self._jointmax = np.array([3.1416, 1.9897, 1.6057,
        #                            -3.1416, 2.1468, 3.1415, 1.0], dtype=np.float32)
        self._jointmin = np.array(JOINT_LIMIT_LOWER)
        self._jointmax = np.array(JOINT_LIMIT_UPPER)
        # joint space + gripper
        self.qpos_space = Box(
            self._jointmin,
            self._jointmax
        )
        
        # final observation space configuration
        env_obs_spaces = {
            'img_obs': Box(0, 255, (100, 100, 3), np.uint8),
            'lowdim_ee': self.ee_space,
            'lowdim_qpos': self.qpos_space,
        }
        if if_rlpd:
            env_obs_spaces = {
                # 'state': self.ee_space,
            }
        if not self._qpos:
            env_obs_spaces.pop('lowdim_qpos', None)
        if not self._ee_pos:
            env_obs_spaces.pop('lowdim_ee', None)
        if use_task_id:
            self.task_id_space = Box(np.zeros((self.task_dim,)),
                                     np.ones((self.task_dim,)))
            env_obs_spaces['task_id'] = self.task_id_space
        if state_only:
            env_obs_spaces = {
                'lowdim_ee': self.ee_space,
                'object_pos': self.object_space
            }
        self._task_id = default_task_ids[0]
        self._forward_task_ids = default_task_ids
        self._backward_task_ids = backward_task_ids
        self.observation_space = Dict(env_obs_spaces)
        print(f'configured observation space: {self.observation_space}')
        self.if_rlpd = if_rlpd
        self.reset()
        

        # super().__init__(*args_parent)
    def normalize_ee_obs(self, obs):
        """Normalizes low-dim obs between [-1,1]."""
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2 + min(x)
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        normalized_obs = (obs - self.ee_space.low) / (self.ee_space.high - self.ee_space.low) * 2 - 1
        return normalized_obs

    def unnormalize_ee_obs(self, obs):
        return (obs + 1) * (self.ee_space.high - self.ee_space.low) / 2 + self.ee_space.low

    def normalize_qpos(self, qpos):
        """Normalizes qpos between [-1,1]."""
        # The ranges for the joint limits are taken from
        # the franka emika page: https://frankaemika.github.io/docs/control_parameters.html
        norm_qpos = (qpos - self.qpos_space.low) / (self.qpos_space.high - self.qpos_space.low) * 2 - 1
        return norm_qpos

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        """
        TODO(avi) This needs to be cleaned up, generate function should only 
                  take in (x,y) positions instead. 
        """
        #assert self.container_position_low[2] == self.object_position_low[2]
        if self.bin_obj:
            self.object_position_low[2] = self.container_position_low[2]
        else:
            self.object_position_low[2] = self._object_position_low_z
        if not self.in_vr_replay:
            (
                positions,
                self.original_object_positions,
            ) = object_utils.generate_object_positions_v3(
                self.object_position_low,
                self.object_position_high,
                [self.container_position_low, self.container2_position_low],
                [self.container_position_high, self.container2_position_high],
                min_distance_large_obj=self.min_distance_from_object,
                num_large=2,
                num_small=self.num_objects,
            )
            self.container_position, self.container2_position = positions
        self.random_obj_poss = self.original_object_positions[:]
        if self.bin_obj or self.get_task() == "backward":
            # Loop through objects and locate their target containers!
            idx = 0
            for object_name in self.object_names:
                if idx == 0:
                    flip = bin_sort_hash(object_name) % 2 == 0
                else:
                    flip = bin_sort_hash(object_name) % 2 == 1
                pos = (
                        [self.container_position[0], self.container_position[1],
                            self.container_position[2]]
                    if flip
                    else [self.container2_position[0], self.container_position[1],
                        self.container_position[2]]
                )
                if "thin" in self.container1_name:
                    object_position = np.random.uniform(low=np.array([-0.035, -0.025, 0]), high=np.array([0.04, 0.025, 0]))
                else:
                    object_position = np.random.uniform(low=np.array([-0.015, -0.02, 0]), high=np.array([0.015, 0.02, 0]))
                pos[-1] = self.original_object_positions[idx][-1]
                pos += object_position
                self.original_object_positions[idx] = pos
                idx += 1
        self.container_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(
            self.container1_name,
            self.container_position,
            self.container_orientation,
            self.container_scale,
        )
        bullet.step_simulation(self.num_sim_steps_reset)
        self.container2_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(
            self.container2_name,
            self.container2_position,
            self.container2_orientation,
            self.container2_scale,
        )
        
        bullet.step_simulation(self.num_sim_steps_reset)
        for object_name, object_position in zip(
            self.object_names, self.original_object_positions
        ):
            default_ori = self.object_orientations[object_name]
            if self.random_ori and not self.bin_obj:
                # Random z axis
                x, y, _ = bullet.quat_to_deg(default_ori)
                new_z = int(np.random.randint(0, 60) - 30)
                default_ori = bullet.deg_to_quat((x, y, new_z))
            elif self.random_ori and self.bin_obj:
                x, y, _ = bullet.quat_to_deg(default_ori)
                default_ori = bullet.deg_to_quat((x, y, int(np.random.randint(0, 60) - 30)))
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=default_ori,
                scale=self.object_scales[object_name],
            )
            bullet.step_simulation(self.num_sim_steps_reset)
    def _set_action_space(self):
        self.action_dim = ACTION_DIM
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.observation_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'img_obs': img_space, 'lowdim_ee': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def step(self, action):
        # TODO: step through the env
        if np.isnan(np.sum(action)):
            print("action", action)
            assert False

        assert len(action) == (self.DoF + 1)
        action = np.clip(action, -1, +1)  # TODO Clean this up
        xyz_action = action[:3]  # ee position actions
        if self.DoF == 3:
            gripper_action = action[3]
        elif self.DoF == 6:
            abc_action = action[3:6]  # ee orientation actions
            gripper_action = action[6]

        ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        joint_states, _ = bullet.get_joint_states(self.robot_id, self.movable_joints)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])

        target_ee_pos = ee_pos + self.xyz_action_scale * xyz_action
        ee_deg = bullet.quat_to_deg(ee_quat)
        if self.DoF == 3:
            target_ee_quat = bullet.deg_to_quat(self._default_angle)
        if self.DoF == 6:
            target_ee_deg = ee_deg + self.abc_action_scale * abc_action
            target_ee_quat = bullet.deg_to_quat(target_ee_deg)   

        if self.control_mode == "continuous":
            num_sim_steps = self.num_sim_steps
            target_gripper_state = gripper_state + [
                -self.gripper_action_scale * gripper_action,
                self.gripper_action_scale * gripper_action,
            ]

        elif self.control_mode == "discrete_gripper":
            if gripper_action > 0.5 and not self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_OPEN_STATE
                self.is_gripper_open = True  # TODO(avi): Clean this up

            elif gripper_action < -0.5 and self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_CLOSED_STATE
                self.is_gripper_open = False  # TODO(avi): Clean this up
            else:
                num_sim_steps = self.num_sim_steps
                if self.is_gripper_open:
                    target_gripper_state = GRIPPER_OPEN_STATE
                else:
                    target_gripper_state = GRIPPER_CLOSED_STATE
                # target_gripper_state = gripper_state
        else:
            raise NotImplementedError
        # print("before", target_ee_pos)
        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_low, self.ee_pos_high)
        target_gripper_state = np.clip(
            target_gripper_state, GRIPPER_LIMITS_LOW, GRIPPER_LIMITS_HIGH
        )
        # print("cliup?", target_ee_pos)
        # print()
        bullet.apply_action_ik(
            target_ee_pos,
            target_ee_quat,
            target_gripper_state,
            self.robot_id,
            self.end_effector_index,
            self.movable_joints,
            lower_limit=JOINT_LIMIT_LOWER,
            upper_limit=JOINT_LIMIT_UPPER,
            rest_pose=RESET_JOINT_VALUES,
            joint_range=JOINT_RANGE,
            num_sim_steps=num_sim_steps,
        )

        info = self.get_info()
        
        reward = self.get_reward(info)
        done = False
        if self.terminate_on_success:
            if reward >= self.max_reward:
                done = True
            elif (
                self.objects_in_container
                and "sort_success" in info
                and info["sort_success"] and not self.bin_obj
            ):
                done = True

                # final_observation = self.get_observation()
                # plt.imshow(final_observation["image"], interpolation="nearest")
                # plt.show()
                # time.sleep(1)
            elif self.bin_obj and "placed_near_target" in info and info["placed_near_target"]:
                done = True
        self._curr_path_length += 1
        if self._max_path_length is not None and self._curr_path_length >= self._max_path_length:
            done = True

        self._robot_is_reset = False
        final_observation = self.get_observation()
        if self.if_rlpd:
            final_observation['pixels'] = self.render_obs()
        return final_observation, reward, done, info
        # obs, reward, done = self.get_observation(), 0.0, False
        # self._robot_is_reset = False
        # return obs, reward, done, {}

    def get_reward(self, info):
        if self.dense_reward:
            if not self.bin_obj and 'sort_success' in info and info['sort_success']:
                return 1.0
            object_name = self.object_names[0]
            pos = bullet.get_object_position(self.objects[object_name])[0][:3]
            ee_pos = self.get_ee_pos()
            # Phase 1, if the robot hasn't picked up the object
            if self.is_gripper_open or np.clip(np.linalg.norm(ee_pos - pos) - 0.1, 0, 0.5) != 0:    
                # print("phase1")            
                return 0.5 - np.clip(np.linalg.norm(ee_pos - pos) - 0.055, 0, 0.5)
            else:
                # Phase 2, helping the object get into the bin
                # Reward is 1 / distance to the bin
                # First get the correct bin pose
                # print("phase 2")
                which_container = bin_sort_hash(object_name) % 2
                target_drop=self.container_position if which_container == 0 else self.container2_position
                target_drop[2] = -0.17
                # Get pos of ee
                pos = bullet.get_object_position(self.objects[object_name])[0][:3]
                return 0.9 - np.clip(np.linalg.norm(pos - target_drop), 0, 0.4)
                
        if self.objects_in_container and not self.bin_obj and 'all_placed' in info and info['all_placed'] and info['place_success_target'] != self.num_objects:
            return (info['place_success_target']-self.num_objects) / len(self.object_names) # number of objects placed wrong
        elif not self.bin_obj:
            return float(info['place_success_target']) / len(self.object_names)
        elif "num_placed_table" in info and self.bin_obj:
            return info["num_placed_table"] / len(self.object_names)

    @property
    def _default_angle(self):
        return np.array([90.0, 0.0, 180.0])

    def reset(self):
        # TODO: reset sim env
        if self.specific_task_id:
            self.object_names = tuple(
                [BIN_SORT_OBJECTS[x] for x in self.desired_task_id]
            )
        elif self.rand_obj:
            if self.trunc == 0:
                self.object_names = tuple(
                    np.random.choice(
                        BIN_SORT_OBJECTS, size=self.num_objects, replace=False
                    )
                )
            else:
                self.object_names = tuple(
                    np.random.choice(
                        BIN_SORT_OBJECTS[: self.trunc],
                        size=self.num_objects,
                        replace=False,
                    )
                )
        else:
            self.object_names = ("ball", "shed")
        if self.debug:
            print("objects in scene", self.object_names)

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]
        self._episode_count += 1
        if self._running_reset_free:
            # Determine if robot is outside distribution
        
            current_ee_pose = self.get_ee_pos()
            check_pose = np.array([current_ee_pose[0], current_ee_pose[1], current_ee_pose[2], 0, 0, 0, 0]).astype(np.float32)
            in_space = self.ee_space.contains(check_pose)
            # print(check_pose, self.ee_space, in_space)
            # print(np.can_cast(check_pose.dtype, np.float32))
            # print(check_pose.shape)
            if self.debug:
                print("is robot in space?", in_space)

            # Determine if object is outside distribution if it is not placed
            cur_info = self.get_info()
            objects_in_space = True
            check_objects_pos = False
            # Only check objects if we are not successful at task

            if self.task == "forward":
                if "all_placed" in cur_info and not cur_info['all_placed']:
                    check_objects_pos = True
                    if self.debug:
                        print("all placed is failed, so need to check objects")
                # else:
                #     import pdb; pdb.set_trace()
            elif self.task == "backward":
                if "placed_near_target" in cur_info and not cur_info['placed_near_target']:
                    check_objects_pos = True
                    if self.debug:
                        print("placed_near_target is failed, so need to check objects")
            if check_objects_pos:
                for object_name in self.object_names:
                    pos = bullet.get_object_position(self.objects[object_name])[0][:2]
                    check_pos = np.array([pos[0], pos[1]]).astype(np.float32)  # Note, that we are not considering z position
                    if self.task == "forward" and not (np.all(pos >= self.object_position_low[:2]) and np.all(pos <= self.object_position_high[:2])):
                        objects_in_space = False
                        if self.debug:
                            print("not in space: ", object_name, check_pos)
                            print("Lower than lower bound", np.all(pos >= self.object_position_low[:2]))
                            print("Higher than upper bound", np.all(pos <= self.object_position_high[:2]))
                    elif self.task == "backward" and "sort_success" in cur_info and not cur_info['sort_success']:
                        # Object is not in the bin
                        objects_in_space = False
                        if self.debug:
                            print(cur_info['place_success_target'], "place_success_target")
                            print(cur_info['all_placed'], "placed at all")
            if self._episode_count % (self.max_length_reset // 40) != 0:
                # self._curr_path_length < self.max_length_reset and objects_in_space:
                # Only move robot
                if self.debug:
                    print("reset robot only")
                bullet.reset_robot(
                    self.robot_id, self.reset_joint_indices, self.reset_joint_values
                )
                bullet.control_gripper(self.robot_id, self.movable_joints, GRIPPER_OPEN_STATE)
                
                if self._randomize_ee_on_reset:
                    self._desired_pose = {'position': self.get_ee_pos(),
                                          'angle': self.get_ee_angle(),
                                          'gripper': 1}
                    self._randomize_reset_pos()
                
                self.is_gripper_open = True
                self._robot_is_reset = True
                self._prev_life = self.n_life
                obs = self.get_observation()
                return obs
            else:
                if self.debug:
                    print("reset environment")
                # import pdb; pdb.set_trace()
                # Otherwise reset environment
                bullet.reset()
                bullet.setup_headless()
                self._load_meshes()
                bullet.reset_robot(
                    self.robot_id, self.reset_joint_indices, self.reset_joint_values
                )
                self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
                    self.robot_id, self.end_effector_index
                )
                self.is_gripper_open = True
                
                obs = self.get_observation()

                self._robot_is_reset = True
                if self._randomize_ee_on_reset:
                    self._desired_pose = {'position': self.get_ee_pos(),
                                        'angle': self.get_ee_angle(),
                                        'gripper': 1}
                    self._randomize_reset_pos()

                if self._pause_after_reset:
                    user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
                    if user_input in ['s', 'S']:
                        time.sleep(5)

                # initialize desired pose correctly for env.step
                self._desired_pose = {'position': self.get_ee_pos(),
                                    'angle': self.get_ee_angle(),
                                    'gripper': 1}

                self._curr_path_length = 0
                self._prev_life = self.n_life

                return obs

        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id, self.reset_joint_indices, self.reset_joint_values
        )
        self.is_gripper_open = True

        obs = self.get_observation()

        self._robot_is_reset = True
        if self._randomize_ee_on_reset:
            self._desired_pose = {'position': self.get_ee_pos(),
                                  'angle': self.get_ee_angle(),
                                  'gripper': 1}
            self._randomize_reset_pos()

        if self._pause_after_reset:
            user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
            if user_input in ['s', 'S']:
                time.sleep(5)

        # initialize desired pose correctly for env.step
        self._desired_pose = {'position': self.get_ee_pos(),
                              'angle': self.get_ee_angle(),
                              'gripper': 1}

        self._curr_path_length = 0
        self._prev_life = self.n_life
        if self.if_rlpd:
            obs['pixels'] = self.render_obs()
        return obs

    def _randomize_reset_pos(self): # train or eval
        '''takes random action along x-y plane, no change to z-axis / gripper'''

        # random_xy = np.random.uniform(-0.5, 0.5, (2,))
        # random_z = np.random.uniform(-0.2, 0.2, (1,))
        # if self.DoF == 4:
        #     random_rot = np.random.uniform(-0.5, 0.5, (1,))
        #     act_delta = np.concatenate([random_xy, random_z, random_rot, np.zeros((1,))])
        # elif self.DoF == 3:
        #     act_delta = np.concatenate([random_xy, random_z, np.zeros((1,))])
        # else:
        #     random_rot = np.random.uniform(-0.5, 0.5, (1,))
        #     act_delta = np.concatenate([random_xy, random_z, random_rot, np.zeros((3,))])
        # for _ in range(10):
        #     self.step(act_delta)
        'Better randomization'
        # This is for the left one (.75, 0.26, -.30)
        # Right bin (.45, 0.26, -.30)
        # bullet.draw((.75, 0.26, -.30), "L") # Increase y is going down, increase Z is up, increase x is left
        # bullet.draw((.45, 0.26, -.30), "R")
        bullet.reset_robot(
            self.robot_id, self.reset_joint_indices, self.reset_joint_values
        )
        _, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        random_z = self.random_z_offset + np.random.uniform(0, 0.15)
        random_y = np.random.uniform(0.2, 0.32)
        random_x = np.random.uniform(0.54, 0.67)
        bullet.apply_action_ik(
            (random_x, random_y, random_z),
            ee_quat,
            GRIPPER_OPEN_STATE,
            self.robot_id,
            self.end_effector_index,
            self.movable_joints,
            lower_limit=JOINT_LIMIT_LOWER,
            upper_limit=JOINT_LIMIT_UPPER,
            rest_pose=RESET_JOINT_VALUES,
            joint_range=JOINT_RANGE,
            num_sim_steps=100,
        )
        

    def get_task(self):
        return self.task

    def get_info(self):

        info = {"grasp_success": False}
        for object_name in self.object_names:
            grasp_success = object_utils.check_grasp_wo_height(
                object_name,
                self.objects,
                self.robot_id,
                self.end_effector_index,
                self.grasp_success_object_gripper_threshold,
            )
            # grasp_success = object_utils.check_grasp(
            #     object_name,
            #     self.objects,
            #     self.robot_id,
            #     self.end_effector_index,
            #     self.grasp_success_height_threshold,
            #     self.grasp_success_object_gripper_threshold,
            # )
            if grasp_success:
                info["grasp_success"] = True
        
        info["placed_near_target"] = True
        info["num_placed_table"] = 0
        for object_name in self.object_names:
            pos = bullet.get_object_position(self.objects[object_name])[0]
            # print(pos, object_name)
            if not (np.all(pos[:2] >= self.object_position_low[:2]) and np.all(pos[:2] <= self.object_position_high[:2]) and np.abs(pos[2] + 0.35) < 0.02) :
                # The block is not in the region so it's not successful
                info["placed_near_target"] = False
            else:
                info["num_placed_table"] += 1
        info["grasp_success_target"] = object_utils.check_grasp(
            self.target_object,
            self.objects,
            self.robot_id,
            self.end_effector_index,
            self.grasp_success_height_threshold,
            self.grasp_success_object_gripper_threshold,
        )

        # From widow bin sort
        threshold_scale=1

        info['place_success'] = False
        info['place_success_target'] = 0
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            place2_success = object_utils.check_in_container(
                object_name, self.objects, self.container2_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            info['place_success'] = info['place_success'] or place_success or place2_success
            
        info['all_placed'] = True
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            place2_success = object_utils.check_in_container(
                object_name, self.objects, self.container2_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            info['all_placed'] = info['all_placed'] and (place_success or place2_success)
        idx = 0
        for object_name in self.object_names:
            if idx == 0:
                which_container = bin_sort_hash(object_name) % 2
            else:
                which_container = (bin_sort_hash(object_name) + 1) % 2
            target_container_position=self.container_position if which_container == 0 else self.container2_position

            curr_place = object_utils.check_in_container(
                object_name, self.objects, target_container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            info['hover_success'] = object_utils.check_over_container(
                object_name, self.objects, target_container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            
            
            if self.success_open_gripper and not self.is_gripper_open:
                info['place_success_target'] = 0
            else:
                info['place_success_target'] = info['place_success_target'] + int(curr_place)
            idx += 1
        info['sort_success'] = info['place_success_target'] == len(self.object_names)

        # if debug:
        #     print('place success target', info['place_success_target'])
        #     print('sort success', info['sort_success'])

        return info

    @property
    def _curr_pos(self):
        # get current robot end-effector position
        return self.get_ee_pos()

    @property
    def _curr_angle(self):
        # get current robot end-effector orientation
        return self.get_ee_angle()

    def render_obs(self):
        if self.wrist_cam:
            cur_pos = self._curr_pos
            cur_pos[1] += 0.05
            view_matrix_args = dict(
                target_pos=cur_pos,
                distance=0.1,
                yaw=0,
                pitch=-90,
                roll=0,
                up_axis_index=2,
            )
            new_view = bullet.get_view_matrix(**view_matrix_args)
            img, depth, segmentation = bullet.render(
            self.observation_img_dim, self.observation_img_dim,
            new_view, self._projection_matrix_obs, shadow=self.shadow)
            return img
        img, depth, segmentation = bullet.render(
            self.observation_img_dim, self.observation_img_dim,
            self._view_matrix_obs, self._projection_matrix_obs, shadow=self.shadow)
        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def get_images(self):
        # get a list of (H, W, 3) camera images
        # note: by default only use one camera (so the list only has one item)
        camera_feed = [self.render_obs()]
        # add images to camera_feed
        return camera_feed

    def get_gripper_state(self):
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray(joint_states[-2:])
        return gripper_state

    def get_ee_pos(self):
        return bullet.get_link_state(self.robot_id, self.end_effector_index)[0]

    def get_ee_angle(self):
        a = bullet.get_link_state(self.robot_id, self.end_effector_index)[1]
        return bullet.quat_to_rad(a)

    def get_joint_positions(self):
        return bullet.get_joint_states(self.robot_id, self.movable_joints)[0]

    def get_joint_velocities(self):
        return bullet.get_joint_states(self.robot_id, self.movable_joints)[1]

    def get_state(self):
        state_dict = {}
        gripper_state = self.get_gripper_state()

        state_dict['control_key'] = 'desired_pose' if \
            self.use_desired_pose else 'current_pose'

        state_dict['desired_pose'] = np.concatenate(
            [self._desired_pose['position'],
             self._desired_pose['angle'],
             [self._desired_pose['gripper']]])
        # ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)

        state_dict['current_pose'] = np.concatenate(
            [self.get_ee_pos(),
             self.get_ee_angle(),
             [gripper_state]])

        state_dict['joint_positions'] = self.get_joint_positions()
        state_dict['joint_velocities'] = self.get_joint_velocities()
        # don't track gripper velocity
        state_dict['gripper_velocity'] = 0
        return state_dict

    def get_observation(self):
        gripper_state = self.get_gripper_state()
        gripper_binary_state = [float(self.is_gripper_open)]
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        ee_deg = bullet.quat_to_deg(ee_quat)
        if self.if_rlpd:
            observation = {
                    # 'state': np.concatenate(
                    #     (ee_pos, ee_deg, gripper_binary_state)),
            }
        elif self.state_only:
            observation = {
                    'lowdim_ee': np.concatenate(
                        (ee_pos, ee_deg, gripper_binary_state)),
                    'object_pos': bullet.get_object_position(self.objects[self.object_names[0]])[0][:3]
                }
        elif self.observation_mode == 'pixels':
            if self.DoF == 6:
                observation = {
                    'lowdim_ee': np.concatenate(
                        (ee_pos, ee_deg, gripper_binary_state)),
                }
            elif self.DoF == 3:
                observation = {
                    'lowdim_ee': np.concatenate(
                        (ee_pos, gripper_binary_state)),
                }
            if self.render_enabled and not self.if_rlpd:
                image_observation = self.render_obs()
                observation['img_obs'] = image_observation
            if self._normalize_obs:
                observation['lowdim_ee'] = self.normalize_ee_obs(observation['lowdim_ee'])
        else:
            raise NotImplementedError
        if self.use_task_id:
            observation['task_id'] = self._task_id
        return observation

    def set_task(self, task, task_id):
        if task != self.task:
            self._switched_task_reset = True
        self.task = task
        self._task_id = task_id        
        if self.task == "forward":
            self.bin_obj = False  
        else:
            self.bin_obj = True  # We expect the object to start in the bin

    def render(self, mode=None, height=None, width=None, camera_id=None):
        if mode == 'video':
            image_obs = self.get_images()[0]
            obs = image_obs
            return obs
        elif self.if_rlpd:
            return self.render_obs()
        else:
            return self.get_observation()
