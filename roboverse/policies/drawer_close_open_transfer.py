import numpy as np
import roboverse.bullet as bullet


class DrawerCloseOpenTransfer:

    def __init__(self, env):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.04
        self.ending_z = -0.25
        self.top_drawer_offset = np.array([0, 0, 0.02])
        self.reset()

    def reset(self):
        self.drawer_never_opened = True
        offset_coeff = (-1) ** (1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])
        self.reached_pushing_region = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        top_drawer_pos = self.env.get_drawer_pos("drawer_top")
        top_drawer_push_target_pos = (
            top_drawer_pos + np.array([0.15, 0, 0.05]))
        is_gripper_ready_to_push = (
            ee_pos[0] > top_drawer_push_target_pos[0] and
            ee_pos[2] < top_drawer_push_target_pos[2]
        )
        done = False
        neutral_action = [0.0]
        if (not self.env.is_top_drawer_closed() and
                not self.reached_pushing_region and
                not is_gripper_ready_to_push):
            # print("move up and left")
            action_xyz = [0.3, -0.2, -0.15]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.env.is_top_drawer_closed():
            # print("close top drawer")
            self.reached_pushing_region = True
            action_xyz = (top_drawer_pos + self.top_drawer_offset - ee_pos) * 7.0
            action_xyz[0] *= 3
            action_xyz[1] *= 0.6
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_drawer_open()):
            # print('xy - approaching handle')
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        elif (gripper_handle_dist > self.gripper_dist_thresh
                and not self.env.is_drawer_open()):
            # moving down toward handle
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.env.is_drawer_open():
            # print("opening drawer")
            x_command = (-1) ** (1 - self.env.left_opening)
            action_xyz = np.array([x_command, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif ee_pos[2] < self.ending_z:
            action_xyz = [0., 0., 0.5]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        else:
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
            neutral_action = [0.7]
            done = True

        agent_info = dict(done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info
