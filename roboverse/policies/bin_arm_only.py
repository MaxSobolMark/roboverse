import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS, BIN_SORT_OBJECTS
from roboverse.envs.widow250_binsort import bin_sort_hash

class BinArm:

    def __init__(self, env, pick_height_thresh=-0.25, xyz_action_scale=7.0,
                 pick_point_noise=0.00, drop_point_noise=0.00, 
                 correct_bin_per=1.0, open_steps=5, fail_move_on=False, grasp_distance_thresh=0.03, center_back=False, p_place_correct=1):
        self.env = env
        self.pick_height_thresh_noisy = pick_height_thresh \
                                            # + np.random.normal(scale=0.01)
        self.center_back = center_back
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.drop_point_noise = drop_point_noise
        self.correct_bin_per = correct_bin_per
        self.center_back = center_back
        self.grasp_distance_thresh = grasp_distance_thresh
        self.open_steps=open_steps
        self.curr_steps=0
        self.fail_move_on = fail_move_on
        self.start_ee_pos, self.start_ee_ori = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        self.oriented = False
        self.state = 0

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        print("reset")
        self.curr_steps=0

        if hasattr(self.env, 'bin_obj') and self.env.bin_obj and len(self.env.object_names) > 1:
            self.object_to_target = np.random.choice(self.env.object_names[:-1]) # last object ignored if one in bin
        elif hasattr(self.env, 'specific_task_id') and self.env.specific_task_id:
            which_obj_index = self.env.desired_task_id[0]
            self.object_to_target = BIN_SORT_OBJECTS[which_obj_index]
            assert self.object_to_target in self.env.object_names, f'object to target {self.object_to_target} not in env.object_names {self.env.object_names}'
        elif self.env.get_task() == "backward":
            self.object_to_target = np.random.choice(self.env.object_names[:-1])
        else:
            self.object_to_target = np.random.choice(self.env.object_names)
            
        self.get_pickpoint()

        self.drop_point = self.env.container_position if bin_sort_hash(self.object_to_target) % 2 == 0 else self.env.container2_position
        alternate_drop_point = self.env.container_position if bin_sort_hash(self.object_to_target) % 2 == 1 else self.env.container2_position
        self.drop_point = self.drop_point if np.random.rand() < self.correct_bin_per else alternate_drop_point
        if self.env.bin_obj or self.env.get_task() == "backward":
            idx_use = 0
            for idx in range(len(self.env.object_names)):
                if self.object_to_target == self.env.object_names[idx]:
                    idx_use = idx

            self.drop_point = [0.6, 0.265, 0]
            self.drop_point[2] = -0.26
            self.grasp_distance_thresh = 0.02
        else:
            self.drop_point[2] = -0.26
        self.place_attempted = False

        self.reset_pos, self.start_ee_ori = bullet.get_link_state(self.env.robot_id, self.env.end_effector_index)
        self.oriented = False
        self.state = 0
    def get_pickpoint(self):
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        if not self.env.bin_obj and self.env.get_task() != "backward":
            self.pick_point[2] = -0.29
        else:
            self.pick_point[2] = -0.29
    
    def get_action(self):
        # if self.env.desired_task_id[0] in [0, 15, 16] or self.env.bin_obj:
        #     return self.get_action_wo_orient()
        action_angles = [0, 0, 0]
        action_xyz = [0, 0, 0]   
        ee_pos, ee_ori = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        # bullet.visualize_frame(bullet.get_matrix(ee_ori, ee_pos))
        # bullet.visualize_frame(bullet.get_matrix(self.start_ee_ori, ee_pos), width=5)
        object_pos, object_ori = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        # bullet.visualize_frame(bullet.get_matrix(object_ori, object_pos))
        if self.env.bin_obj:
            object_lifted = ee_pos[2] > -0.23
        else:
            object_lifted = ee_pos[2] > self.pick_height_thresh_noisy
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point[:2] - ee_pos[:2])
        gripper_droppoint_dist = np.linalg.norm(self.drop_point[:2] - ee_pos[:2])
        done = False
        DEBUG = True
        if not self.env.bin_obj:
            goal_acc = 0.015 #0.015
        else:
            goal_acc = 0.015
        if np.random.randint(0, 3) == 0:
            action_gripper = [-0.7]
        else:
            action_gripper = [0.7]
        # print(self.state)
        if gripper_pickpoint_dist > self.grasp_distance_thresh and self.state == 0:
            self.get_pickpoint()
            if DEBUG:
                print('moving near object ', "distance: ", gripper_pickpoint_dist)
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            a, b = np.random.normal(scale=0.05, size=(2))
            action_xyz[0] += a
            action_xyz[1] += b
            action_xyz[2] = 0.07
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            # if xy_diff > 0.03:
            #     action_xyz[2] = 0.0
            error_r, dir = bullet.get_error(ee_ori, object_ori)
            error = bullet.rad_to_deg((0, 0, error_r)) / 360
            if dir:
                action_angles = (0, 0, error[2])
            else:
                action_angles = (0, 0, -error[2])
            action_gripper = [0.7]
            # action_gripper = [0.0]
            
            
        
        # elif self.env.is_gripper_open:
        #     self.oriented = True
        #     if DEBUG:
        #         print('peform grasping, gripper open:', self.env.is_gripper_open)
        #     # near the object enough, performs grasping action
        #     action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
        #     action_xyz = [0., 0., 0.]
        #     action_angles = [0., 0., 0.]
        #     action_gripper = [-0.7]
        elif self.state < 6:
            action_xyz = (0, 0, -0.2)
            action_angles = [0., 0., 0.]
            self.state += 1
        elif not object_lifted and self.state < 11:
            if DEBUG:
                print('lift object')
            self.state += 1
            # lifting objects above the height threshold for picking
            # action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_xyz = np.array([0., 0., 0.08]) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            error_r, dir = bullet.get_error(ee_ori, self.start_ee_ori, False)
            error = bullet.rad_to_deg((0, 0, error_r)) / 360
            if dir:
                action_angles = (0, 0, error[2])
            else:
                action_angles = (0, 0, -error[2])
            # action_gripper = [0.]
        elif gripper_droppoint_dist > goal_acc and self.state < 14: #  
            if DEBUG:
                print('move towards container', ee_pos[2], self.drop_point - ee_pos)
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale / 2
            action_xyz[2] = 0.1
            self.state += 1
            action_angles = [0., 0., 0.]
            error_r, dir = bullet.get_error(ee_ori, self.start_ee_ori, False)
            error = bullet.rad_to_deg((0, 0, error_r)) / 360
            if dir:
                action_angles = (0, 0, error[2])
            else:
                action_angles = (0, 0, -error[2])
            
            # action_gripper = [0.]
        elif self.state < 20:
            # already moved above the container; drop object
            if DEBUG:
                print('drop', self.state)
            # a, b = np.random.normal(scale=0.5, size=(2))
            action_xyz = (0, 0, -0.3)
            action_angles = [0., 0., 0.]
            self.state += 1
            # action_gripper = [0.7]
            self.place_attempted = True
            self.curr_steps += 1
            self.oriented = False
            self.reset_pos = ee_pos
            self.state += 1
        elif self.state < 23:
            # already moved above the container; drop object
            if DEBUG:
                print('rise', self.state)
            # a, b = np.random.normal(scale=0.5, size=(2))
            action_xyz = (0, 0, 0.3)
            action_angles = [0., 0., 0.]
            self.state += 1
            # action_gripper = [0.7]
            self.place_attempted = True
            self.curr_steps += 1
            self.oriented = False
            self.reset_pos = ee_pos
            self.state += 1
        elif self.state < 27:
            # already moved above the container; drop object
            if DEBUG:
                print('random')
            a, b = np.random.normal(scale=0.3, size=(2))
            action_xyz = (a, b, 0.1)
            action_angles = [0., 0., 0.]
            self.state += 1
            # action_gripper = [0.7]
            self.place_attempted = True
            self.curr_steps += 1
            self.oriented = False
            self.reset_pos = ee_pos
        else:
            a, b, c = np.random.normal(scale=0.3, size=(3))
            action_xyz = (a, b, c)
            d, e, f = np.random.normal(scale=0.05, size=(3))
            action_angles = [d, e, f]
         
        agent_info = dict(place_attempted=self.place_attempted, done=done)
        
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        action += np.random.normal(scale=0.01, size=(7,))
        return action, agent_info
    def get_action_wo_orient(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point[:2] - ee_pos[:2])
        gripper_droppoint_dist = np.linalg.norm(self.drop_point[:2] - ee_pos[:2])
        done = False
        if not self.env.bin_obj:
            goal_acc = 0.01
        else:
            goal_acc = 0.02
        # print('gripper_pickpoint_dist ', gripper_pickpoint_dist )
        # print('self.env.is_gripper_open ', self.env.is_gripper_open)
        if self.place_attempted:
            # print('Neutral attemtped')
            # Reset after one attempt
            action_xyz = (self.reset_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            self.place_attempted = True
        elif gripper_pickpoint_dist > self.grasp_distance_thresh and self.env.is_gripper_open:
            self.get_pickpoint()
            # print('moving near object ')
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # print('peform grasping, gripper open:', self.env.is_gripper_open)
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # print('lift object')
            # lifting objects above the height threshold for picking
            # action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_xyz = np.array([0., 0., 0.08]) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_droppoint_dist > goal_acc:
            print('move towards container', gripper_droppoint_dist)
            # lifted, now need to move towards the container
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            print('drop')
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True
            self.curr_steps += 1
            self.reset_pos = ee_pos

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info

class PickPlaceOld:

    def __init__(self, env, pick_height_thresh=-0.31):
        self.env = env
        self.pick_height_thresh_noisy = pick_height_thresh \
                                            + np.random.normal(scale=0.01)
        self.xyz_action_scale = 7.0
        self.reset()

    def reset(self):
        self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.place_attempted = False
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)

        container_pos = self.env.container_position
        target_pos = np.append(container_pos[:2], container_pos[2] + 0.15)
        target_pos = target_pos + np.random.normal(scale=0.01)
        gripper_target_dist = np.linalg.norm(target_pos - ee_pos)
        gripper_target_threshold = 0.03

        done = False

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif object_gripper_dist > self.dist_thresh and self.env.is_gripper_open:
            # move near the object
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_target_dist > gripper_target_threshold:
            # lifted, now need to move towards the container
            action_xyz = (target_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info
