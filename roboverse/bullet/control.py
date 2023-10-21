import pybullet as p
import numpy as np


def get_joint_states(body_id, joint_indices):
    all_joint_states = p.getJointStates(body_id, joint_indices)
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities)


def get_movable_joints(body_id):
    num_joints = p.getNumJoints(body_id)
    movable_joints = []
    for i in range(num_joints):
        if p.getJointInfo(body_id, i)[2] != p.JOINT_FIXED:
            movable_joints.append(i)
    return movable_joints


def get_link_state(body_id, link_index):
    position, orientation, _, _, _, _ = p.getLinkState(body_id, link_index)
    return np.asarray(position), np.asarray(orientation)


def get_joint_info(body_id, joint_id, key):
    keys = ["jointIndex", "jointName", "jointType", "qIndex", "uIndex",
            "flags", "jointDamping", "jointFriction", "jointLowerLimit",
            "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName",
            "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"]
    value = p.getJointInfo(body_id, joint_id)[keys.index(key)]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return value

def draw(pos, text=""):
    p.addUserDebugText(text, pos)
def apply_action_ik(target_ee_pos, target_ee_quat, target_gripper_state,
                    robot_id, end_effector_index, movable_joints,
                    lower_limit, upper_limit, rest_pose, joint_range,
                    num_sim_steps=5):
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               lowerLimits=lower_limit,
                                               upperLimits=upper_limit,
                                               jointRanges=joint_range,
                                               restPoses=rest_pose,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.01)

    p.setJointMotorControlArray(robot_id,
                                movable_joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                # targetVelocity=0,
                                forces=[500] * len(movable_joints),
                                positionGains=[0.03] * len(movable_joints),
                                # velocityGain=1
                                )
    # set gripper action
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.03)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.03)

    for _ in range(num_sim_steps):
        p.stepSimulation()
def control_gripper(robot_id, movable_joints, target_gripper_state, num_sim_steps=50):
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.03)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.03)
    for _ in range(num_sim_steps):
        p.stepSimulation()

def reset_robot(robot_id, reset_joint_indices, reset_joint_values):
    assert len(reset_joint_indices) == len(reset_joint_values)
    for i, value in zip(reset_joint_indices, reset_joint_values):
        p.resetJointState(robot_id, i, value)


def reset_object(body_id, position, orientation):
    p.resetBasePositionAndOrientation(body_id,
                                      position,
                                      orientation)


def get_object_position(body_id):
    object_position, object_orientation = \
        p.getBasePositionAndOrientation(body_id)
    return np.asarray(object_position), np.asarray(object_orientation)


def step_simulation(num_sim_steps):
    for _ in range(num_sim_steps):
        p.stepSimulation()


def quat_to_deg(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg

def quat_to_rad(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    return euler_rad

def deg_to_quat(deg):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad)
    return quat

def get_error(ee_quat, obj_quat, second_obj=True): # turning negative is false
    floats = np.array(p.getMatrixFromQuaternion(ee_quat)).reshape((3, 3)).T.flatten()
    proj_x, proj_y = floats[6], floats[7] # get z column projected onto xy plan
    if second_obj:
        floats_2 = np.array(p.getMatrixFromQuaternion(obj_quat)).reshape((3, 3)).T.flatten()
        proj_x_2, proj_y_2 = floats_2[3], floats_2[4] # assume y is in xy plane  
    else:
        floats_2 = np.array(p.getMatrixFromQuaternion(obj_quat)).reshape((3, 3)).T.flatten()
        proj_x_2, proj_y_2 = floats_2[6], floats_2[7] # assume y is in xy plane
    clockwise = True

    if proj_y_2 < 0:
        proj_y_2 *= -1
        proj_x_2 *= -1
    angle_ee = np.arccos(proj_x)
    angle_ob = np.arccos(proj_x_2)
   
    if angle_ee < angle_ob:
        clockwise = True
    else:
        clockwise = False
    ee_norm = np.array([proj_x, proj_y])
    ee_norm = ee_norm / np.linalg.norm(ee_norm)
    obj_norm = np.array([proj_x_2, proj_y_2])
    obj_norm = obj_norm / np.linalg.norm(obj_norm)
    
    # Now we have two normalized vectors we want to get the angle between them
    mag = float(ee_norm.T @ obj_norm)
    angle = np.arccos(mag)
    return angle, clockwise # 0 to pi
def get_matrix(quat, pos):
    floats = p.getMatrixFromQuaternion(quat)
    m = np.array([[floats[0], floats[1], floats[2], pos[0]], [floats[3], floats[4], floats[5], pos[1]], 
                  [floats[6], floats[7], floats[8], pos[2]], [0, 0, 0, 1]])
    return m
def visualize_frame(
    tmat: np.ndarray, length: float = 1, width: float = 3, lifetime: float = 0
):
    """
    Written by Dan Morton: 
    Adds RGB XYZ axes to the Pybullet GUI for a speficied transformation/frame/pose

    Args:
        tmat (np.ndarray): Transformation matrix specifying a pose w.r.t world frame, shape (4, 4)
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    x_color = [1, 0, 0]  # R
    y_color = [0, 1, 0]  # G
    z_color = [0, 0, 1]  # B
    origin = tmat[:3, 3]
    x_endpt = origin + tmat[:3, 0] * length
    y_endpt = origin + tmat[:3, 1] * length
    z_endpt = origin + tmat[:3, 2] * length
    x_ax_id = p.addUserDebugLine(origin, x_endpt, x_color, width, lifetime)
    y_ax_id = p.addUserDebugLine(origin, y_endpt, y_color, width, lifetime)
    z_ax_id = p.addUserDebugLine(origin, z_endpt, z_color, width, lifetime)
    return x_ax_id, y_ax_id, z_ax_id

def close(cur, des):
    return quat_to_deg(cur - des)[2]
def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])
