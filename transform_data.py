import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import io

def get_gripper_rotation_offset(data_version, robot_id):
    """根据机器人类型获取旋转偏移
    
    Args:
        robot_id: 机器人类型字符串，如aloha_3, Franka_3
        
    Returns:
        Rotation对象，表示该机器人类型的旋转偏移
    """
    if data_version == "v1":
        return Rotation.from_euler("xyz", [0, 0, 0])
    elif data_version == "v2":
        if "aloha" in robot_id.lower():
            # aloha使用当前的offset
            return Rotation.from_euler("xyz", [0, -np.pi / 2, 0])
        elif "arx5" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, 0, 0])
        elif "ur5" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, -np.pi / 2, 0]) * Rotation.from_euler("xyz", [np.pi / 2, 0, 0])
        elif "franka_3" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, np.pi / 2, 0]) * Rotation.from_euler("xyz", [np.pi, 0, 0])
        elif "franka_4" in robot_id.lower():
            return Rotation.from_euler("xyz", [-1.40893619, -1.56357505, -1.75949757])
            # return Rotation.from_euler("xyz", [0, np.pi / 2, 0]) * Rotation.from_euler("xyz", [np.pi, 0, 0])
        else:
            raise ValueError(f"Unknown robot type: {robot_id}")
    elif data_version == "v2_euler":
        if "aloha" in robot_id.lower():
            # aloha使用当前的offset
            return Rotation.from_euler("xyz", [0, -np.pi / 2, 0])
        elif "arx5" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, 0, 0])
        elif "ur5" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, -np.pi / 2, 0]) * Rotation.from_euler("xyz", [np.pi / 2, 0, 0]) \
                * Rotation.from_euler("xyz", [0, np.pi / 2, 0])
        elif "franka_3" in robot_id:
            return Rotation.from_euler("xyz", [0, np.pi / 2, 0]) * Rotation.from_euler("xyz", [np.pi, 0, 0]) \
                * Rotation.from_euler("xyz", [0, np.pi / 2, 0])
        elif "franka_4" in robot_id:
            return Rotation.from_euler("xyz", [-1.40893619, -1.56357505, -1.75949757]) * Rotation.from_euler("xyz", [0, np.pi / 2, 0])
            # return Rotation.from_euler("xyz", [0, -np.pi / 2, 0]) * Rotation.from_euler("xyz", [np.pi, 0, 0]) \
            #     * Rotation.from_euler("xyz", [0, np.pi / 2, 0])
        else:
            raise ValueError(f"Unknown robot type: {robot_id}")

def get_world_rotation_offset(data_version, robot_id):
    """
    根据机器人类型获取世界坐标系下的旋转偏移

    Args:
        data_version: 数据版本，如v1, v2, v2_euler
        robot_id: 机器人类型字符串，如aloha_3, Franka_3

    Returns:
        world_rotation_offset: 世界坐标系下的旋转偏移，Rotation对象。
    """
    if data_version == "v1":
        return Rotation.from_euler("xyz", [0, 0, 0])
    elif data_version == "v2" or data_version == "v2_euler":
        if "aloha" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, 0, 0])
        elif "arx5" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, 0, 0])
        elif "ur5" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, 0, np.pi])
        elif "franka" in robot_id.lower():
            return Rotation.from_euler("xyz", [0, 0, 0])
        else:
            raise ValueError(f"Unknown robot type: {robot_id}")

def transform_images(images, data_version, robot_id):
    """
    Args:
        images: 原始图像，字典，key为摄像头名称（包括high, left_hand, right_hand），value为图像对象。
        data_version: 数据版本，如v1, v2, v2_euler
        robot_id: 机器人类型字符串，如aloha_3, Franka_3

    Returns:
        transformed_images: 变换后的图像，字典，key为摄像头名称（包括high, left_hand, right_hand），value为图像对象。
    """
    if data_version == "v1":
        return images
    elif data_version == "v2" or data_version == "v2_euler":
        if "ur5_3" in robot_id.lower():
            # rotate left hand camera by 180 degrees
            left_hand_image = images['left_hand']
            left_hand_image = left_hand_image.transpose(Image.ROTATE_180)
            images['left_hand'] = left_hand_image
            return images
        elif "franka" in robot_id.lower():
            # rotate left hand camera by 180 degrees
            left_hand_image = images['left_hand']
            left_hand_image = left_hand_image.transpose(Image.ROTATE_180)
            images['left_hand'] = left_hand_image
            return images
        else:
            return images

def transform_action(action, data_version, robot_id):
    """
    Args: 
        action: 原始动作，[x, y, z, quaternion(xyzw)/euler(xyz), gripper]，根据不同机器人本体和机械臂数量，action的长度不同。
        data_version: 数据版本，如v1, v2, v2_euler
        robot_id: 机器人类型字符串，如aloha_3, Franka_3

    Returns:
        transformed_action: 变换后的动作，[x, y, z, quaternion(xyzw)/euler(xyz), gripper]，根据不同机器人本体和机械臂数量，action的长度不同。
    """
    if 'arx5' in robot_id.lower():
        position, rotation, gripper = action[:3], Rotation.from_euler("xyz", action[3:6]), action[6:]
        inputs = [(position, rotation, gripper,)]
        rotation_format = 'euler'
    elif 'aloha' in robot_id.lower():
        position_left, rotation_left, gripper_left = action[:3], Rotation.from_quat(action[3:7]), action[7:8]
        position_right, rotation_right, gripper_right = action[8:11], Rotation.from_quat(action[11:15]), action[15:]
        inputs = [(position_left, rotation_left, gripper_left), (position_right, rotation_right, gripper_right)]
        rotation_format = 'quaternion'
    elif 'ur5' in robot_id.lower():
        position, rotation, gripper = action[:3], Rotation.from_quat(action[3:7]), action[7:]
        inputs = [(position, rotation, gripper,)]
        rotation_format = 'quaternion'
    elif 'franka' in robot_id.lower():
        position, rotation, gripper = action[:3], Rotation.from_quat(action[3:7]), action[7:]
        inputs = [(position, rotation, gripper,)]
        rotation_format = 'quaternion'
    else:
        raise ValueError(f"Unknown robot type: {robot_id}")

    transformed_inputs = []

    world_rotation_offset = get_world_rotation_offset(data_version, robot_id)
    gripper_rotation_offset = get_gripper_rotation_offset(data_version, robot_id)

    for position, rotation, gripper in inputs:
        position = world_rotation_offset.as_matrix() @ position
        rotation = world_rotation_offset * rotation * gripper_rotation_offset
        rotation = rotation.as_quat() if rotation_format == 'quaternion' else rotation.as_euler("xyz")
        transformed_input = np.concatenate([position, rotation, gripper])
        transformed_inputs.append(transformed_input)
    
    transformed_inputs = np.concatenate(transformed_inputs, axis=0)

    return transformed_inputs

def transform_action_inverse(action, data_version, robot_id):
    """
    Args:
        action: 模型输出动作，为7维向量，[x, y, z, euler(xyz), gripper]
        data_version: 数据版本，如v1, v2, v2_euler
        robot_id: 机器人类型字符串，如aloha_3, Franka_3

    Returns:
        transformed_action: 变换后的动作，[x, y, z, euler(xyz), gripper]
    """
    position, rotation, gripper = action[:3], Rotation.from_euler("xyz", action[3:6]), action[6:]
    world_rotation_offset = get_world_rotation_offset(data_version, robot_id)
    gripper_rotation_offset = get_gripper_rotation_offset(data_version, robot_id)
    position = world_rotation_offset.inv().as_matrix() @ position
    rotation = world_rotation_offset.inv() * rotation * gripper_rotation_offset.inv()
    data = np.concatenate([position, rotation.as_euler("xyz"), gripper])
    return data

def preprocess(input_data, data_version, robot_id, action_space='pos'):
    """
    预处理输入数据，将输入数据变换到新的坐标系，并进行图像变换。
    Args:
        input_data: 原始输入数据，字典，key为数据名称，value为数据对象。
        data_version: 数据版本，如v1, v2, v2_euler
        robot_id: 机器人类型字符串，如aloha_3, Franka_3
        action_space: 动作空间，如pos, joint

    Returns:
        transformed_input_data: 变换后的输入数据，字典，key为数据名称，value为数据对象。
    """
    if action_space == 'joint':
        return input_data
    images = {}
    raw_images = input_data['images']
    for key in raw_images:
        image_stream = io.BytesIO(raw_images[key])
        images[key] = Image.open(image_stream)

    input_data['images'] = transform_images(images, data_version, robot_id)
    
    input_data['action'] = [v[0] if isinstance(v, list) else v for v in input_data['action']]
    input_data['action'] = np.array(input_data['action'])
    # transform action to new coordinate system
    input_data['action'] = transform_action(input_data['action'], data_version, robot_id)

    return input_data

def postprocess(action, data_version, robot_id, action_space='pos'):
    """
    后处理模型输出动作，将动作变换回原始坐标系。
    Args:
        action: 模型输出动作，为14维向量，[x, y, z, euler(xyz), gripper, x, y, z, euler(xyz), gripper]
        data_version: 数据版本，如v1, v2, v2_euler
        robot_id: 机器人类型字符串，如aloha_3, Franka_3
        action_space: 动作空间，如pos, joint

    Returns:
        transformed_action: 变换后的动作，[x, y, z, euler(xyz), gripper, x, y, z, euler(xyz), gripper]
    """
    if action_space == 'joint':
        return action

    transformed_actions = []
    for t in range(action.shape[0]):
        left_action = action[t, :7]
        right_action = action[t, 7:14]
        left_transformed_action = transform_action_inverse(left_action, data_version, robot_id)
        right_transformed_action = transform_action_inverse(right_action, data_version, robot_id)
        transformed_actions.append(np.concatenate([left_transformed_action, right_transformed_action], axis=0))
    transformed_actions = np.stack(transformed_actions, axis=0)
    return transformed_actions
    
