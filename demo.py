import argparse
import logging
import sys
import os
import io
import time
import pickle
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from datetime import datetime
import matplotlib
import traceback
import yaml
import copy
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from task_info import TASK_INFO
import os
from scipy.spatial.transform import Rotation
import time
import json
from transform_data import preprocess, postprocess

logging.basicConfig(
    filename='mylogfile.log',  # Log file name
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)


TASK_ROBOT_MAPPING = {
    'arrange_fruits_in_basket':     'robochallenge_UR5',
    'hang_toothbrush_cup':          'robochallenge_UR5',
    'sort_books':                   'robochallenge_UR5',
    'shred_scrap_paper':            'robochallenge_UR5',
    'stack_color_blocks':           'robochallenge_UR5',
    'set_the_plates':               'robochallenge_UR5',
    'plug_in_network_cable':        'robochallenge_aloha',
    'make_vegetarian_sandwich':     'robochallenge_aloha',
    'sweep_the_rubbish':            'robochallenge_aloha',
    'scan_QR_code':                 'robochallenge_aloha',
    'put_pen_into_pencil_case':     'robochallenge_aloha',
    'pour_fries_into_plate':        'robochallenge_aloha',
    'turn_on_faucet':               'robochallenge_aloha',
    'stack_bowls':                  'robochallenge_aloha',
    'clean_dining_table':           'robochallenge_aloha',
    'put_opener_in_drawer':         'robochallenge_aloha',
    'stick_tape_to_box':            'robochallenge_aloha',
    'arrange_flowers':              'robochallenge_arx5',
    'arrange_paper_cups':           'robochallenge_arx5',
    'open_the_drawer':              'robochallenge_arx5',
    'water_potted_plant':           'robochallenge_arx5',
    'wipe_the_table':               'robochallenge_arx5',
    'sort_electronic_products':     'robochallenge_arx5',
    'turn_on_light_switch':         'robochallenge_arx5',
    'put_cup_on_coaster':           'robochallenge_arx5',
    'place_shoes_on_rack':          'robochallenge_arx5',
    'fold_dishcloth':               'robochallenge_arx5',
    'search_green_boxes':           'robochallenge_arx5',
    'move_objects_into_box':        'robochallenge_Franka',
    'press_three_buttons':          'robochallenge_Franka'
 }

def isstatic(action_pred, threshold=0.001):
    # action_pred ：{'follow1_pos': [list], 'follow2_pos': [list]}
    if action_pred is None:
        return False
        
    follow1_pos = action_pred['follow1_pos']
    follow2_pos = action_pred['follow2_pos']
    left_action = np.array(follow1_pos) # horizon * action_dim
    right_action = np.array(follow2_pos) # horizon * action_dim
    
    # Check if the range (max - min) of each dimension is within the threshold
    left_static = np.all(np.ptp(left_action, axis=0) < threshold)
    right_static = np.all(np.ptp(right_action, axis=0) < threshold)
    
    return left_static and right_static

class WallxPolicy:
    """
    Example policy class.
    Users should implement the __init__ and run_policy methods according to their own logic.
    """

    def __init__(self, checkpoint_path, dataset_name, config_path=None, end_ratio=None, task_name="arrange_flowers", duration=0.05,
                data_version="v1", visualize_path='visualize', replay_episode_id=0, action_predict_mode="diffusion", action_space="pos"
            ):
        """
        Initialize the policy.
        Args:
            checkpoint_path (str): Path to the model checkpoint file.
            action_type (str): Action type, 'joint' or 'pos'. Default is 'joint'.
        """
        from scripts.infer_robochallenge import WallxInferArgs, WallxInfer

        self.args = WallxInferArgs()
        if end_ratio is not None:
            self.args.action_end_ratio = end_ratio
        
        self.args.checkpoint_path = checkpoint_path
        self.args.dataset_name = dataset_name
        self.args.config_path = config_path
        if config_path is None:
            config_path = os.path.join(checkpoint_path, 'config.yml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dof_config = self.config['dof_config']
        self.agent_pos_config = self.config['agent_pos_config']
        self.action_predict_mode = action_predict_mode
        self.action_space = action_space
        self.infer = WallxInfer(self.args)
        self.task_name = task_name
        self.duration = duration
        self.frame_idx = 0
        # Set to True to disable replay data loading and visualization
        self.dryrun = True
        self.data_version = data_version
        self.visualize_path = visualize_path
        self.robot_id_full = TASK_INFO[self.task_name]['robot_id']
 

    def run_policy(self, input_data):
        """
        Run inference using the policy/model.
        Args:
            input_data: Input data for inference.
        Returns:
            list: Inference results.
        """
        # TODO: Implement your inference logic here (e.g., GPU model inference)
        input_data_org = copy.deepcopy(input_data)
        input_data = preprocess(input_data, self.data_version, self.robot_id_full, action_space=self.action_space)
        images = input_data['images']
        task_name = input_data['task_name'] if 'task_name' in input_data else self.task_name
        action_gt_raw = input_data['action'] # Ground Truth (Current State)
        timestamp = input_data['timestamp']
        print("input_data['action']:",input_data['action'])
        # ----------------------------------------------------------------------
        # 数据转换与准备 (State & View Construction)
        # ----------------------------------------------------------------------
        pil_images = {}
        for key in images:
            if isinstance(images[key], bytes):
                image_stream = io.BytesIO(images[key])
                pil_images[key] = Image.open(image_stream)
            else:
                pil_images[key] = images[key]

        task_info = TASK_INFO[task_name]
        robot_id_mapped, instruction = task_info['robot_id'], task_info['prompt']
        robot_id = robot_id_mapped.split('_')[0]
        if self.data_version == "v1":
            dataset_name = f'robochallenge_{robot_id}'
        elif self.data_version == "v2":
            dataset_name = f'robochallenge_v2_{robot_id}'
        elif self.data_version == "v2_euler":
            dataset_name = f'robochallenge_v2_euler_{robot_id}'
        else:
            raise ValueError(f"Invalid data version: {self.data_version}")
        self.args.dataset_name = dataset_name
        
        # 准备 State 格式
        state = {}
        if len(action_gt_raw) >= 14:
            single_arm_dimension = len(action_gt_raw) // 2
        else:
            single_arm_dimension = len(action_gt_raw)

        # 辅助函数：四元数转欧拉角
        def quaternion_to_euler(pose):
            quat = pose[3:7]
            r = Rotation.from_quat(quat)
            euler = r.as_euler('xyz', degrees=False).tolist()
            return np.concatenate([pose[:3], euler, pose[7:]])

        # 辅助函数：欧拉角转四元数
        def euler_to_quaternion(pose, is_2d=True, q=True):
            if is_2d:
                euler = pose[:,3:6]
                r = Rotation.from_euler('xyz', euler)
                quat = r.as_quat()
                if robot_id.lower() != 'ur5' and robot_id.lower() != 'franka':
                    if q:
                        fix_quat = np.array([q if q[-1]>0 else -q for q in quat])
                    else:
                        fix_quat = np.array([q if q[-1]<0 else -q for q in quat])
                else:
                    if q:
                        fix_quat = np.array([q if q[0]>0 else -q for q in quat])
                    else:
                        fix_quat = np.array([q if q[0]<0 else -q for q in quat])
                new_pose = np.concatenate([pose[:, :3], fix_quat, pose[:, 6:]], axis=1)
            else:
                euler = pose[3:6]
                r = Rotation.from_euler('xyz', euler)
                quat = r.as_quat()
                if robot_id.lower() != 'ur5' and robot_id.lower() != 'franka':
                    if q:
                        fix_quat = np.array([q if q[-1]>0 else -q for q in quat])
                    else:
                        fix_quat = np.array([q if q[-1]<0 else -q for q in quat])
                else:
                    if q:
                        fix_quat = np.array([q if q[0]>0 else -q for q in quat])
                    else:
                        fix_quat = np.array([q if q[0]<0 else -q for q in quat])
                new_pose = np.concatenate([pose[:3], fix_quat, pose[6:]])
            return new_pose


        # 构建 state 字典
        state_first_q1, state_first_q2 = False, False
        if len(action_gt_raw) >= 14:
            if single_arm_dimension == 7:
                state['follow1_pos'] = action_gt_raw[0:single_arm_dimension]
                # state['follow1_pos'] = []
                state['follow2_pos'] = action_gt_raw[single_arm_dimension:2*single_arm_dimension]
            else:
                state['follow1_pos'] = quaternion_to_euler(action_gt_raw[0:single_arm_dimension])
                state['follow2_pos'] = quaternion_to_euler(action_gt_raw[single_arm_dimension:2*single_arm_dimension])
                if robot_id.lower() != 'ur5':
                    state_first_q1 = action_gt_raw[6] > 0
                    state_first_q2 = action_gt_raw[single_arm_dimension + 6] > 0
                else:
                    state_first_q1 = action_gt_raw[3] > 0
                    state_first_q2 = action_gt_raw[single_arm_dimension + 3] > 0
        else:
            if single_arm_dimension == 7:
                state['follow1_pos'] = [0.0] * single_arm_dimension 
                state['follow2_pos'] = action_gt_raw[0:single_arm_dimension]
            else:
                state['follow1_pos'] = [0.0] * 7
                state['follow2_pos'] = quaternion_to_euler(action_gt_raw[0:single_arm_dimension])
                state_first_q1 = True
                state_first_q2 = action_gt_raw[6] > 0 if robot_id.lower() != 'ur5' else action_gt_raw[3] > 0

        # build views dict
        views = {}
        if robot_id.lower() == 'aloha':
            key_mapping = {'high': 'camera_front', 'left_hand': 'camera_left', 'right_hand': 'camera_right'}  
        elif robot_id.lower() == 'ur5':
            key_mapping = {'right_hand': 'camera_global', 'left_hand': 'camera_right'}
        elif robot_id.lower() == 'franka':
            key_mapping = {'right_hand': 'camera_global', 'high': 'camera_side', 'left_hand': 'camera_right'}
        elif robot_id.lower() == 'arx5':
            key_mapping = {
                'left_hand': 'camera_right', ## NOTE
                'right_hand': 'camera_global',
                'high': 'camera_side',
            }
        for old_key, new_key in key_mapping.items():
            if old_key in pil_images:
                img_array = np.array(pil_images[old_key])
                if img_array.ndim == 2: img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 4: img_array = img_array[:, :, :3]
                views[new_key] = img_array[None, ...]
        
        agent_pos_mask = torch.ones(sum(self.agent_pos_config.values()))
        dof_mask = torch.ones(sum(self.dof_config.values()))
        if robot_id.lower() in ['arx5', 'ur5', 'franka']:
            dof_mask[:sum(self.agent_pos_config.values())//2] = 0
            agent_pos_mask[:sum(self.agent_pos_config.values())//2] = 0

        valid_action_dim = 7 if robot_id.lower() in ['arx5', 'ur5', 'franka'] else 14
        action_pred = self.infer.run_infer_robochallenge(state, views, instruction, valid_action_dim=valid_action_dim, action_predict_mode=self.action_predict_mode)
        # 检查是否为静止动作，如果是，则换成diffusion模式
        if self.action_predict_mode == "ar" and (isstatic(action_pred) or action_pred is None):
            print("💥💥💥action_pred is static or None, switch to diffusion mode 💥💥💥", flush=True)
            action_pred = self.infer.run_infer_robochallenge(state, views, instruction, valid_action_dim=valid_action_dim, action_predict_mode="diffusion")

       
        # Convert action_pred dict to T*action_dim format
        # For joint control with dual-arm robot: action_dim = 14
        # Format: [6 joints left, 1 gripper left, 6 joints right, 1 gripper right]
        # Note: action_type should match the action_type used in main() function
        actions = self._convert_action_pred_to_actions(action_pred)

        if robot_id.lower() == 'aloha':
            # actions[:, 6] = [a+0.01 if a>0.06 else a for a in actions[:, 6]]
            actions[:, 13] = [a+0.01 if a>0.06 else a for a in actions[:, 13]]

            # actions[:, 6] = [a-0.01 if a<0.04 else a for a in actions[:, 6]]
            actions[:, 13] = [a-0.01 if a<0.03 else a for a in actions[:, 13]]

        actions_rpy_org = copy.deepcopy(actions)
        actions = postprocess(actions, self.data_version, self.robot_id_full, action_space=self.action_space)
        actions_rpy = copy.deepcopy(actions)

        if robot_id.lower() in ['arx5', 'ur5', 'franka']:
            actions = actions[:, 7:]
            if single_arm_dimension == 8:
                actions = euler_to_quaternion(actions, is_2d=True)
        else:
            if single_arm_dimension == 8:
                actions_left = euler_to_quaternion(actions[:, :7], is_2d=True, q=state_first_q1)
                actions_right = euler_to_quaternion(actions[:, 7:], is_2d=True, q=state_first_q2)
                actions = np.concatenate([actions_left, actions_right], axis=1)
        
        self.frame_idx += 1
        return actions.tolist()
    
    def _convert_action_pred_to_actions(self, action_pred):
        follow1_pos = action_pred.get('follow1_pos', [])
        follow2_pos = action_pred.get('follow2_pos', [])
        T = max(len(follow1_pos), len(follow2_pos))
        actions = []
        for t in range(T):
            left_action = follow1_pos[t] if t < len(follow1_pos) else [0.0] * 7
            # left_action = [0.0]*7
            right_action = follow2_pos[t] if t < len(follow2_pos) else [0.0] * 7
            
            if len(left_action) == 7 and len(right_action) == 7:
                # Combine: [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
                combined_action = np.concatenate([left_action, right_action], axis=0)
                actions.append(combined_action)
            else:
                # Fallback: pad with zeros if dimensions don't match
                raise ValueError(f"Left action and right action must have length 7, but got {len(left_action)} and {len(right_action)}")
        
        actions = np.array(actions)
        return actions

    def visualize_actions(self, actions, task_name="unknown", save_dir='visualize/actions', key='actions'):
        """
        Visualize actions in N*d format.
        Args:
            actions: numpy array of shape (N, d) where N is number of timesteps, d is action dimension
            task_name: task name for saving the visualization
            save_dir: directory to save the visualization
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        if actions.ndim != 2:
            print(f"Warning: actions should be 2D (N*d), but got shape {actions.shape}")
            return
        
        N, d = actions.shape
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Heatmap visualization
        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(actions.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Action Dimension', fontsize=12)
        ax1.set_title(f'Action Heatmap (Shape: {N}×{d})', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Action Value')
        
        # 2. Time series for each dimension (if d <= 14, show all; otherwise sample)
        ax2 = fig.add_subplot(gs[1, :])
        if d <= 14:
            # Show all dimensions
            for dim in range(d):
                ax2.plot(actions[:, dim], label=f'Dim {dim}', alpha=0.7, linewidth=1.5)
        else:
            # Sample dimensions to avoid clutter
            sample_dims = np.linspace(0, d-1, 14, dtype=int)
            for dim in sample_dims:
                ax2.plot(actions[:, dim], label=f'Dim {dim}', alpha=0.7, linewidth=1.5)
        
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Action Value', fontsize=12)
        ax2.set_title(f'Action Time Series ({"All" if d <= 14 else "Sampled"} Dimensions)', fontsize=14, fontweight='bold')
        ax2.legend(ncol=min(7, d), fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Action Visualization - {task_name}', fontsize=16, fontweight='bold', y=0.98)
        
        # Save visualization
        save_path = os.path.join(save_dir, task_name, key, f'{self.frame_idx}.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Action visualization saved to: {save_path} (Shape: {N}×{d})")


    def visualize_action_comparison(self, actions, replay_actions, task_name="unknown", save_dir='visualize', key='actions_vs_replay'):
        """
        Plot action prediction and replay action per dimension in a single figure.
        """
        os.makedirs(save_dir, exist_ok=True)
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(replay_actions, np.ndarray):
            replay_actions = np.array(replay_actions)

        if actions.ndim != 2 or replay_actions.ndim != 2:
            print(f"Warning: actions/replay_actions should be 2D (N*d), got {actions.shape} and {replay_actions.shape}")
            return

        # Align length and dimensions
        T = min(actions.shape[0], replay_actions.shape[0])
        D = min(actions.shape[1], replay_actions.shape[1])
        actions = actions[:T, :D]
        replay_actions = replay_actions[:T, :D]

        # Layout grid
        cols = 4
        rows = int(np.ceil(D / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=True)


        for dim in range(D):
            r, c = divmod(dim, cols)
            ax = axes[r, c]
            ax.plot(actions[:, dim], label='action_pred', linewidth=1.4)
            ax.plot(replay_actions[:, dim], label='replay_action', linewidth=1.2, linestyle='--')
            ax.set_title(f'Dim {dim}', fontsize=10)
            ax.grid(True, alpha=0.3)
            y_min = min(actions[:, dim].min(), replay_actions[:, dim].min())
            y_max = max(actions[:, dim].max(), replay_actions[:, dim].max())
            span = y_max - y_min
            if span < 0.1:
                pad = (0.1 - span) / 2.0
                y_min -= pad
                y_max += pad
            ax.set_ylim(y_min, y_max)

        # Hide unused subplots
        for idx in range(D, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis('off')

        # Global legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f'Action vs Replay Comparison - {task_name} (T={T}, D={D})', fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()

        save_path = os.path.join(save_dir, task_name, key, f'{self.frame_idx}.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Action comparison saved to: {save_path} (Shape: {T}×{D})")

    def visualize_input_data(self, input_data, task_name=None, save_path=f'visualize'):
        """
        
        Visualize all information in input_data on a single image.
        Args:
            input_data: Input data for inference.
            save_path: Path to save the visualization image.
        """
        os.makedirs(save_path, exist_ok=True)
        # Extract data
        images = input_data['images']
        pil_images = {}
        for key in images:
            if isinstance(images[key], bytes):
                image_stream = io.BytesIO(images[key])
                pil_images[key] = Image.open(image_stream)
            else:
                pil_images[key] = images[key]
            
        action = input_data['action']
        pending_actions = input_data['pending_actions']
        state = input_data['state']
        timestamp = input_data['timestamp']
        
        # Create figure with subplots
        num_images = len(pil_images)
        # Calculate grid: images on top, info below
        rows = 2 if num_images > 0 else 1
        cols = max(num_images, 2) if num_images > 0 else 1
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Display images
        image_keys = sorted(pil_images.keys())
        if num_images > 0:
            for idx, key in enumerate(image_keys):
                ax = fig.add_subplot(gs[0, idx])
                ax.imshow(pil_images[key])
                ax.set_title(f'Image: {key}', fontsize=12, fontweight='bold')
                ax.axis('off')
            
            # Fill remaining image slots if any
            for idx in range(len(image_keys), cols):
                ax = fig.add_subplot(gs[0, idx])
                ax.axis('off')
        
        # Check if we can plot action
        can_plot_action = False
        action_array = None
        if len(action) > 0 and isinstance(action, (list, np.ndarray)):
            try:
                action_array = np.array(action)
                can_plot_action = (action_array.ndim == 1 and len(action_array) > 0 and 
                                 np.issubdtype(action_array.dtype, np.number))
            except:
                pass
        
        # Create info panel - adjust layout based on whether we'll plot action
        info_row = 1 if num_images > 0 else 0
        if can_plot_action and cols > 1:
            # Split bottom row: text on left, plot on right
            info_ax = fig.add_subplot(gs[info_row, :cols//2])
            plot_ax = fig.add_subplot(gs[info_row, cols//2:])
        else:
            # Use full width for text
            info_ax = fig.add_subplot(gs[info_row, :])
            plot_ax = None
        info_ax.axis('off')
        
        # Prepare text information
        info_text = []
        info_text.append("=" * 60)
        info_text.append("INPUT DATA INFORMATION")
        info_text.append("=" * 60)
        info_text.append(f"\nState: {state}")
        info_text.append(f"Pending Actions: {pending_actions}")
        info_text.append(f"Timestamp: {timestamp:.4f}")
        info_text.append(f"\nAction Length: {len(action)}")
        
        # Display action data
        if len(action) > 0:
            info_text.append(f"\nAction Data:")
            if isinstance(action, (list, np.ndarray)):
                if action_array is None:
                    action_array = np.array(action)
                if action_array.ndim == 1:
                    info_text.append(f"  Shape: {action_array.shape}")
                    info_text.append(f"  Type: {type(action[0]).__name__}")
                    # Show first and last few values if too long
                    if len(action) <= 20:
                        info_text.append(f"  Values: {action}")
                    else:
                        info_text.append(f"  First 10: {action[:10]}")
                        info_text.append(f"  Last 10: {action[-10:]}")
                else:
                    info_text.append(f"  Shape: {action_array.shape}")
                    info_text.append(f"  Type: {type(action_array.flat[0]).__name__}")
                    if action_array.size <= 50:
                        info_text.append(f"  Values:\n{action_array}")
                    else:
                        info_text.append(f"  First few values:\n{action_array.flat[:20]}")
            else:
                info_text.append(f"  Type: {type(action).__name__}")
                info_text.append(f"  Value: {action}")
        else:
            info_text.append("\nAction: Empty")
        
        # Display image information
        info_text.append(f"\n\nImages ({num_images}):")
        for key in image_keys:
            img = pil_images[key]
            info_text.append(f"  {key}: {img.size[0]}x{img.size[1]}, Mode: {img.mode}")
        
        # Display all text
        info_str = "\n".join(info_text)
        info_ax.text(0.05, 0.95, info_str, transform=info_ax.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add action visualization if it's numeric
        if can_plot_action and plot_ax is not None:
            plot_ax.plot(action_array, 'b-', linewidth=2, marker='o', markersize=4)
            plot_ax.set_title('Action Values', fontsize=12, fontweight='bold')
            plot_ax.set_xlabel('Index')
            plot_ax.set_ylabel('Value')
            plot_ax.grid(True, alpha=0.3)

        save_path = os.path.join(save_path, task_name, f'input_img.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.suptitle('Input Data Visualization', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {save_path}")



class GPUClient:
    def __init__(self, policy):
        self.policy = policy
    def infer(self, state):
        result = self.policy.run_policy(state)
        # print(f"Sending results: {result}")
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_token', type=str, required=True, help='User token')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--task_name', type=str, default="arrange_flowers", help='Task name')
    parser.add_argument('--config_path', type=str, default=None, help='Task name')
    parser.add_argument('--robot_id', type=str, default="UR5", help='Robot ID')
    parser.add_argument('--end_ratio', type=float, default="0.5", help='Robot ID')
    parser.add_argument('--action_predict_mode', type=str, default="diffusion", help='Action predict mode')
    parser.add_argument('--data_version', type=str, default="v1", choices=["v1", "v2", "v2_euler"], help='Data version')
    parser.add_argument('--action_space', type=str, default="pos", choices=["pos", "joint"], help='Action space')
    # you can modify or add your own parameters

    args = parser.parse_args()
    args.robot_id = TASK_INFO[args.task_name]['robot_id'].split('_')[0]
    # dataset_name = TASK_ROBOT_MAPPING[args.task_name]
    if args.data_version == "v1":
        dataset_name = f'robochallenge_{args.robot_id}'
    elif args.data_version == "v2":
        dataset_name = f'robochallenge_v2_{args.robot_id}'
    elif args.data_version == "v2_euler":
        dataset_name = f'robochallenge_v2_euler_{args.robot_id}'
    else:
        raise ValueError(f"Invalid data version: {args.data_version}")    
    # dataset_name = f'robochallenge_v2_euler_{args.robot_id}'


    # these args are generally not changed during evaluation, so we put them here.
    image_size = [640, 480]  # [640, 480]
    if args.robot_id.lower() == 'ur5':
        image_size = [1280, 720] # [1280, 720]
    # image_size = [224, 224] # this refers to README.md#get-state request parameter `width` and `height`
    if args.robot_id.lower() == 'ur5':
        image_type = ["left_hand", "right_hand"]
    else:
        image_type = ["high", "left_hand", "right_hand"] # this refers to README.md#get-state request parameter `image_type`
    # action_type = "joint"
    if args.action_space == "pos":
        if args.robot_id.lower() in ['arx5', 'ur5', 'franka']:
            action_type = 'leftpos'
        else:
            action_type = "pos"
    elif args.action_space == "joint":
        if args.robot_id.lower() in ['arx5', 'ur5', 'franka']:
            action_type = 'leftjoint'
        else:
            action_type = "joint"        
    else:
        raise ValueError(f"Invalid action space: {args.action_space}")
        # action_type = "joint" # this refers to both README.md#get-state and README.md#post-action parameters `action_type`
    duration = 0.05 # this refers to README.md#post-action request parameter `duration`

    client = InterfaceClient(args.user_token)
    policy = WallxPolicy(args.checkpoint, dataset_name, config_path=args.config_path, task_name=args.task_name, end_ratio=args.end_ratio, duration=duration, action_predict_mode=args.action_predict_mode, data_version=args.data_version, action_space=args.action_space)  # add your own parameters
    gpu_client = GPUClient(policy)  # add your own parameters

    if 'infer' in dir(policy):
        policy.duration = duration

    job_loop(client, gpu_client, args.run_id, image_size, image_type, action_type, duration)

if __name__ == '__main__':
    main()
