import argparse
import logging
import time
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from task_info import TASK_INFO

from demo import GPUClient, WallxPolicy
from robot.interface_client import InterfaceClient

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

DEFAULT_USER_ID = "test_user"
DEFAULT_JOBS = ["test_job"]
DEFAULT_ROBOT_ID = "test_robot"

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



def plot_openloop(action_pred_list, action_gt_list, save_path):
    assert len(action_pred_list) == len(
        action_gt_list
    ), "Predicted action and ground truth action must have the same shape."

    dim = len(action_pred_list[0])
    plt.figure(figsize=(12, 4 * dim))

    for i in range(dim):
        plt.subplot(dim, 1, i + 1)

        # plot every 10th action
        plt.xticks(np.arange(0, sum(len(gt) for gt in action_gt_list), step=10))
        # for j in range(len(action_gt_list)):
        gt_action = np.array(action_gt_list)
        predict_action = np.array(action_pred_list)
        
        plt.plot(gt_action[:, i], label="Ground Truth", color="blue")
        plt.plot(predict_action[:, i], label="Model Output", color="orange")
 
        plt.title(f"Action Dimension {i + 1}")
        plt.xlabel("Time Step")
        plt.ylabel("Action Value")
        plt.legend()


    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}.jpg", dpi=200)
    print(f"Saved plot to {save_path}.jpg")


# === 核心可视化逻辑 ===
def visualize_and_save(pred, gt, task_name, save_dir='debug_viz'):
    os.makedirs(save_dir, exist_ok=True)
    # Save NPY
    np.save(os.path.join(save_dir, f'{task_name}_pred.npy'), pred)
    np.save(os.path.join(save_dir, f'{task_name}_gt.npy'), gt)
    
    if pred.ndim != 2 or gt.ndim != 2: return
    
    # 对齐长度
    L = min(len(pred), len(gt))
    if L == 0: return
    pred_vis, gt_vis = pred[:L], gt[:L]
    
    D = pred_vis.shape[1]
    D_gt = gt_vis.shape[1]
    
    # 计算 MSE
    mse_val = -1
    if D == D_gt:
        mse_val = np.mean((pred_vis - gt_vis)**2)
        title = f"{task_name} | Step: {L} | MSE: {mse_val:.5f}"
    else:
        title = f"{task_name} | Step: {L} | Dim Mismatch ({D} vs {D_gt})"

    # Plot
    cols = int(math.ceil(math.sqrt(D)))
    rows = int(math.ceil(D/cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2*rows), constrained_layout=True)
    axes = np.array(axes).flatten()
    
    for d in range(D):
        if d >= len(axes): break
        ax = axes[d]
        ax.plot(pred_vis[:, d], 'b-', label='Pred', alpha=0.7)
        if d < D_gt:
            ax.plot(gt_vis[:, d], 'r--', label='GT', alpha=0.7)
        ax.set_title(f'Dim {d}')
        if d == 0: ax.legend(fontsize='x-small')
        
    for d in range(D, len(axes)): axes[d].axis('off')
    
    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, f'{task_name}_viz.jpg'))
    plt.close()
    
    if L % 10 == 0:
        logging.info(f"Viz saved. MSE: {mse_val:.5f}")


class VizGPUClient:
    def __init__(self, internal_client, task_name, current_time_str):
        self.client = internal_client
        self.task_name = task_name
        self.save_dir = os.path.join("debug_viz", current_time_str)
        self.episode_action_history = []
        self.episode_gt_history = []
        
    def reset(self):
        self.episode_action_history = []
        self.episode_gt_history = []
        logging.info("Viz history reset.")

    def infer(self, state, robot_id):
        # 1. 调用原始的标准推理 (Removed unnecessary manual processing)
        result = self.client.infer(state, robot_id)
        
        # 2. 处理 Ground Truth (保留数据处理逻辑以确保可视化正确)
        try:
            # 原始 GT 提取逻辑
            input_data = state
            raw_gt = input_data.get('gt_action', input_data.get('action'))
            if raw_gt is None: raw_gt = []
            
            # 处理 list 嵌套和深拷贝
            gt_temp = list(raw_gt)
            for i, v in enumerate(gt_temp):
                if isinstance(v, list): gt_temp[i] = v[0]
            if isinstance(gt_temp, list) and len(gt_temp) > 0 and isinstance(gt_temp[0], list):
                gt_temp = gt_temp[0]
            gt_action = np.array(gt_temp)

            # import ipdb; ipdb.set_trace()

            # 针对单臂机器人进行 GT 切片 (对齐 Server 发送的双臂数据)
            if robot_id.lower() in ['arx5', 'ur5', 'franka']:
                if len(gt_action) >= 14:
                    gt_action = gt_action[7:] # 取后半部分 (右臂)
            
            if len(gt_action) > 0:
                self.episode_gt_history.append(gt_action)
                
            # 3. 处理 Prediction
            if len(result) > 0:
                current_pred = np.array(result[0]) # 取第一帧动作
                self.episode_action_history.append(current_pred)

            # 4. 执行可视化
            visualize_and_save(
                np.array(self.episode_action_history), 
                np.array(self.episode_gt_history), 
                self.task_name,
                self.save_dir
            )
        except Exception as e:
            logging.warning(f"Viz error (ignored): {e}")

        return result

def process_job(client, gpu_client, job_id, robot_id, image_size, image_type, action_type, save_path, duration, max_wait=600):
    action_pred_list=[]
    action_gt_list=[]
    count = 0
    # 如果是包装类，手动重置历史
    if hasattr(gpu_client, 'reset'):
        gpu_client.reset()
        
    try:
        start_time = time.time()
        logging.info(f"--- Processing Job: {job_id} on {robot_id} ---")
        while True:
            client.start_motion()
            state = client.get_state(image_size, image_type, action_type)
            if not state:
                time.sleep(0.5); continue
            if state['state'] == "size_none":
                client.post_size()
                time.sleep(0.5); continue
            if state['state'] != "normal" or state['pending_actions'] != 0:
                time.sleep(0.5); continue
        
            time.sleep(0.5)
            state = client.get_state(image_size, image_type, action_type)
            action_gt_list.append(state['action'])
            
            logging.info("get_robot_state time: %.2f", time.time() - state['timestamp'])
            # count=0
            cursor=0
            result = gpu_client.infer(state)
            logging.info(f"Inference result: {len(result)}")
            client.post_actions(result, duration, action_type)

            action_pred_list.append(result[0])
            count+=1

            if time.time() - start_time > max_wait:
                logging.warning(f"Job {job_id} exceeded max wait time.")
                break
            client.end_motion()

        if save_path:
            plot_openloop(action_pred_list[:-1], action_gt_list[1:], save_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Error processing job {job_id}: {e}")
    finally:
        client.end_motion()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--openloop_save_path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--task_name', type=str, default="arrange_flowers", help='Task name')
    parser.add_argument('--config_path', type=str, default=None, help='Config path')
    parser.add_argument('--robot_id', type=str, default="UR5", help='Robot ID')
    parser.add_argument('--end_ratio', type=float, default="0.5", help='Robot ID')
    parser.add_argument('--action_predict_mode', type=str, default="diffusion", help='Action predict mode')
    parser.add_argument('--data_version', type=str, default="v1", choices=["v1", "v2", "v2_euler"], help='Data version')
    parser.add_argument('--replay_episode_id', type=int, default=0, help='Episode ID for Replay')
    parser.add_argument('--action_space', type=str, default="pos", choices=["pos", "joint"], help='Action space')
    
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
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # image_size = [320, 240]
    image_size = [640, 480]
    if args.robot_id.lower() == 'ur5':
        # image_size = [320, 180]
        image_size = [1280, 720]
        
    if args.robot_id.lower() == 'ur5':
        image_type = ["left_hand", "right_hand"]
    else:
        image_type = ["high", "left_hand", "right_hand"]
        
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
    duration = 0.05

    client = InterfaceClient(DEFAULT_USER_ID, mock=True)
    client.update_job_info(DEFAULT_JOBS[0], args.robot_id.lower())


    save_path=args.openloop_save_path

    
    policy = WallxPolicy(
        args.checkpoint, dataset_name=dataset_name, config_path=args.config_path, task_name=args.task_name, 
        end_ratio=args.end_ratio,
        duration=duration, action_predict_mode=args.action_predict_mode, 
        data_version=args.data_version, visualize_path='visualize_replay', 
        replay_episode_id=args.replay_episode_id,
        action_space=args.action_space
    )
    gpu_client = GPUClient(policy)

    if 'infer' in dir(policy):
        duration = duration / policy.infer.args.interpolate_multiplier
        policy.duration = duration

    jobs = DEFAULT_JOBS
    while jobs:
        for job_id in jobs[:]:
            try:
                process_job(
                    client, gpu_client, job_id, args.robot_id.lower(),
                    image_size, image_type, action_type, save_path, duration
                )
                jobs.remove(job_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logging.error(f"Error processing job {job_id}: {e}")
                jobs.remove(job_id)
    
    logging.info("All jobs processed.")
    return True

if __name__ == "__main__":
    main()
