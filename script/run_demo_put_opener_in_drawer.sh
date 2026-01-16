
#!/bin/sh

# 将路径添加到 PYTHONPATH 的最前面
export PYTHONPATH="/path/to/wall-x:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=3

python demo.py \
    --checkpoint /path/to/ckpt \
    --user_token your_token \
    --run_id your_run_id \
    --task_name put_opener_in_drawer \
    --data_version v2_euler \
    --action_predict_mode diffusion \
    --action_space joint \
    --end_ratio 0.8