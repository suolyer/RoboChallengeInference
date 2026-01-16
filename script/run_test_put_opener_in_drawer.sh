
#!/bin/sh

# 将路径添加到 PYTHONPATH 的最前面
export PYTHONPATH="/path/to/wall-x:$PYTHONPATH"


python test.py --checkpoint /path/to/ckpt \
    --task_name put_opener_in_drawer \
    --data_version v2_euler \
    --action_predict_mode diffusion \
    --action_space joint \
    --end_ratio 0.8 \
    --openloop_save_path /path/to/save/visualize/