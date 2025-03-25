#!/bin/bash

# 激活 Conda 环境，这里假设环境名为 your_conda_env
/root/anaconda3/bin/activate env_cp311_SegPNN

# 定义循环条件
while true
do
    # 执行你的 Python 脚本，这里假设脚本名为 your_script.py
    python3 /home/yingmuzhi/SegPNN/inference.py
    # 等待 60 秒，你可以根据需求修改这个时间
    sleep 60
done
