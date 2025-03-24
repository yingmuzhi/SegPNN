import os

def maybe_mkdir(path):
    # 使用 os.makedirs 函数创建目录，exist_ok=True 表示如果目录已经存在，不会抛出异常
    os.makedirs(path, exist_ok=True)