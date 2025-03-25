'''
author: ymz
data: 20250325
intro:
    1. turn gray into rgb.
    2. separate whole image into pieces.
    3. save as .tif formate.
'''
import tifffile, os
from PIL import Image
import numpy as np
import utils.dir_utils as dir_utils


### preprocess
# 将灰度图像转换为 RGB 格式
def grayscale_to_rgb(gray_image, method=1):
    '''
    intro: 
        turn gray into rgb. 
        two method:
        0.      [[gray]
                [0]
                [0]]

        1.      [[gray]
                [gray]
                [gray]]
        
        2.      pillow method

    '''
    # 获取灰度图像的高度和宽度
    height, width = gray_image.shape

    # 创建一个空的 RGB 图像，形状为 (height, width, 3)
    rgb_image = np.zeros((height, width, 3), dtype=np.uint16)

    if method == 0:
        rgb_image[:, :, 0] = gray_image  # 红色通道
        rgb_image[:, :, 1] = 0  # 绿色通道
        rgb_image[:, :, 2] = 0  # 蓝色通道
    elif method == 1:
        # 将灰度值复制到 RGB 的三个通道
        rgb_image[:, :, 0] = gray_image  # 红色通道
        rgb_image[:, :, 1] = gray_image  # 绿色通道
        rgb_image[:, :, 2] = gray_image  # 蓝色通道
    elif method == 2:
        # 将灰度图像转换为 PIL 图像对象
        pil_image = Image.fromarray(tiff_image)

        # 将灰度图像转换为 RGB 格式（三通道）
        rgb_image = pil_image.convert('RGB')
        
        # 如果图像是 Pillow 对象，转换为 ndarray
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)

    return rgb_image

def min_max_normalize_to_range(data, range=255):
    """
    将数据归一化到 [0, range] 范围
    :param data: 输入的 ndarray
    :return: 归一化后的 ndarray，数据类型为 uint8
    """
    # 计算数据的最小值和最大值
    data_min = np.min(data)
    data_max = np.max(data)

    # 避免除零错误（如果数据的最小值和最大值相等）
    if data_max == data_min:
        return np.zeros_like(data, dtype=np.uint8)

    # 归一化到 [0, range]
    normalized_data = (data - data_min) / (data_max - data_min) * range

    # 将结果转换为 uint8 类型
    normalized_data = normalized_data.astype(np.uint8)

    return normalized_data

### 分批量保存
def save_block(image_array, save_directory, block_size=(1000, 1000)):
    # 创建目录
    dir_utils.maybe_mkdir(save_directory)

    # 定义每个小块的大小
    block_height = block_size[0]  # 每个小块的高度
    block_width = block_size[1]   # 每个小块的宽度

    # 获取图像的总尺寸
    total_height, total_width, _ = image_array.shape

    # 遍历图像并拆分
    for i in range(0, total_height, block_height):
        for j in range(0, total_width, block_width):
            # 计算当前块的边界, 已经考虑边界值
            row_start, row_end = i, min(i + block_height, total_height)
            col_start, col_end = j, min(j + block_width, total_width)

            # 提取当前小块
            block = image_array[row_start:row_end, col_start:col_end, :]

            # 将小块保存为 tif 文件
            block_image = Image.fromarray(block) if type(block) == Image.Image else block
            save_block_path = os.path.join(save_directory, f'block_r{row_start}-{row_end}_c{col_start}-{col_end}.tif') 
            tifffile.imsave(save_block_path, block_image)

            print(f'Saved block(3 channel tif): block_r{row_start}-{row_end}_c{col_start}-{col_end}.tif')



if __name__ == '__main__':
    # region 20250323 
    # read
    tif_path = "/home/yingmuzhi/SegPNN/src/20250325/origin/JS WFA+PV 60X.tif"
    tif_image = tifffile.imread(tif_path)
    print(tif_image.shape, tif_image.dtype, tif_image.max())

    # turn (c, w, h) into (w, h, c)
    tif_image = tif_image.transpose(1, 2, 0)

    # separate pic into blocks
    save_block_path = "/home/yingmuzhi/SegPNN/src/20250325/preprocess/block"
    save_block(tif_image, save_block_path, (500, 500))   

    # process multi-pics
    save_output_path = "/home/yingmuzhi/SegPNN/src/20250325/preprocess/output"
    # generate dir
    dir_utils.maybe_mkdir(save_output_path)
    tif_list = [os.path.join(save_block_path, i) for i in os.listdir(save_block_path)]
    for tif_item in tif_list:
        # process one-pic
        tif_item_image = tifffile.imread(tif_item)
        _, _, channel_sum = tif_item_image.shape
        for channel_num in range(channel_sum):
            tif_image = tif_item_image[:, :, channel_num]
            tif_image = grayscale_to_rgb(tif_image, method=1)
            tif_image = min_max_normalize_to_range(tif_image)
            print(tif_image.shape, tif_image.dtype, tif_image.max())
            # save tif_image
            path_part = tif_item.split('/')[-1][6:] # 只拿出来'r3000-4000_c0-1000.tif'
            save_path = os.path.join(save_output_path, "ch{}_{}".format(channel_num, path_part))
            tifffile.imwrite(save_path, tif_image)

    # endregion
    