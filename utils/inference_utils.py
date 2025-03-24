import numpy as np
import matplotlib.pyplot as plt

# ================ add script for plotting
np.random.seed(3)

def caculate_masks(anns):
    """
    intro:
        calculate mask number
    """
    sum = len(anns)
    print("this mask has {} hole".format(sum))
    return sum

def show_anns(anns, borders=True, plot_white_mask=False, image=None):
    if image is not None:
        plt.imshow(image)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))    # 0 for black
    img[:, :, 3] = 0     # 1 for visiable
    for ann in sorted_anns:
        m = ann['segmentation']
        if plot_white_mask:
            color_mask = [1., 1., 1., 0.5]  
        else:
            color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def show_anns_and_save(anns, borders=True, output_path="output.png"):
    """
    根据 anns 绘制分割结果并保存为图像文件。

    :param anns: 包含分割区域信息的列表，每个元素应包含:
                 'segmentation' (bool mask) 和 'area' 用于排序
    :param borders: 是否在蒙版边缘绘制轮廓
    :param output_path: 输出图像文件路径
    """

    if len(anns) == 0:
        print("anns 为空，不进行绘制。")
        return

    # 按面积从大到小排序，确保大面积的区域先绘制
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 创建 figure 和 ax
    fig, ax = plt.subplots()
    ax.set_autoscale_on(False)

    # 以最大区域尺寸初始化 RGBA 图像 (默认白色 + alpha=0)
    height, width = sorted_anns[0]['segmentation'].shape
    img = np.ones((height, width, 4), dtype=np.float32)
    img[:, :, 3] = 0  # alpha 初始化为0

    # 依次绘制蒙版
    for ann in sorted_anns:
        m = ann['segmentation']  # bool 类型或 0/1 的掩码数组
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # RGBA, alpha=0.5
        img[m] = color_mask  # 将当前 mask 区域染成随机颜色

        # 如果需要边界轮廓
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            # 对轮廓进行平滑
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                        for contour in contours]
            # 在 img 上绘制轮廓
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    # 在画布上显示合成图像
    ax.imshow(img)
    # 可选：隐藏坐标刻度
    ax.axis('off')

    # 保存图像到指定文件
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭 figure，释放资源


def caculate_masks(anns):
    """
    intro:
        calculate mask number
    """
    sum = len(anns)
    print("this mask has {} hole".format(sum))
    return sum


def calculate_IoU(mask1, mask2):
    # 计算两个mask的交集
    intersection = np.logical_and(mask1, mask2)
    # 计算两个mask的并集
    union = np.logical_or(mask1, mask2)
    # 计算交集和并集的元素数量
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    # 避免除零错误，如果并集元素数量为0，则IoU设为0
    if union_sum == 0:
        iou = 0
    else:
        # 计算IoU
        iou = intersection_sum / union_sum
    return iou
def count_overlapped_masks(photo1, photo2, threshold=0.8):
    """
    intro:
        1. 循环遍历 photo1 的 hole
        2. 循环 photo2 的 hole
        3. 使用 IoU 程度判定 计算重合个数
        """
    overlap_num = 0
    for mask1 in photo1:
        hole1 = mask1["segmentation"]
        for mask2 in photo2:
            hole2 = mask2["segmentation"]
            IoU_score = calculate_IoU(hole1, hole2)
            if IoU_score > threshold or IoU_score == threshold:
                overlap_num += 1
                break
    print("overlap mask's num is {}".format(overlap_num))
    return overlap_num