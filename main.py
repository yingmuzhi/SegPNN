# ================ import env
import os,sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))    # add path
import tifffile


# ================ add device for calculate :: CUDA
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")   # actually
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


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


# ================ test image path
# test_iamge_path = "/home/yingmuzhi/SegPNN/src/20250108/8.png"
# image = Image.open(test_image_path)
# image = np.array(image.convert("RGB"))
# print(image.shape)

# 1. 使用 tifffile 读取 TIFF 文件
def open_image(file_path):
    # 获取文件扩展名
    ext = os.path.splitext(file_path)[1].lower()

    # 根据扩展名选择打开方式
    if ext in ['.tif', '.tiff']:
        # 使用 tifffile 打开 TIFF 文件
        image = tifffile.imread(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        # 使用 Pillow 打开其他格式的图像
        image = Image.open(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # 如果图像是 Pillow 对象，转换为 ndarray
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    return image

test_image_path = "/home/yingmuzhi/SegPNN/src/20250123/block_2000_3000.tif"
image = open_image(test_image_path)
print("读取后图像的原始形状:", image.shape)

# 2. 根据需要，转换成 RGB。以下做法仅在你确实希望得到 (H, W, 3) 的 RGB 时使用
#    - 如果原图是单通道 (H, W)，就把它复制 3 份到最后一维，模拟一个“灰度转RGB”的效果。
#    - 如果原图本身已有多个通道 (例如 3 或 4 通道)，可酌情处理。
if image.ndim == 2:
    # 单通道灰度 -> 3 通道 RGB
    image_rgb = np.stack([image, image, image], axis=-1)
    print("转换后图像形状:", image_rgb.shape)
elif image.ndim == 3:
    # 如果是 (H, W, 4) 可能是 RGBA，你可以根据需要只取前 3 个通道
    if image.shape[-1] == 4:
        image_rgb = image[..., :3]
        print("去掉Alpha通道后的形状:", image_rgb.shape)
    else:
        # 如果本身就是 (H, W, 3)，那么已经是 RGB 了
        image_rgb = image
        print("已是 3 通道形状:", image_rgb.shape)
else:
    # 其他情况根据你的实际场景来处理
    image_rgb = image
    print("图像形状未做转换:", image_rgb.shape)

# 到这里，image_rgb 就是你可以在后续继续处理或可视化的数组
# 例如:
print("最终处理后图像形状:", image_rgb.shape)




# ================ plot test image
save_test_image_path = "/home/yingmuzhi/SegPNN/src/20250121/test_image.png"
plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')

# 保存图片到文件，例如保存为 "test_image.png"
plt.savefig(save_test_image_path, bbox_inches='tight', pad_inches=0)
plt.close()  # 关闭当前的绘图，释放资源


# ================ set class SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = os.path.join(os.path.dirname(__file__), "sam2", "checkpoints/sam2.1_hiera_large.pt")   # "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)


# ================ generate mask automatically using class
###### turn unit16 -> unint8
image = (image / 65535 * 255 ).astype(np.uint8)
masks = mask_generator.generate(image)

print(image.shape,)
#print(len(masks), '\n', masks)
print(masks[0])
# print(masks[1])


# ================ plot mask
mask_save_path = "/home/yingmuzhi/SegPNN/src/20250123/test_mask.png"
plt.close()
plt.figure(figsize=(20, 20))
# plt.imshow(image)
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
    print("overlap mask's num is {}".format(overlap_num))
    return overlap_num

### plot mask
show_anns(masks, plot_white_mask=False, image=image)
plt.axis('off')
# 保存图片到文件
plt.savefig(mask_save_path, bbox_inches='tight', pad_inches=0)
plt.close()  # 关闭当前的绘图，释放资源

### cal hole num
caculate_masks(masks)

### cal overlap num
image2 = image
masks2 = mask_generator.generate(image)
count_overlapped_masks(masks, masks2)





pass