# ================ import env
import os,sys; sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"));    # add path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils.inference_utils import *
import tifffile
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import pickle
import utils.dir_utils as dir_utils
import gc
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



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

def process_one_mask(img_path, save_inference_path):
    '''
    intro:
        generate one mask.
    '''
    # read pic
    image = tifffile.imread(img_path)

    # make mask_generater instance `SAM2`
    sam2_checkpoint = "/home/yingmuzhi/SegPNN/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # adjust some hyper-parameters in model
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )

    # generate masks
    masks2 = mask_generator_2.generate(image)
    print("image shape is {}, \nmasks number is {}, \nthere are these object in masks: {}".format(image.shape, len(masks2), masks2[0].keys() if len(masks2) > 0 else 0))

    # save fig
    mask_png_all_save_path = os.path.join(save_inference_path, "mask_png")
    dir_utils.maybe_mkdir(mask_png_all_save_path)
    mask_png_save_path = os.path.join(mask_png_all_save_path, "mask_" + img_path.split('/')[-1][:-4] + ".png")
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.savefig(mask_png_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # save mask object
    mask_pkl_all_save_path = os.path.join(save_inference_path, "mask_pkl")
    dir_utils.maybe_mkdir(mask_pkl_all_save_path)
    mask_pkl_save_path = os.path.join(mask_pkl_all_save_path, "mask_" + img_path.split('/')[-1][:-4] + ".pkl")
    try:
        with open(mask_pkl_save_path, 'wb') as file:
            pickle.dump(masks2, file)
        print("对象已成功保存到 mask2.pkl 文件中。")
    except Exception as e:
        print(f"保存对象时出现错误: {e}")
    finally:
        if 'file' in locals():
            file.close()


    # 释放模型占用的空间
    # logits_input = np.zeros((3, 3, 3), dtype=np.uint8)
    # logits_output = mask_generator_2.generate(logits_input)
    # del logits_input, logits_output
    # del sam2, mask_generator, mask_generator_2, image, masks2
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()

    pass


import re
def extract_key(filename):
    match = re.search(r'ch(\d+)_r(\d+)-(\d+)_c(\d+)-(\d+)', filename)
    if match:
        return match.group(0)
    return None
def process_multi_masks(save_output_path, save_inference_path):
    '''
    intro:
        generate multi masks
    '''
    # add not generate same pic
    exit_pic = os.listdir(os.path.join(save_inference_path, "mask_png"))
    all_pic = os.listdir(save_output_path)
    keys_in_list1 = {extract_key(item) for item in exit_pic if extract_key(item)}
    result = [item for item in all_pic if extract_key(item) not in keys_in_list1]
    if len(result) != len(all_pic) - len(exit_pic):
        raise EOFError

    tif_list = [os.path.join(save_output_path, i) for i in result]
    for item in tif_list:
        process_one_mask(item, save_inference_path)


if __name__=='__main__':

    save_output_path = "/home/yingmuzhi/SegPNN/src/20250324/preprocess/output"
    save_inference_path = "/home/yingmuzhi/SegPNN/src/20250324/inference"
    process_multi_masks(save_output_path, save_inference_path)

    # # region session1  
    # # read pic
    # img_path = "/home/yingmuzhi/SegPNN/src/20250323/2.tif"
    # image = tifffile.imread(img_path)

    # # make mask_generater instance `SAM2`
    # sam2_checkpoint = "/home/yingmuzhi/SegPNN/sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    # mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # # generate masks
    # masks = mask_generator.generate(image)
    # print(image.shape,)
    # print(len(masks),)
    # print(masks[0].keys()) 

    # # make mask_generater instance `SAM2` -- Change some params
    # mask_generator_2 = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=64,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.92,
    #     stability_score_offset=0.7,
    #     crop_n_layers=1,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=25.0,
    #     use_m2m=True,
    # )

    # # generate masks
    # masks2 = mask_generator_2.generate(image)
    # print(len(masks2)) 

    # # save fig
    # fig_save_path = "/home/yingmuzhi/SegPNN/src/20250323/2_output_pro.png"
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns(masks2)
    # plt.axis('off')
    # plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # # save mask
    # save_path = "/home/yingmuzhi/SegPNN/src/20250323/2_mask.pkl"
    # try:
    #     with open(save_path, 'wb') as file:
    #         pickle.dump(masks2, file)
    #     print("对象已成功保存到 mask2.pkl 文件中。")
    # except Exception as e:
    #     print(f"保存对象时出现错误: {e}")
    # pass
    # #endregion

    # # region session2
    # # read mask
    # import pickle
    # save_path = "/home/yingmuzhi/SegPNN/src/20250323/1_mask.pkl"
    # try:
    #     with open(save_path, 'rb') as file:
    #         masks2 = pickle.load(file)
    #     print("对象已成功从 masks2.pkl 文件中读取。")
    #     # 你可以在这里对读取的对象进行后续操作，例如打印对象信息
    #     print(masks2)
    # except FileNotFoundError:
    #     print("错误：未找到 mask2.pkl 文件。")
    # except Exception as e:
    #     print(f"读取对象时出现错误: {e}")
    
    # save_path = "/home/yingmuzhi/SegPNN/src/20250323/2_mask.pkl"
    # try:
    #     with open(save_path, 'rb') as file:
    #         img2_masks2 = pickle.load(file)
    #     print("对象已成功从 masks2.pkl 文件中读取。")
    #     # 你可以在这里对读取的对象进行后续操作，例如打印对象信息
    #     print(img2_masks2)
    # except FileNotFoundError:
    #     print("错误：未找到 mask2.pkl 文件。")
    # except Exception as e:
    #     print(f"读取对象时出现错误: {e}")

    # # calculate the number of wholes
    # # img2_path = "/home/yingmuzhi/SegPNN/src/20250323/2.tif"
    # # image2 = tifffile.imread(img_path)
    # # sam3 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    # # mask_generator_3 = SAM2AutomaticMaskGenerator(
    # #     model=sam3,
    # #     points_per_side=64,
    # #     points_per_batch=128,
    # #     pred_iou_thresh=0.7,
    # #     stability_score_thresh=0.92,
    # #     stability_score_offset=0.7,
    # #     crop_n_layers=1,
    # #     box_nms_thresh=0.7,
    # #     crop_n_points_downscale_factor=2,
    # #     min_mask_region_area=25.0,
    # #     use_m2m=True,
    # # )
    # # img2_masks2 = mask_generator_3.generate(image2)
    # hole1 = caculate_masks(masks2)
    # hole2 = caculate_masks(img2_masks2)
    # print("WFA has {} holes, and NeuNs has {} holes.".format(hole1, hole2))

    # # calculate overlap
    # overlaps = count_overlapped_masks(masks2, img2_masks2)
    # print("WFA and NeuNs has {} overlaps".format(overlaps))
    # pass

    # #endregion
