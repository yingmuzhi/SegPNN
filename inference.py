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






if __name__=='__main__':

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
    # import pickle
    # save_path = "/home/yingmuzhi/SegPNN/src/20250323/2_mask.pkl"
    # try:
    #     with open(save_path, 'wb') as file:
    #         pickle.dump(masks2, file)
    #     print("对象已成功保存到 mask2.pkl 文件中。")
    # except Exception as e:
    #     print(f"保存对象时出现错误: {e}")
    # pass
    # #endregion

    # read mask
    import pickle
    save_path = "/home/yingmuzhi/SegPNN/src/20250323/1_mask.pkl"
    try:
        with open(save_path, 'rb') as file:
            masks2 = pickle.load(file)
        print("对象已成功从 masks2.pkl 文件中读取。")
        # 你可以在这里对读取的对象进行后续操作，例如打印对象信息
        print(masks2)
    except FileNotFoundError:
        print("错误：未找到 mask2.pkl 文件。")
    except Exception as e:
        print(f"读取对象时出现错误: {e}")
    
    save_path = "/home/yingmuzhi/SegPNN/src/20250323/2_mask.pkl"
    try:
        with open(save_path, 'rb') as file:
            img2_masks2 = pickle.load(file)
        print("对象已成功从 masks2.pkl 文件中读取。")
        # 你可以在这里对读取的对象进行后续操作，例如打印对象信息
        print(img2_masks2)
    except FileNotFoundError:
        print("错误：未找到 mask2.pkl 文件。")
    except Exception as e:
        print(f"读取对象时出现错误: {e}")

    # calculate the number of wholes
    # img2_path = "/home/yingmuzhi/SegPNN/src/20250323/2.tif"
    # image2 = tifffile.imread(img_path)
    # sam3 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    # mask_generator_3 = SAM2AutomaticMaskGenerator(
    #     model=sam3,
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
    # img2_masks2 = mask_generator_3.generate(image2)
    hole1 = caculate_masks(masks2)
    hole2 = caculate_masks(img2_masks2)
    print("WFA has {} holes, and NeuNs has {} holes.".format(hole1, hole2))

    # calculate overlap
    overlaps = count_overlapped_masks(masks2, img2_masks2)
    print("WFA and NeuNs has {} overlaps".format(overlaps))
    pass