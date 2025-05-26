from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
import os
import cv2
import mmcv
from tqdm import tqdm


def vis_infer(checkpoints, config, data_dir, output_dir):
    """
    Function to run the DetInferencer for visual inference with default parameters.
    Args:
    checkpoints (str): Path to the model checkpoint.
    config (str): Path to the configuration file.
    data_dir (str): Path to the directory containing the test data.
    output_dir (str): Path to the output directory where results will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = init_detector(config, checkpoints, device='cuda:0')

    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100)]

    for idx, img_name in enumerate(tqdm(image_files)):
        img_path = os.path.join(data_dir, img_name)

        result = inference_detector(model, img_path)

        out_file = os.path.join(output_dir, img_name)
        show_result_pyplot(model,
                           img_path,
                           result,
                           score_thr=0.8,
                           out_file=out_file,
                           palette=PALETTE,
                           is_draw_bbox=False,
                           is_draw_labels=False)


if __name__ == "__main__":
    # BARIS-ERA Config (Swin Backbone)
    config = {
        "checkpoints": "./pretrained/baris-era_swin_base.pth",
        "config": "./configs/_ours_/ablation/mask_rcnn_swin-b-p4-w7_fpn_1x_coco_pr2.py",
        "data_dir": "./data/",
        "output_dir": "./outputs/",
    }

    # BARIS-ERA Config (ConvNeXt Backbone)
    # config = {
    #     "checkpoints": "./pretrained/baris-era_convnext_base.pth",
    #     "config": "./configs/_ours_/mask_rcnn_convnext-b_p4_w7_fpn_1x_coco.py",
    #     "data_dir": "./data/",
    #     "output_dir": "./outputs/",
    # }

    vis_infer(
        checkpoints=config["checkpoints"],
        config=config["config"],
        data_dir=config["data_dir"],
        output_dir=config["output_dir"]
    )
