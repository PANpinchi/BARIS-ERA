# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
import cv2

import mmcv
import torch
import torch.distributed as dist
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def print_layer_names(model):
    print("Layer names in the model:")
    for name, module in model.named_modules():
        print(name)


def save_tsne_results(filename, d_x):
    np.save(filename, d_x)


def load_tsne_results(filename):
    return np.load(filename)


def register_hooks(model, layer_names):
    feature_maps = {name: None for name in layer_names}

    def hook_fn(module, input, output, name):
        feature_maps[name] = output

    hooks = []
    for layer_name in layer_names:
        hook = model.module.backbone.get_submodule(layer_name).register_forward_hook(
            lambda module, input, output, name=layer_name: hook_fn(module, input, output, name))
        hooks.append(hook)

    return hooks, feature_maps


def save_results(pred_features, imgs, name):
    if not os.path.isdir('./plots_ema'):
        os.mkdir('./plots_ema')

    num_environment_embeds = 17
    ncols = 9
    nrows = num_environment_embeds // ncols if num_environment_embeds % ncols == 0 else num_environment_embeds // ncols + 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 5))
    feature_size = (imgs[0].shape[1], imgs[0].shape[0])
    for i in range(num_environment_embeds):
        ax[i // ncols][i % ncols].imshow(imgs[0][:, :, ::-1])

        if i == 0:
            ax[i // ncols][i % ncols].set_title('Original Image')
        else:
            single_channel = pred_features[0, :, :, i - 1]
            mask = cv2.resize(single_channel, feature_size)
            ax[i // ncols][i % ncols].set_title(i)
            ax[i // ncols][i % ncols].imshow(mask, alpha=0.7, cmap='jet')
        ax[i // ncols][i % ncols].axis('off')

    for i in range(num_environment_embeds, nrows * ncols):
        ax[i // ncols][i % ncols].axis('off')
    plt.savefig("./plots_ema/vis_ema_{}.png".format(name))
    plt.close()


def single_gpu_vis_ema(model,
                       data_loader,
                       show=False,
                       out_dir=None,
                       show_score_thr=0.3,
                       is_draw_bbox=False,
                       is_draw_labels=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    # print_layer_names(model)
    layer_names = ['stages.0.blocks.0.ema.softmax']
    hooks, feature_maps = register_hooks(model, layer_names)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # XL_118.jpg, XL_63.jpg
        file_path = data['img_metas'][0].data[0][0]['filename']
        file_name_with_ext = os.path.basename(file_path)
        file_name, _ = os.path.splitext(file_name_with_ext)
        if os.path.isfile("./plots_ema/vis_ema_{}.png".format(file_name)):
            print("{}: File {} already exists. Skipping save.".format(i, file_name))
            continue
        # if i < 0 or data['img_metas'][0].data[0][0]['filename'] != 'data/UDW/val/XL_118.jpg':
        #     print('skip {}: {}'.format(i, data['img_metas'][0].data[0][0]['filename']))
        #     continue

        name = layer_names[0]
        print(feature_maps[name].shape)
        print(data['img'][0].shape)
        if feature_maps[name].shape[1] == 625:
            h, w = 25, 25
        elif feature_maps[name].shape[1] == 850:
            h, w = 25, 34
        elif feature_maps[name].shape[1] == 625 * 4:
            h, w = 25 * 2, 25 * 2
        elif feature_maps[name].shape[1] == 850 * 4:
            h, w = 25 * 2, 34 * 2
        elif feature_maps[name].shape[1] == 625 * 16:
            h, w = 25 * 4, 25 * 4
        elif feature_maps[name].shape[1] == 850 * 16:
            h, w = 25 * 4, 34 * 4
        elif feature_maps[name].shape[1] == 625 * 64:
            h, w = 25 * 8, 25 * 8
        elif feature_maps[name].shape[1] == 850 * 64:
            h, w = 25 * 8, 34 * 8
        # special case for XXL image
        else:
            h, w = data['img'][0].shape[-2] // 4, data['img'][0].shape[-1] // 4
        features = feature_maps[name].view(feature_maps[name].size(0), h, w, -1).detach().cpu().numpy()

        print('===== img_metas =====')
        print(data['img_metas'][0].data[0][0]['filename'])
        print('===== img_metas =====')
        batch_size = len(result)
        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

        file_path = data['img_metas'][0].data[0][0]['filename']
        file_name_with_ext = os.path.basename(file_path)
        file_name, _ = os.path.splitext(file_name_with_ext)
        save_results(features, imgs, name=file_name)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr,
                    is_draw_bbox=is_draw_bbox,
                    is_draw_labels=is_draw_labels)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    for hook in hooks:
        hook.remove()

