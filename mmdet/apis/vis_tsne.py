# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

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


def single_gpu_vis_tsne(model,
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
    # layer_names = ['stages.3.blocks.1.ema', 'stages.3.blocks.1.ffn']
    # layer_names = ['stages.2.blocks.17.ema']
    layer_names = ['stages.2.blocks.17.ffn']
    hooks, feature_maps = register_hooks(model, layer_names)

    TSNE_results = {name: None for name in layer_names}
    target_sizes = {name: None for name in layer_names}
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        for name in layer_names:
            if target_sizes[name] is None:
                target_sizes[name] = feature_maps[name].shape[1]
            if feature_maps[name].shape[1] > target_sizes[name]:
                break  # feature_maps[name] = feature_maps[name][:, :target_sizes[name], :]
            flat_features = feature_maps[name].view(feature_maps[name].size(0), -1).detach().cpu().numpy()
            if TSNE_results[name] is None:
                TSNE_results[name] = flat_features
            else:
                TSNE_results[name] = np.vstack((TSNE_results[name], flat_features))

        if len(TSNE_results[name]) == 50:
            break

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

    if not os.path.isdir('./t-SNE_plots'):
        os.mkdir('./t-SNE_plots')

    tsne = TSNE(n_components=2)
    for i, name in enumerate(layer_names):
        TSNE_results[name] = TSNE_results[name].reshape(TSNE_results[name].shape[0], -1)
        d_x = tsne.fit_transform(TSNE_results[name])
        save_tsne_results('./t-SNE_plots/swin.npy', TSNE_results[name])
        save_tsne_results('./t-SNE_plots/swin_d_x.npy', d_x)

    for hook in hooks:
        hook.remove()


def save_tSNE_results():
    print('Calculate t-SNE results')
    tsne = TSNE(n_components=2)

    out_dir = './t-SNE_plots'
    layer_names = ['ours', 'swin', 'ori']
    d_x_names = ['ours_d_x', 'swin_d_x', 'ori_d_x']
    model_names = ['Swin w/ EMR (Ours)', 'Swin w/o EMR', 'Swin (Pretrained)']
    colors = ['b', 'g', 'r']  # Red for first layer, Blue for second layer

    TSNE_results = {}
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(layer_names):
        name = osp.join(out_dir, name)
        TSNE_results[name] = load_tsne_results('{}.npy'.format(name))
        d_x = tsne.fit_transform(TSNE_results[name])
        plt.scatter(d_x[:, 0], d_x[:, 1], c=colors[i], label=model_names[i], alpha=0.5)

    plt.title("t-SNE results")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig("./t-SNE_plots/t-SNE_test.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(d_x_names):
        name = osp.join(out_dir, name)
        d_x = load_tsne_results('{}.npy'.format(name))
        save_tsne_results('{}_new.npy'.format(name), d_x)
        plt.scatter(d_x[:, 0], d_x[:, 1], c=colors[i], label=model_names[i], alpha=0.5)

    plt.title("t-SNE d_x results")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig("./t-SNE_plots/t-SNE_d_x_test.png")
    plt.close()

