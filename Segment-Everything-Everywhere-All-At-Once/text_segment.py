#!/usr/bin/env python
# --------------------------------------------------------
# SEEM Text-to-Mask Segmentation Script
# 使用文本提示进行图像分割
# --------------------------------------------------------

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.arguments import load_opt_from_config_files
from utils.distributed import init_distributed
from modeling import build_model
from modeling.BaseModel import BaseModel
from utils.constants import COCO_PANOPTIC_CLASSES
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from modeling.language.loss import vl_similarity

# 图像预处理
t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other', '').replace('-merged', '') for name in COCO_PANOPTIC_CLASSES] + ["others"]


def load_model(config_path='configs/seem/focalt_unicl_lang_demo.yaml',
               checkpoint_path='seem_focalt_v0.pt'):
    """加载 SEEM 模型"""
    print('Loading model...')

    # 加载配置
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)

    # 构建模型
    model = BaseModel(opt, build_model(opt))

    # 加载权重
    model = model.from_pretrained(checkpoint_path)

    # 移到 GPU 并设置为评估模式
    model = model.eval().cuda()

    # 预计算文本嵌入
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ['background'], is_eval=True
        )

    print('Model loaded!')
    return model


def segment_with_text(model, image_path, text_prompt, output_path=None):
    """
    使用文本提示进行图像分割

    Args:
        model: SEEM 模型
        image_path: 输入图像路径
        text_prompt: 文本提示，如 "the dog", "red car", "person"
        output_path: 输出图像路径（可选）

    Returns:
        mask: 分割掩码 (numpy array)
        result_image: 结果图像 (PIL Image)
    """
    print(f'Processing: {image_path}')
    print(f'Text prompt: "{text_prompt}"')

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_ori = transform(image)

    width = image_ori.size[0]
    height = image_ori.size[1]
    image_np = np.asarray(image_ori)

    # 可视化器
    visual = Visualizer(image_np, metadata=metadata)

    # 转换为 tensor
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

    # 准备输入数据
    data = {
        "image": image_tensor,
        "height": height,
        "width": width,
        "text": [text_prompt]
    }

    # 设置任务
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = True
    model.model.task_switch['audio'] = False

    batch_inputs = [data]

    # 推理
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            results, image_size, extra = model.model.evaluate_demo(batch_inputs)

    # 处理结果 - 文本提示模式
    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id, :, :]
    pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    # 调整掩码大小
    pred_masks_pos = (F.interpolate(
        pred_masks_pos[None, ],
        image_size[-2:],
        mode='bilinear'
    )[0, :, :data['height'], :data['width']] > 0.0).float().cpu().numpy()

    # 获取类别名称
    texts = [all_classes[pred_class[0]]]
    print(f'Detected class: {texts[0]}')

    # 可视化
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors_list = [(np.array(color['color']) / 255).tolist()
                   for color in COCO_CATEGORIES] + [[1, 1, 1]]

    for idx, mask in enumerate(pred_masks_pos):
        out_txt = texts[idx] if 'Text' not in ['Text'] else text_prompt
        demo = visual.draw_binary_mask(
            mask,
            color=colors_list[pred_class[0] % len(colors_list)],
            text=out_txt
        )

    res = demo.get_image()
    result_image = Image.fromarray(res)

    # 保存结果
    if output_path:
        result_image.save(output_path)
        print(f'Result saved to: {output_path}')

    torch.cuda.empty_cache()

    return pred_masks_pos[0], result_image


def segment_panoptic(model, image_path, output_path=None):
    """
    全景分割（自动分割所有对象）

    Args:
        model: SEEM 模型
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）

    Returns:
        result_image: 结果图像 (PIL Image)
    """
    print(f'Panoptic segmentation: {image_path}')

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_ori = transform(image)

    width = image_ori.size[0]
    height = image_ori.size[1]
    image_np = np.asarray(image_ori)

    visual = Visualizer(image_np, metadata=metadata)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

    data = {
        "image": image_tensor,
        "height": height,
        "width": width
    }

    model.model.metadata = metadata

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            results = model.model.evaluate([data])

    pano_seg = results[-1]['panoptic_seg'][0]
    pano_seg_info = results[-1]['panoptic_seg'][1]

    demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)
    res = demo.get_image()
    result_image = Image.fromarray(res)

    if output_path:
        result_image.save(output_path)
        print(f'Result saved to: {output_path}')

    torch.cuda.empty_cache()

    return result_image


def main():
    parser = argparse.ArgumentParser(description='SEEM Text-to-Mask Segmentation')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='Text prompt for segmentation (e.g., "the dog", "red car")')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--config', '-c', type=str,
                        default='configs/seem/focalt_unicl_lang_demo.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint', '-p', type=str,
                        default='seem_focalt_v0.pt',
                        help='Model checkpoint path')
    parser.add_argument('--panoptic', action='store_true',
                        help='Run panoptic segmentation (segment all objects)')

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f'{base}_segmented{ext}'

    # 加载模型
    model = load_model(args.config, args.checkpoint)

    # 执行分割
    if args.panoptic:
        result = segment_panoptic(model, args.image, args.output)
    elif args.text:
        mask, result = segment_with_text(model, args.image, args.text, args.output)
    else:
        print('Please provide --text for text-guided segmentation or --panoptic for panoptic segmentation')
        return

    print('Done!')


if __name__ == '__main__':
    main()
