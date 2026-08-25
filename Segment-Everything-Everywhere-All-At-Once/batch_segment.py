#!/usr/bin/env python
# --------------------------------------------------------
# SEEM Batch Segmentation Script
# 批量处理文件夹下的所有图片
# --------------------------------------------------------

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp

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
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from modeling.language.loss import vl_similarity

# 图像预处理
t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other', '').replace('-merged', '') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color']) / 255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def load_model(config_path='configs/seem/focalt_unicl_lang_demo.yaml',
               checkpoint_path='seem_focalt_v0.pt'):
    """加载 SEEM 模型"""
    print('Loading model...')

    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt))
    model = model.from_pretrained(checkpoint_path)
    model = model.eval().cuda()

    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ['background'], is_eval=True
        )

    print('Model loaded!')
    return model


def segment_with_text(model, image_path, text_prompt, output_path):
    """使用文本提示进行图像分割，输出保留原图颜色，背景黑色"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)

        image_ori = transform(image)
        width = image_ori.size[0]
        height = image_ori.size[1]
        image_np = np.asarray(image_ori)

        image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

        data = {
            "image": image_tensor,
            "height": height,
            "width": width,
            "text": [text_prompt]
        }

        model.model.task_switch['spatial'] = False
        model.model.task_switch['visual'] = False
        model.model.task_switch['grounding'] = True
        model.model.task_switch['audio'] = False

        batch_inputs = [data]

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                results, image_size, extra = model.model.evaluate_demo(batch_inputs)

        # 处理结果
        pred_masks = results['pred_masks'][0]
        v_emb = results['pred_captions'][0]
        t_emb = extra['grounding_class']

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id, :, :]

        # 调整掩码到原始图像尺寸
        pred_masks_pos = (F.interpolate(
            pred_masks_pos[None, ],
            image_size[-2:],
            mode='bilinear'
        )[0, :, :data['height'], :data['width']] > 0.0).float()

        # 调整到原始图像尺寸
        mask_resized = F.interpolate(
            pred_masks_pos[None, ],
            (original_size[1], original_size[0]),
            mode='bilinear'
        )[0, 0].cpu().numpy()

        # 创建二值掩码
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # 加载原始图像（原始尺寸）
        original_image = np.array(Image.open(image_path).convert('RGB'))

        # 创建黑色背景
        result_image = np.zeros_like(original_image)

        # 将原图中掩码区域复制到结果图像
        result_image[mask_binary == 1] = original_image[mask_binary == 1]

        # 保存结果
        result_pil = Image.fromarray(result_image)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_pil.save(output_path)

        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f'Error processing {image_path}: {e}')
        return False


def segment_panoptic(model, image_path, output_path):
    """全景分割"""
    try:
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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_image.save(output_path)

        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f'Error processing {image_path}: {e}')
        return False


def get_all_images(input_dir):
    """获取文件夹下所有图片文件"""
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description='SEEM Batch Segmentation')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='Text prompt for segmentation')
    parser.add_argument('--panoptic', action='store_true',
                        help='Run panoptic segmentation')
    parser.add_argument('--config', '-c', type=str,
                        default='configs/seem/focalt_unicl_lang_demo.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint', '-p', type=str,
                        default='seem_focalt_v0.pt',
                        help='Model checkpoint path')
    parser.add_argument('--ext', '-e', type=str, default='.jpg',
                        help='Output file extension (default: .jpg)')

    args = parser.parse_args()

    # 验证参数
    if not args.text and not args.panoptic:
        print('Error: Please provide --text or use --panoptic')
        return

    # 获取所有图片
    print(f'Scanning {args.input}...')
    image_files = get_all_images(args.input)
    print(f'Found {len(image_files)} images')

    if len(image_files) == 0:
        print('No images found!')
        return

    # 加载模型
    model = load_model(args.config, args.checkpoint)

    # 处理所有图片
    success_count = 0
    fail_count = 0

    print(f'\nProcessing with mode: {"Panoptic" if args.panoptic else f"Text: {args.text}"}')
    print(f'Output directory: {args.output}')
    print('-' * 50)

    for image_path in tqdm(image_files, desc='Processing'):
        # 计算相对路径
        rel_path = os.path.relpath(image_path, args.input)

        # 更改扩展名
        base_name = os.path.splitext(rel_path)[0]
        output_path = os.path.join(args.output, base_name + args.ext)

        # 跳过已存在的文件
        if os.path.exists(output_path):
            success_count += 1
            continue

        # 处理
        if args.panoptic:
            success = segment_panoptic(model, image_path, output_path)
        else:
            success = segment_with_text(model, image_path, args.text, output_path)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print('-' * 50)
    print(f'Done! Success: {success_count}, Failed: {fail_count}')


if __name__ == '__main__':
    main()
