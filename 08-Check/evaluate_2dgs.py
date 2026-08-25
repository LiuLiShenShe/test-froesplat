"""
2DGS模型渲染 + PSNR/SSIM/LPIPS评估
在2DGS虚拟环境中运行: 
  D:\CAAS\2d-gaussian-splatting-great-again-dev\.venv_uv\Scripts\python.exe
"""
import sys
import os
import json
import torch
import torchvision.transforms.functional as tf
from pathlib import Path
from PIL import Image
import logging

# 设置日志到文件，避免管道缓冲问题
LOG_FILE = Path(r"D:\CAAS\08-Check\evaluate_2dgs_v3.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# 重定向stdout/stderr到文件，避免终端缓冲区阻塞
# 2DGS Scene模块内部的print()会写入stdout，如果终端缓冲满就会阻塞
_stdout_log = open(r"D:\CAAS\08-Check\evaluate_2dgs_stdout.log", 'a', encoding='utf-8')
sys.stdout = _stdout_log
sys.stderr = _stdout_log

def tqdm(iterable, **kwargs):
    """替代tqdm，直接写入日志，避免管道阻塞"""
    total = kwargs.get('total', None)
    desc = kwargs.get('desc', '')
    items = list(iterable)
    if total is None:
        total = len(items)
    for i, item in enumerate(items):
        if i % 5 == 0 or i == total - 1:
            logger.info(f"{desc}: {i+1}/{total}")
        yield item

# 添加2DGS代码路径
DGS_ROOT = r"D:\CAAS\2d-gaussian-splatting-great-again-dev"
sys.path.insert(0, DGS_ROOT)

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.mesh_utils import GaussianExtractor
from utils.loss_utils import ssim as compute_ssim
from utils.image_utils import psnr as compute_psnr
from lpipsPyTorch import lpips as compute_lpips
from arguments import ModelParams, PipelineParams
from argparse import Namespace

BASE_DIR = Path(r"D:\CAAS")
MODEL_DIR = BASE_DIR / "05-2DGS-new"
OUTPUT_DIR = BASE_DIR / "08-Check"
EVAL_SPLIT_INTERVAL = 8  # 每8张图取1张作为测试集

def get_species_list():
    """获取所有物种目录"""
    species = []
    for d in sorted(MODEL_DIR.iterdir()):
        if d.is_dir() and (d / "point_cloud").exists():
            species.append(d.name)
    return species

def evaluate_single_species(species_name, iteration=30000):
    """对单个物种进行渲染和评估"""
    model_path = str(MODEL_DIR / species_name)
    
    logger.info(f"{'='*60}")
    logger.info(f"评估 2DGS: {species_name} (iteration={iteration})")
    logger.info(f"{'='*60}")
    
    # 读取cfg_args获取配置
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        logger.info(f"  跳过 {species_name}: cfg_args不存在")
        return None
    
    with open(cfg_path, 'r') as f:
        cfg_text = f.read().strip()
    
    # 解析cfg_args
    cfg = eval(cfg_text)
    
    # 创建dataset参数 - 包含所有ModelParams的默认属性
    dataset = Namespace(
        source_path=cfg.source_path,
        model_path=model_path, 
        images=getattr(cfg, 'images', 'images'),
        resolution=getattr(cfg, 'resolution', 2),
        white_background=getattr(cfg, 'white_background', False),
        data_device="cuda",
        eval=False,
        sh_degree=getattr(cfg, 'sh_degree', 3),
        render_items=getattr(cfg, 'render_items', ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']),
        w_normal_prior=getattr(cfg, 'w_normal_prior', ''),
        use_decoupled_appearance=getattr(cfg, 'use_decoupled_appearance', False),
    )
    
    try:
        # 加载模型
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 获取训练相机
        train_cameras = scene.getTrainCameras()
        
        # 创建评估分割: 每EVAL_SPLIT_INTERVAL张取1张作为测试
        all_cameras = list(train_cameras)
        test_indices = list(range(0, len(all_cameras), EVAL_SPLIT_INTERVAL))
        train_indices = [i for i in range(len(all_cameras)) if i not in test_indices]
        
        test_cameras = [all_cameras[i] for i in test_indices]
        
        logger.info(f"  总相机数: {len(all_cameras)}")
        logger.info(f"  测试集相机数: {len(test_cameras)} (每{EVAL_SPLIT_INTERVAL}张取一张)")
        logger.info(f"  训练集相机数: {len(train_indices)}")
        logger.info(f"  高斯点数: {gaussians.get_xyz.shape[0]}")
        
        # 创建管线参数
        pipe = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            depth_ratio=0.0,
            debug=False,
        )
        
        # 渲染和评估
        ssims = []
        psnrs = []
        lpipss = []
        per_view = {}
        
        with torch.no_grad():
            for idx, viewpoint in tqdm(enumerate(test_cameras), total=len(test_cameras), desc=f"  渲染{species_name}"):
                # 渲染图像
                render_output = render(viewpoint, gaussians, pipe, background)
                rendered = render_output["render"].clamp(0, 1)  # [3, H, W]
                
                # GT图像
                gt = viewpoint.original_image[0:3, :, :].cuda()
                
                # 扩展维度用于计算metrics
                rendered_batch = rendered.unsqueeze(0)  # [1, 3, H, W]
                gt_batch = gt.unsqueeze(0)  # [1, 3, H, W]
                
                # 计算指标
                s = compute_ssim(rendered_batch, gt_batch)
                p = compute_psnr(rendered_batch, gt_batch)
                l = compute_lpips(rendered_batch, gt_batch, net_type='vgg')
                
                ssims.append(s.item() if torch.is_tensor(s) else s)
                psnrs.append(p.item() if torch.is_tensor(p) else p)
                lpipss.append(l.item() if torch.is_tensor(l) else l)
                
                per_view[f"view_{test_indices[idx]:04d}"] = {
                    "SSIM": ssims[-1],
                    "PSNR": psnrs[-1],
                    "LPIPS": lpipss[-1]
                }
        
        # 计算平均值
        avg_ssim = sum(ssims) / len(ssims)
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_lpips = sum(lpipss) / len(lpipss)
        
        logger.info(f"  结果: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        
        result = {
            "species": species_name,
            "model_type": "2DGS",
            "iteration": iteration,
            "num_gaussians": gaussians.get_xyz.shape[0],
            "num_total_cameras": len(all_cameras),
            "num_test_cameras": len(test_cameras),
            "eval_split_interval": EVAL_SPLIT_INTERVAL,
            "metrics": {
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "LPIPS": avg_lpips,
            },
            "per_view": per_view,
        }
        
        # 释放显存
        del gaussians, scene
        torch.cuda.empty_cache()
        
        return result
    
    except Exception as e:
        logger.error(f"  评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"species": species_name, "error": str(e)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    species_list = get_species_list()
    logger.info(f"找到 {len(species_list)} 个2DGS模型: {species_list}")
    
    # 加载已有结果，跳过已完成的
    output_path = OUTPUT_DIR / "eval_2dgs_results.json"
    all_results = {}
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        logger.info(f"已加载 {len(all_results)} 个已有结果")
    
    for species in species_list:
        # 评估 iteration 30000
        key_30k = f"{species}_iter30000"
        if key_30k in all_results and "error" not in all_results[key_30k]:
            logger.info(f"跳过已完成: {key_30k}")
        else:
            result_30k = evaluate_single_species(species, iteration=30000)
            if result_30k:
                all_results[key_30k] = result_30k
        
        # 评估 iteration 7000
        key_7k = f"{species}_iter7000"
        if key_7k in all_results and "error" not in all_results[key_7k]:
            logger.info(f"跳过已完成: {key_7k}")
        else:
            result_7k = evaluate_single_species(species, iteration=7000)
            if result_7k:
                all_results[key_7k] = result_7k
        
        # 保存中间结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"已保存 {len(all_results)} 个结果到JSON")
    
    # 打印汇总表格
    logger.info("=" * 90)
    logger.info("2DGS 评估结果汇总")
    logger.info("=" * 90)
    
    for key, result in sorted(all_results.items()):
        if "error" in result:
            logger.info(f"{result['species']}: ERROR")
            continue
        metrics = result.get("metrics", {})
        logger.info(f"{result.get('species','?')}, iter={result.get('iteration','?')}, "
              f"PSNR={metrics.get('PSNR', 0):.4f}, SSIM={metrics.get('SSIM', 0):.4f}, "
              f"LPIPS={metrics.get('LPIPS', 0):.4f}, pts={result.get('num_gaussians', 0)}")
    
    logger.info(f"ALL DONE - 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
