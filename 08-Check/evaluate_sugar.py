"""
3DGS/SuGaR模型渲染 + PSNR/SSIM/LPIPS评估
在SuGaR虚拟环境中运行: 
  D:\CAAS\SuGaR-main\.venv\Scripts\python.exe
"""
import sys
import os
import json
import torch
import open3d as o3d
from pathlib import Path
import logging

# 设置日志
LOG_FILE = Path(r"D:\CAAS\08-Check\evaluate_sugar.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# 重定向stdout/stderr到文件，避免终端缓冲阻塞
_stdout_log = open(r"D:\CAAS\08-Check\evaluate_sugar_stdout.log", 'a', encoding='utf-8')
sys.stdout = _stdout_log
sys.stderr = _stdout_log

def tqdm(iterable, **kwargs):
    """替代tqdm，直接写入日志避免管道阻塞"""
    total = kwargs.get('total', None)
    desc = kwargs.get('desc', '')
    items = list(iterable)
    if total is None:
        total = len(items)
    for i, item in enumerate(items):
        if i % 5 == 0 or i == total - 1:
            logger.info(f"{desc}: {i+1}/{total}")
        yield item

# 添加SuGaR代码路径
SUGAR_ROOT = r"D:\CAAS\SuGaR-main"
sys.path.insert(0, SUGAR_ROOT)
os.chdir(SUGAR_ROOT)  # SuGaR代码中有些相对路径需要

os.makedirs('./lpipsPyTorch/weights/', exist_ok=True)
torch.hub.set_dir('./lpipsPyTorch/weights/')

from gaussian_splatting.utils.loss_utils import ssim as compute_ssim
from gaussian_splatting.utils.image_utils import psnr as compute_psnr
from gaussian_splatting.lpipsPyTorch import lpips as compute_lpips
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB

BASE_DIR = Path(r"D:\CAAS")
SUGAR_GS_DIR = BASE_DIR / "07-SuGaR-GS"
SUGAR_MESH_DIR = BASE_DIR / "07-SuGaR-Mesh"
COLMAP_DIR = BASE_DIR / "04-COLMAP"
OUTPUT_DIR = BASE_DIR / "08-Check"

EVAL_SPLIT_INTERVAL = 8  # 每8张图取1张作为测试集


def get_species_list():
    """获取所有有效物种目录"""
    species = []
    for d in sorted(SUGAR_GS_DIR.iterdir()):
        if d.is_dir():
            vanilla_gs_path = d / "vanilla_gs" / "point_cloud"
            if vanilla_gs_path.exists():
                species.append(d.name)
    return species


def evaluate_vanilla_gs(species_name):
    """评估vanilla 3DGS (iteration 7000)"""
    logger.info(f"--- 评估 Vanilla 3DGS: {species_name} ---")
    
    source_path = str(COLMAP_DIR / species_name)
    output_path = str(SUGAR_GS_DIR / species_name / "vanilla_gs")
    
    if not os.path.exists(source_path):
        logger.info(f"跳过: source_path不存在 {source_path}")
        return None
    
    try:
        # 加载模型 (只有iteration 7000可用)
        nerfmodel = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=output_path,
            iteration_to_load=7000,
            load_gt_images=True,
            eval_split=True,
            eval_split_interval=EVAL_SPLIT_INTERVAL,
        )
        
        n_test = len(nerfmodel.test_cameras)
        n_total = len(nerfmodel.cam_list) + n_test
        n_gaussians = nerfmodel.gaussians.get_xyz.shape[0]
        
        logger.info(f"总相机数: {n_total}")
        logger.info(f"测试集: {n_test}, 训练集: {len(nerfmodel.cam_list)}")
        logger.info(f"高斯点数: {n_gaussians}")
        
        ssims = []
        psnrs = []
        lpipss = []
        per_view = {}
        
        with torch.no_grad():
            for cam_idx in tqdm(range(n_test), desc=f"    渲染{species_name}"):
                # GT图像
                gt_img = nerfmodel.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                
                # 渲染图像
                rendered_img = nerfmodel.render_image(
                    nerf_cameras=nerfmodel.test_cameras,
                    camera_indices=cam_idx
                ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                
                s = compute_ssim(rendered_img, gt_img)
                p = compute_psnr(rendered_img, gt_img)
                l = compute_lpips(rendered_img, gt_img, net_type='vgg')
                
                ssims.append(s.item() if torch.is_tensor(s) else s)
                psnrs.append(p.item() if torch.is_tensor(p) else p)
                lpipss.append(l.item() if torch.is_tensor(l) else l)
                
                per_view[f"view_{cam_idx:04d}"] = {
                    "SSIM": ssims[-1], "PSNR": psnrs[-1], "LPIPS": lpipss[-1]
                }
        
        avg_ssim = sum(ssims) / len(ssims)
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_lpips = sum(lpipss) / len(lpipss)
        
        logger.info(f"Vanilla 3DGS 7k: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        
        result = {
            "model_type": "3DGS (Vanilla)",
            "iteration": 7000,
            "num_gaussians": n_gaussians,
            "num_total_cameras": n_total,
            "num_test_cameras": n_test,
            "eval_split_interval": EVAL_SPLIT_INTERVAL,
            "metrics": {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips},
            "per_view": per_view,
        }
        
        del nerfmodel
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        logger.error(f"Vanilla 3DGS评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def evaluate_sugar_refined(species_name):
    """评估SuGaR refined模型"""
    logger.info(f"--- 评估 SuGaR Refined: {species_name} ---")
    
    source_path = str(COLMAP_DIR / species_name)
    output_path = str(SUGAR_GS_DIR / species_name / "vanilla_gs")
    
    # SuGaR refined路径
    refined_dir = SUGAR_GS_DIR / species_name / "refined"
    coarse_mesh_dir = SUGAR_GS_DIR / species_name / "coarse_mesh"
    
    if not refined_dir.exists() or not coarse_mesh_dir.exists():
        logger.info(f"跳过: refined或coarse_mesh目录不存在")
        return None
    
    # 查找refined checkpoint
    refined_pt = None
    for sub in refined_dir.iterdir():
        if sub.is_dir():
            for pt_file in sub.glob("*.pt"):
                refined_pt = str(pt_file)
                break
    
    if refined_pt is None:
        logger.info(f"跳过: 未找到refined checkpoint")
        return None
    
    # 查找coarse mesh
    coarse_mesh_file = None
    for mesh_file in coarse_mesh_dir.glob("*.ply"):
        coarse_mesh_file = str(mesh_file)
        break
    
    if coarse_mesh_file is None:
        logger.info(f"跳过: 未找到coarse mesh")
        return None
    
    try:
        # 加载基模型
        nerfmodel = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=output_path,
            iteration_to_load=7000,
            load_gt_images=True,
            eval_split=True,
            eval_split_interval=EVAL_SPLIT_INTERVAL,
        )
        
        n_test = len(nerfmodel.test_cameras)
        sh_deg_to_use = nerfmodel.gaussians.active_sh_degree
        
        # 加载coarse mesh
        logger.info(f"加载coarse mesh: {coarse_mesh_file}")
        o3d_mesh = o3d.io.read_triangle_mesh(coarse_mesh_file)
        
        # 加载refined SuGaR模型
        logger.info(f"加载refined checkpoint: {refined_pt}")
        checkpoint = torch.load(refined_pt, map_location=nerfmodel.device)
        
        # 从checkpoint路径推断n_gaussians_per_surface_triangle
        n_gaussians_per_face = 1  # 默认
        if 'gaussperface' in refined_pt:
            import re
            match = re.search(r'gaussperface(\d+)', refined_pt)
            if match:
                n_gaussians_per_face = int(match.group(1))
        
        refined_sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=checkpoint['state_dict']['_points'],
            colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
            initialize=False,
            sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
            keep_track_of_knn=False,
            knn_to_track=0,
            beta_mode='average',
            surface_mesh_to_bind=o3d_mesh,
            n_gaussians_per_surface_triangle=n_gaussians_per_face,
        )
        refined_sugar.load_state_dict(checkpoint['state_dict'])
        refined_sugar.eval()
        
        n_sugar_points = refined_sugar._points.shape[0]
        logger.info(f"SuGaR points: {n_sugar_points}")
        logger.info(f"测试集: {n_test} 图像")
        
        ssims = []
        psnrs = []
        lpipss = []
        per_view = {}
        
        with torch.no_grad():
            for cam_idx in tqdm(range(n_test), desc=f"    渲染SuGaR {species_name}"):
                # GT图像
                gt_img = nerfmodel.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                
                # SuGaR渲染
                sugar_img = refined_sugar.render_image_gaussian_rasterizer(
                    nerf_cameras=nerfmodel.test_cameras,
                    camera_indices=cam_idx,
                    verbose=False,
                    bg_color=None,
                    sh_deg=sh_deg_to_use,
                    compute_color_in_rasterizer=True,
                ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                
                s = compute_ssim(sugar_img, gt_img)
                p = compute_psnr(sugar_img, gt_img)
                l = compute_lpips(sugar_img, gt_img, net_type='vgg')
                
                ssims.append(s.item() if torch.is_tensor(s) else s)
                psnrs.append(p.item() if torch.is_tensor(p) else p)
                lpipss.append(l.item() if torch.is_tensor(l) else l)
                
                per_view[f"view_{cam_idx:04d}"] = {
                    "SSIM": ssims[-1], "PSNR": psnrs[-1], "LPIPS": lpipss[-1]
                }
        
        avg_ssim = sum(ssims) / len(ssims)
        avg_psnr = sum(psnrs) / len(psnrs)  
        avg_lpips = sum(lpipss) / len(lpipss)
        
        logger.info(f"SuGaR Refined: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        
        result = {
            "model_type": "SuGaR Refined",
            "num_sugar_points": n_sugar_points,
            "num_total_cameras": len(nerfmodel.cam_list) + n_test,
            "num_test_cameras": n_test,
            "eval_split_interval": EVAL_SPLIT_INTERVAL,
            "metrics": {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips},
            "per_view": per_view,
        }
        
        del nerfmodel, refined_sugar
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        logger.error(f"SuGaR评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    species_list = get_species_list()
    logger.info(f"找到 {len(species_list)} 个SuGaR模型: {species_list}")
    
    # 加载已有结果
    output_path = OUTPUT_DIR / "eval_sugar_results.json"
    all_results = {}
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        logger.info(f"已加载 {len(all_results)} 个已有结果")
    
    for species in species_list:
        logger.info(f"{'='*60}")
        logger.info(f"处理: {species}")
        logger.info(f"{'='*60}")
        
        # 评估vanilla 3DGS
        key_vanilla = f"{species}_vanilla_3dgs_7k"
        if key_vanilla in all_results and "error" not in all_results[key_vanilla]:
            logger.info(f"跳过已完成: {key_vanilla}")
        else:
            vanilla_result = evaluate_vanilla_gs(species)
            if vanilla_result:
                all_results[key_vanilla] = vanilla_result
        
        # 评估SuGaR refined
        key_sugar = f"{species}_sugar_refined"
        if key_sugar in all_results and "error" not in all_results[key_sugar]:
            logger.info(f"跳过已完成: {key_sugar}")
        else:
            sugar_result = evaluate_sugar_refined(species)
            if sugar_result:
                all_results[key_sugar] = sugar_result
        
        # 保存中间结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"已保存 {len(all_results)} 个结果到JSON")
    
    logger.info("ALL DONE - SuGaR评估完成")


if __name__ == "__main__":
    main()
