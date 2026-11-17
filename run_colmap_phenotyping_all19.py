"""Convenient repository-root entry for the existing uv phenotyping environment."""
from pathlib import Path
import runpy,sys
code=Path(__file__).resolve().parent/'00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/phenotyping'
sys.path.insert(0,str(code))
runpy.run_path(str(code/'run_colmap_phenotyping_all19.py'),run_name='__main__')
