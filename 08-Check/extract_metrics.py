import json

# 1) 2DGS results
with open('D:/CAAS/08-Check/eval_2dgs_results.json') as f:
    data = json.load(f)
print('=== eval_2dgs_results.json ===')
print(f'Total entries: {len(data)}')
for key, val in data.items():
    m = val.get('metrics', {})
    species = val.get('species')
    iteration = val.get('iteration')
    gaussians = val.get('num_gaussians')
    cams = val.get('num_total_cameras')
    test_cams = val.get('num_test_cameras')
    psnr = m.get('PSNR', 0)
    ssim = m.get('SSIM', 0)
    lpips = m.get('LPIPS', 0)
    print(f'  {key}: species={species}, iter={iteration}, gaussians={gaussians}, cams={cams}, test_cams={test_cams}, PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}')

print()

# 2) SuGaR results
with open('D:/CAAS/08-Check/eval_sugar_results.json') as f:
    data2 = json.load(f)
print('=== eval_sugar_results.json ===')
print(f'Total entries: {len(data2)}')
for key, val in data2.items():
    m = val.get('metrics', {})
    mt = val.get('model_type', '')
    gs = val.get('num_gaussians', val.get('num_sugar_points', 'N/A'))
    cams = val.get('num_total_cameras')
    test_cams = val.get('num_test_cameras')
    psnr = m.get('PSNR', 0)
    ssim = m.get('SSIM', 0)
    lpips = m.get('LPIPS', 0)
    print(f'  {key}: type={mt}, gaussians/points={gs}, cams={cams}, test_cams={test_cams}, PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}')

print()

# 3) model_statistics summary
with open('D:/CAAS/08-Check/model_statistics.json') as f:
    data3 = json.load(f)
print('=== model_statistics.json ===')
for section_key, section_val in data3.items():
    print(f'\nSection: {section_key}')
    if not isinstance(section_val, dict):
        continue
    for species, info in section_val.items():
        if not isinstance(info, dict):
            continue
        mt = info.get('model_type', '')
        nc = info.get('num_cameras', 'N/A')
        res = info.get('image_resolution', 'N/A')
        
        # Point clouds
        pcs = info.get('point_clouds', {})
        if pcs:
            for iter_key, iter_val in pcs.items():
                counts = {k: v for k, v in iter_val.items() if isinstance(v, int)}
                print(f'  {species} [{mt}] {iter_key}: {counts}, cameras={nc}, resolution={res}')
        
        # Vanilla GS point clouds
        vpcs = info.get('vanilla_gs_point_clouds', {})
        if vpcs:
            for iter_key, iter_val in vpcs.items():
                counts = {k: v for k, v in iter_val.items() if isinstance(v, int)}
                cm = info.get('coarse_mesh', {})
                cm_info = f"mesh_verts={cm.get('vertices','N/A')}, mesh_faces={cm.get('faces','N/A')}" if cm else ''
                print(f'  {species} [{mt}] {iter_key}: {counts}, cameras={nc}, resolution={res}, {cm_info}')
        
        # Meshes
        meshes = info.get('meshes', {})
        if meshes:
            for mesh_name, mesh_info in meshes.items():
                if isinstance(mesh_info, dict):
                    v = mesh_info.get('vertices', 'N/A')
                    fa = mesh_info.get('faces', 'N/A')
                    sz = mesh_info.get('size_MB', 'N/A')
                    print(f'  {species} [{mt}] {mesh_name}: vertices={v}, faces={fa}, size={sz}MB')
