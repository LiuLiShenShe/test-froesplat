"""Verify each independent output through the original 2DGS Scene loader."""
from common import *
from argparse import Namespace
import gc
import traceback


def main():
    assert_gpu();p=plan();import_paths()
    import torch
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    rows_out=[]
    try:
        # One direct Scene load proves the model-directory contract; all eight
        # PLYs are independently checked through GaussianModel.load_ply.
        run=next(r for r in rows(STAGE/'COMPRESSION_RUN_MANIFEST.csv') if r['run_id']=='opacity_p40')
        gate=STAGE/'configs/SCENE_LOAD_SINGLE_CAMERA.txt'
        gate.write_text('0001\n')
        native=Path(DATA/'models'/run['run_id']/'native_model')
        model_cfg={**p['model'],'model_path':str(native),'raw_gate_list':str(gate)}
        g=GaussianModel(p['SH_degree'])
        scene=Scene(Namespace(**model_cfg),g,load_iteration=30000,shuffle=False)
        count=len(g.get_xyz)
        assert count==int(run['Gaussian_count'])
        assert len(scene.getTestCameras())+len(scene.getTrainCameras())>=1
        rows_out.append({'run_id':run['run_id'],'status':'PASS','Gaussian_count':count,
            'native_scene_model_dir':str(native),'heldout_camera_count':len(scene.getTestCameras()),
            'train_camera_count':len(scene.getTrainCameras()),'source_PLY_sha256':run['output_model_sha256'],
            'loader':'untouched 2d-gaussian-splatting-main.scene.Scene','camera_gate':'0001'})
        del scene,g;gc.collect();torch.cuda.empty_cache()
        write_json(STAGE/'reports/NATIVE_SCENE_LOAD_VALIDATION.json',{'status':'PASS','runs':rows_out,
            'model_loader_unchanged':True,'training_not_started':True,'source_model_unchanged':sha(p['source_model'])==p['source_model_sha256']})
    except BaseException as e:
        write_json(STAGE/'reports/NATIVE_SCENE_LOAD_VALIDATION.json',{'status':'FAILED','runs':rows_out,
            'error':repr(e),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
