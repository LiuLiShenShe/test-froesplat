"""Reuse historical full-resolution208 mesh evaluation without changing historical code."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["OMP_NUM_THREADS"]="8"
import argparse,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT.parents[1]/"geometry_mesh_final_selection"
sys.path.insert(0,str(HISTORY/"scripts"))
import evaluate_meshes as official_protocol
from surface_protocol import write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--shared-depth-range",action="store_true")
    a=parser.parse_args()
    folder=ROOT/("meshes_shared_depth_range" if a.shared_depth_range else "meshes")
    method="P3_GS2Mesh_shared_depth_range" if a.shared_depth_range else "P3_GS2Mesh"
    official_protocol.METHOD_INPUT[method]="e10_gggs_surface"
    sys.argv=[__file__,"--mesh-dir",str(folder),"--method",method]
    official_protocol.main()
    write(folder/"evaluation/provenance.json",dict(source_script=str(Path(official_protocol.__file__).resolve()),
       source_sha256=hashlib.sha256(Path(official_protocol.__file__).read_bytes()).hexdigest(),
       source_input="Same208 E10 camera and RAP mask; native GGGS median depth is reference proxy, not3D truth",
       original_script_modified=False,method_registered_in_memory_only=True))

if __name__=="__main__":
    main()
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

