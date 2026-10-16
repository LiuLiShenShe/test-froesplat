import argparse
import json
from . import compress_model


def main():
    p=argparse.ArgumentParser(description='Independent, inference-only 2DGS Gaussian pruning')
    p.add_argument('--input-model',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--method',default='opacity')
    p.add_argument('--prune-ratio',type=str,required=True)
    p.add_argument('--config',help='JSON file; enabled=false bypasses scoring and deletion')
    a=p.parse_args()
    config=json.loads(open(a.config).read()) if a.config else {}
    result=compress_model(a.input_model,a.output_dir,a.method,a.prune_ratio,config)
    print(json.dumps({k:str(v) if hasattr(v,'as_posix') else v for k,v in result.__dict__.items()},indent=2))


if __name__=='__main__':
    main()
