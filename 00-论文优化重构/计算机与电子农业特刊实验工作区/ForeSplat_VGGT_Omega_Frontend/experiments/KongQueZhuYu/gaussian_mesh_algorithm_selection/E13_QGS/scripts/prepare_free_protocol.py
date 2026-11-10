"""Record the single shared comparison camera protocol; never recompute poses."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 shared=ROOT.parent/'comparison/shared_free_views.json'
 assert shared.is_file()
 (ROOT/'free_view_protocol_reference.json').write_text(json.dumps(dict(source=str(shared.resolve()),sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),full_camera_matrices_consumed_exactly=True),indent=2)+'\n')
 print('Single shared free-camera protocol:',shared.resolve())
