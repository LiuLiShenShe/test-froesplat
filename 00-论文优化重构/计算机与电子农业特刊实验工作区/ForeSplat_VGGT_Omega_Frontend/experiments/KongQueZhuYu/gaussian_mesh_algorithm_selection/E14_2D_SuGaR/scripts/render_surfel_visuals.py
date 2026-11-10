import runpy,sys
from pathlib import Path
sys.argv=["render_visual_views.py","--method","surfel"]+sys.argv[1:]
runpy.run_path(str(Path(__file__).with_name("render_visual_views.py")),run_name="__main__")
