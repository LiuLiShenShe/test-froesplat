"""Apply confirmed UI masks/landmarks to derived outputs, without GPU work."""
from common import *
import subprocess

if __name__=='__main__':
 for script in ['extract_instances.py','measure_phenotypes.py','manual_errors.py','build_report.py','validate_delivery.py']:
  subprocess.run([sys.executable,str(Path(__file__).with_name(script))],check=True)
 print('Confirmed corrections applied; original input mesh and historical masks unchanged.')
