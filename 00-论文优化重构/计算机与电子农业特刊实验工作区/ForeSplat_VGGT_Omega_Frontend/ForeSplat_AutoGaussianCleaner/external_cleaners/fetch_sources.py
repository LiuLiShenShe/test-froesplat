"""Fetch only the three pinned official repositories; do not install GPU/codec stacks."""
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent / 'upstream'
SOURCES = [
    ('CorbeauSplat', 'freddewitt/CorbeauSplat', '79bda917ddedfc1a26d02ae618c73296e4b133c5'),
    ('3dgsconverter', 'francescofugazzi/3dgsconverter', '99bec766bb074d49da89614d077fba58fdd843ae'),
    ('splat-slim', 'Daceyyreal/splat-slim', '91e08c8b39a2f6b062eb3208eac12dc7b1c58de7'),
]
BASE.mkdir(parents=True, exist_ok=True)
for name, repo, sha in SOURCES:
    target = BASE / name
    if target.exists():
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != sha:
            raise RuntimeError(f'{target}: existing checkout differs from pinned commit; preserved')
        continue
    target.mkdir()
    subprocess.run(['git', 'init', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', 'https://github.com/'+repo], check=True)
    subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', sha], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
