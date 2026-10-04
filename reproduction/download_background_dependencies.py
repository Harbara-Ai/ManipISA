from pathlib import Path
import re, subprocess, sys
root=Path(__file__).resolve().parents[1]
assets=root/'dex2bench_dataset'
prefixes=set()
missing=[]
for p in (assets/'scenes/ithor/FloorPlan220_physics').rglob('*.usda'):
    for raw in re.findall(r'@([^@]+)@',p.read_text(encoding='utf-8')):
        q=(p.parent/raw).resolve()
        if q.exists(): continue
        rel=q.relative_to(assets).as_posix()
        if rel.startswith('Objects/thor/'): prefixes.add('/'.join(rel.split('/')[:3]))
        else: missing.append(rel)
print('Background model directories',len(prefixes),'other missing',missing,flush=True)
if prefixes: subprocess.run([sys.executable,str(root/'reproduction/download_assets.py'),*sorted(prefixes)],check=True)
