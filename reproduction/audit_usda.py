from pathlib import Path
import re,json
root=Path('dex2bench_dataset').resolve()
missing=[]
for f in root.rglob('*.usda'):
    for raw in re.findall(r'@([^@]+)@',f.read_text(encoding='utf-8',errors='replace')):
        if '://' not in raw and not (f.parent/raw).resolve().exists(): missing.append({'source':f.relative_to(root).as_posix(),'reference':raw})
print(json.dumps(missing[:25],indent=2)); print('missing USD ASCII references:',len(missing))
Path('reproduction/usda-missing-references.json').write_text(json.dumps(missing,indent=2))
