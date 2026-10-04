import json, pathlib, hashlib
root=pathlib.Path('.').resolve()
manifest_path=root/'reproduction/assets-manifest.json'
m=json.loads(manifest_path.read_text())
texture=root/'dex2bench_dataset/textures/wood/wood0060.jpg'
if not any(x['path']=='textures/wood/wood0060.jpg' for x in m['files']):
    m['files'].append({'type':'file','path':'textures/wood/wood0060.jpg','size':texture.stat().st_size,'verified_local_sha256':hashlib.file_digest(texture.open('rb'),'sha256').hexdigest()})
manifest_path.write_text(json.dumps(m,indent=2))
print('Pinned downloaded assets:',len(m['files']),'total MiB:',round(sum(x['size'] for x in m['files'])/1024**2,1))
