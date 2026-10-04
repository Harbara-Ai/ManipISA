from pathlib import Path
import concurrent.futures, hashlib, json, sys, urllib.request, urllib.parse, time
ROOT = Path(__file__).resolve().parents[1]
REV = (ROOT / 'reproduction/assets-revision.txt').read_text().strip()
prefixes = sys.argv[1:] or ['Robots_p/ur5+wuji','Objects/154_wooden_box','Objects/053_mini_soccer_ball','Objects/056_tennis_ball','Objects/058_golf_ball','Objects/144_table-tennis']
def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r), r.headers.get('Link', '')
files = []
for prefix in prefixes:
    url = f'https://huggingface.co/api/datasets/Bench2Dex/Assets/tree/{REV}/{urllib.parse.quote(prefix, safe="/")}?recursive=true&limit=1000'
    while url:
        data, link = get_json(url)
        files.extend(x for x in data if x['type']=='file')
        url = next((part.split('<')[1].split('>')[0] for part in link.split(',') if 'rel="next"' in part), None)
print(f'Assets {REV}: {len(files)} files, {sum(x["size"] for x in files)/1024**2:.1f} MiB', flush=True)
def download(item):
    rel = item['path']
    target = ROOT / 'dex2bench_dataset' / rel
    digest = item.get('lfs', {}).get('oid')
    def valid():
        if not target.is_file() or target.stat().st_size != item['size']:
            return False
        return not digest or hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() == digest
    if valid(): return rel
    target.parent.mkdir(parents=True,exist_ok=True)
    url = f'https://huggingface.co/datasets/Bench2Dex/Assets/resolve/{REV}/{urllib.parse.quote(rel, safe="/")}'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, target.with_suffix(target.suffix+'.partial').open('wb') as out:
                while chunk := r.read(1024*1024): out.write(chunk)
            target.with_suffix(target.suffix+'.partial').replace(target)
            if not valid(): raise RuntimeError(f'Checksum/size mismatch: {rel}')
            return rel
        except Exception:
            if attempt==2: raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for i, path in enumerate(pool.map(download,files),1):
        if i%20==0 or i==len(files): print(f'{i}/{len(files)} verified: {path}',flush=True)
manifest_path = ROOT/'reproduction/assets-manifest.json'
old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'revision':REV,'files':[]}
by_path={x['path']:x for x in old['files']}
by_path.update({x['path']:x for x in files})
old['files'] = list(by_path.values())
manifest_path.write_text(json.dumps(old,indent=2),encoding='utf-8')
