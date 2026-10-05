"""Download pinned object directories used by the release task catalog.

Default is a size/missing-file plan. --download verifies files then installs
through temporary files. The earlier task-27 asset manifest is preserved.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manipisa.evaluation.catalog import inventory


def retry_get(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=45)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def list_files(prefix, revision):
    url = f'https://huggingface.co/api/datasets/Bench2Dex/Assets/tree/{revision}/{urllib.parse.quote(prefix, safe="/")}?recursive=true&limit=1000'
    files = []
    while url:
        with retry_get(url) as response:
            data, link = json.load(response), response.headers.get("Link", "")
        files.extend(item for item in data if item["type"] == "file")
        url = next((part.split("<")[1].split(">")[0] for part in link.split(",") if 'rel="next"' in part), None)
    if not files:
        raise ValueError(f"Empty asset directory at pinned revision: {prefix}")
    return files


def target_path(item):
    rel = PurePosixPath(item["path"])
    assets = (ROOT / "dex2bench_dataset").resolve()
    target = (assets / str(rel)).resolve()
    if rel.is_absolute() or ".." in rel.parts or not target.is_relative_to(assets):
        raise ValueError("Remote asset path escapes dataset")
    return target


def valid_file(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    with path.open("rb") as stream:
        if item.get("lfs", {}).get("oid"):
            return hashlib.file_digest(stream, "sha256").hexdigest() == item["lfs"]["oid"]
        digest = hashlib.sha1(f'blob {item["size"]}\0'.encode())
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest() == item["oid"]


def install(item, revision):
    target = target_path(item)
    if valid_file(target, item):
        return
    # Do not silently replace a locally edited asset.
    if target.exists():
        raise ValueError(f"Existing file differs from pinned asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".benchmark-partial")
    url = f'https://huggingface.co/datasets/Bench2Dex/Assets/resolve/{revision}/{urllib.parse.quote(item["path"], safe="/")}'
    for attempt in range(3):
        try:
            with retry_get(url) as response, partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if not valid_file(partial, item):
                raise ValueError(f'Hash/size mismatch: {item["path"]}')
            partial.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    revision = (ROOT / "reproduction/assets-revision.txt").read_text().strip()
    catalog = inventory(ROOT / "Bench2Dex")
    files = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for group in pool.map(lambda p: list_files(p, revision), catalog["asset_prefixes"]):
            files.extend(group)
    files = sorted({item["path"]: item for item in files}.values(), key=lambda x: x["path"])
    missing = [item for item in files if not target_path(item).is_file()]
    plan = {"revision": revision, "task_catalog_sha256": catalog["task_catalog_sha256"],
            "file_count": len(files), "missing_file_count": len(missing),
            "total_bytes": sum(f["size"] for f in files), "missing_bytes": sum(f["size"] for f in missing),
            "files": files, "verified": False, "errors": []}
    output = ROOT / "artifacts/benchmark-asset-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f'{len(files)} files; {len(missing)} absent, {plan["missing_bytes"]/1024**3:.3f} GiB to download', flush=True)
    if not args.download:
        return
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(install, item, revision): item for item in files}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                plan["errors"].append({"path": futures[future]["path"], "error": str(exc)})
            if i % 25 == 0 or i == len(files):
                print(f'{i}/{len(files)} checked; {len(plan["errors"])} errors', flush=True)
    plan["verified"] = not plan["errors"]
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if plan["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
