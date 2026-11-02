"""CPU-only archive verification and source-image audit; no model or GPU imports."""
import argparse, collections, concurrent.futures, hashlib, json, os, time, zipfile
from pathlib import Path

def digest(path, kind='sha256'):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, kind).hexdigest()

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    temporary.replace(path)

def image_record(item):
    from PIL import Image, ImageOps
    import numpy as np
    from scipy.fft import dctn
    path, root = item
    with Image.open(path) as original:
        original.load()
        gps = bool(original.getexif().get(34853))
        mode, size = original.mode, original.size
        image = ImageOps.exif_transpose(original).convert('RGB')
        rgb = np.asarray(image)
        gray = image.convert('L')
        dh = np.asarray(gray.resize((9,8), Image.Resampling.LANCZOS))
        dh = int.from_bytes(np.packbits(dh[:,1:]>dh[:,:-1]).tobytes(), 'big')
        a = np.asarray(gray.resize((32,32), Image.Resampling.LANCZOS), dtype=np.float64)
        low = dctn(a, type=2, norm='ortho')[:8,:8].flatten()
        mask = low > np.median(low[1:]); mask[0] = False
        ph = int.from_bytes(np.packbits(mask).tobytes(), 'big')
    relative = path.relative_to(root).as_posix()
    return {'path': relative, 'file_bytes': path.stat().st_size, 'file_sha256': digest(path),
            'decoded_rgb_sha256': hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest(),
            'width':size[0], 'height':size[1], 'source_mode':mode, 'exif_gps_present':gps,
            'dhash64':f'{dh:016x}', 'phash64':f'{ph:016x}', 'parent_folder':path.parent.name}

def run(args):
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    outcome = root/'audit_complete.json'
    if outcome.exists():
        raise RuntimeError('Preserve completed audit; do not silently repeat')
    archive = root/args.archive
    if args.dataset == 'eurosat_rgb':
        expected_kind, expected = 'md5', 'c8fa014336c82ac7804f0398fcb19387'
    else:
        expected_kind, expected = 'sha256', '6fe6ca45914a453e9b4907f5c2b115fc8cb8481609bd9f8f93008dabf7aca9a3'
    assert digest(archive, expected_kind)==expected, 'Archive differs from pinned published checksum'
    write(root/'archive_verified.json', {'dataset':args.dataset,'archive':str(archive),
          'bytes':archive.stat().st_size,'md5':digest(archive,'md5'),'sha256':digest(archive),
          'published_checksum_kind':expected_kind,'published_checksum':expected,'status':'verified'})
    extracted = root/'original_extracted'
    if extracted.exists():
        raise RuntimeError('Partial extraction preserved; use a reviewed fresh audit directory')
    extracted.mkdir()
    with zipfile.ZipFile(archive) as z:
        names = [{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist()]
        write(root/'archive_members.json', names)
        assert z.testzip() is None, 'ZIP CRC check failed'
        for info in z.infolist():
            target = (extracted/info.filename).resolve()
            assert target.is_relative_to(extracted), 'ZIP path traversal'
            assert not ((info.external_attr >> 16) & 0o170000)==0o120000, 'Unexpected archive symlink'
            z.extract(info,extracted)
    images = sorted(p for p in extracted.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff') and '__MACOSX' not in p.parts)
    image_set = set(images)
    metadata = [p.relative_to(extracted).as_posix() for p in extracted.rglob('*') if p.is_file() and p not in image_set]
    write(root/'nonimage_members.json', metadata)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(image_record,((p,extracted) for p in images)),1):
            records.append(row)
            if i%1000==0:
                write(root/'audit_progress.json',{'pid':os.getpid(),'images_done':i,'images_total':len(images),'unix':time.time()})
    with (root/'image_inventory.jsonl').open('x',encoding='utf8') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    exact=collections.defaultdict(list)
    for row in records:exact[row['decoded_rgb_sha256']].append(row['path'])
    exact={k:v for k,v in exact.items() if len(v)>1}
    write(root/'decoded_exact_duplicate_groups.json',exact)
    sizes=collections.Counter((r['width'],r['height']) for r in records)
    summary={'status':'complete','dataset':args.dataset,'pid':os.getpid(),'images':len(records),
             'parent_folder_counts':dict(collections.Counter(r['parent_folder'] for r in records)),
             'source_modes':dict(collections.Counter(r['source_mode'] for r in records)),
             'dimensions':[{'width':w,'height':h,'count':n} for (w,h),n in sizes.most_common()],
             'exif_gps_images':sum(r['exif_gps_present'] for r in records),
             'exact_duplicate_groups':len(exact),'exact_duplicate_images':sum(map(len,exact.values())),
             'inventory_sha256':digest(root/'image_inventory.jsonl'),
             'nonimage_files':metadata,'GPU_used':False,'split_status':'fixed_reconstruction_rule',
             'near_duplicate_rule':'fixed dHash64<=4 AND pHash64<=4; subsequent CPU audit, not model dependent',
             'hash_definition':'EXIF transposed RGB; dHash9x8 LANCZOS; pHash32x32 LANCZOS+float64 orthonormal DCT, first8x8, median excludingDC, DCbit0'}
    if args.dataset=='eurosat_rgb':
        assert len(records)==27000 and len(summary['parent_folder_counts'])==10
        assert all((r['width'],r['height'])==(64,64) for r in records)
    write(outcome,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('dimensions','nonimage_files')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dataset',choices=['eurosat_rgb','plantwild_v1'],required=True);p.add_argument('--archive',required=True)
    run(p.parse_args())
