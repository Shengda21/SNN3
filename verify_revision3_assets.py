"""Verify the R3 model companion directly in its ZIP or restored directory."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--archive',type=Path);g.add_argument('--folder',type=Path)
 a=p.parse_args();manifest=json.loads((ROOT/'revision3_assets.json').read_text())
 archive=zipfile.ZipFile(a.archive) if a.archive else None
 try:
  for row in manifest['files']:
   if archive:
    assert archive.getinfo(row['path']).file_size==row['bytes'],row['path']
    stream=archive.open(row['path'])
   else:
    path=a.folder/row['path'];assert path.stat().st_size==row['bytes'],str(path)
    stream=path.open('rb')
   with stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
   assert actual==row['sha256'],row['path']
 finally:
  if archive:archive.close()
 print(json.dumps({'status':'passed','files':len(manifest['files']),
  'bytes':sum(r['bytes'] for r in manifest['files'])}))
if __name__=='__main__':main()
