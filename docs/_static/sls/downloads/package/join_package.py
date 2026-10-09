"""Download and verify the complete public SLS Marmousi2 ZIP using only Python's standard library.

Usage: python join_package.py [--output StarWave-SLS-Marmousi2-Example.zip]
The output is written atomically only after every part and the entire ZIP verify.
"""
import argparse,hashlib,json,os,time,urllib.request
from pathlib import Path
BASE='https://starrmoonn.github.io/StarWave-docs/zh/_static/sls/downloads/package/'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
 manifest=json.loads(urllib.request.urlopen(BASE+'package.json',timeout=60).read())
 output=a.output or Path(manifest['filename'])
 if output.exists():raise SystemExit('Output already exists; choose a different --output path.')
 partial=output.with_name(output.name+'.partial')
 if partial.exists():raise SystemExit('Partial file already exists; inspect/remove it or choose a different --output path.')
 total=hashlib.sha256();written=0
 try:
  with partial.open('xb') as stream:
   for i,item in enumerate(manifest['parts'],1):
    for attempt in range(3):
     try:
      with urllib.request.urlopen(BASE+item['file'],timeout=120) as r:data=r.read(item['bytes']+1)
      if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Part size/hash mismatch: '+item['file'])
      break
     except Exception:
      if attempt==2:raise
      time.sleep(attempt+1)
    stream.write(data);total.update(data);written+=len(data);print(f'{i}/{len(manifest["parts"])} parts verified: {written}/{manifest["bytes"]} bytes',flush=True)
  if written!=manifest['bytes'] or total.hexdigest()!=manifest['sha256']:raise ValueError('Complete ZIP hash mismatch')
  os.replace(partial,output)
 except BaseException:
  if partial.exists():partial.unlink()
  raise
 print('Verified ZIP:',output,'SHA-256:',total.hexdigest())
if __name__=='__main__':main()
