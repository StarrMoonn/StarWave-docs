"""Validate the actual Scalar3D report assets and bilingual Example chapter."""
from pathlib import Path
import ast,base64,hashlib,json,re,struct,sys,io,zipfile
from urllib.parse import urlsplit,unquote
from pypdf import PdfReader

PROJECT=Path(__file__).resolve().parents[1]
ASSETS=PROJECT/'docs/_static/scalar3d'
CASES=('enclosed','surface','layered')
FIGURES=('acquisition','model_3d','model_comparison','convergence','gradients_normalized','gradients_shared_scale','shot_gather')
PAGES=('index',*CASES,'reproduce')

def manifest():
 return json.loads((ASSETS/'manifest.json').read_text())

def approved_assets():
 return {ASSETS/e['path'] for e in manifest()['assets']} | {ASSETS/'manifest.json'}

def check(root):
 from check_site import HTMLTree,element_text
 errors=[];root=Path(root).resolve();m=manifest()
 listed=approved_assets();actual={p for p in ASSETS.rglob('*') if p.is_file()}
 if listed!=actual:errors.append('Scalar3D asset allowlist differs')
 for entry in m['assets']:
  p=ASSETS/entry['path']
  if not p.is_file() or not p.resolve().is_relative_to(ASSETS.resolve()):errors.append('Missing/escaping Scalar3D asset: '+entry['path']);continue
  b=p.read_bytes()
  if len(b)!=entry['bytes'] or hashlib.sha256(b).hexdigest()!=entry['sha256']:errors.append('Scalar3D hash mismatch: '+entry['path'])
  if p.suffix=='.png':
   if b[:8]!=b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II',b[16:24]))!=entry['pixels']:errors.append('Scalar3D PNG shape: '+entry['path'])
  for locale in ('zh','en'):
   published=root/locale/'_static/scalar3d'/entry['path']
   if not published.is_file() or published.read_bytes()!=b:errors.append('Published Scalar3D asset differs: '+locale+'/'+entry['path'])
 report=(ASSETS/'downloads/REPORT.html').read_text()
 if re.search(r'<script|<iframe|\bon\w+\s*=|javascript:',report,re.I):errors.append('Active content in Scalar3D report')
 embedded=re.findall(r'<img[^>]+src="data:image/png;base64,([^\"]+)"',report)
 if len(embedded)!=21:errors.append('Scalar3D report must contain 21 figures')
 for i,data in enumerate(embedded):
  case=('enclosed','surface','layered')[i//7];fig=FIGURES[i%7]
  if base64.b64decode(data)!=(ASSETS/case/(fig+'.png')).read_bytes():errors.append('Report/PNG pixels differ: '+case+'/'+fig)
 package=json.loads((ASSETS/'downloads/package/package.json').read_text())
 digest=hashlib.sha256();parts=[]
 for item in package['parts']:
  p=ASSETS/'downloads/package'/item['file'];b=p.read_bytes()
  if len(b)>4*1024*1024 or len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:errors.append('Invalid package part: '+item['file'])
  parts.append(b);digest.update(b)
 if sum(map(len,parts))!=package['bytes'] or digest.hexdigest()!=package['sha256']:errors.append('Complete ZIP hash mismatch')
 with zipfile.ZipFile(io.BytesIO(b''.join(parts))) as archive:
  if archive.testzip():errors.append('Complete ZIP CRC mismatch')
  prefix='Scalar3D_Examples_20261008/'
  integrity=json.loads(archive.read(prefix+'PACKAGE_MANIFEST.json'))['files']
  for name,record in integrity.items():
   if name.startswith('/') or '..' in Path(name).parts:errors.append('Escaping package path');continue
   data=archive.read(prefix+name)
   if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:errors.append('Internal package manifest mismatch: '+name)
   if Path(name).suffix in {'.py','.txt','.json','.log','.html','.ipynb'}:
    text=data.decode('utf-8')
    if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}',text):errors.append('Private content in package: '+name)
  for case in CASES:
   for file in ['summary.json','numerical_checks.json','vp_true_zyx.npy','vp_initial_zyx.npy','vp_inverted_zyx.npy','observed.npy','survey.npz','gradients/gradient_manifest.json']:
    if prefix+'results/'+case+'/'+file not in archive.namelist():errors.append('Missing scientific package asset: '+case+'/'+file)
  if any('/example_support/' in name for name in archive.namelist()):errors.append('Unapproved private helper included')
 pdf=PdfReader(ASSETS/'downloads/Scalar3D-CUDA-Report.pdf')
 if len(pdf.pages)!=m['derived_pdf']['pages'] or sum(len(p.images) for p in pdf.pages)!=21:errors.append('Incomplete derived Scalar3D PDF')
 for p in pdf.pages:
  for annotation in p.get('/Annots',[]):
   a=annotation.get_object().get('/A')
   if a and a.get('/S')!='/URI':errors.append('Active Scalar3D PDF action')
 if any(k in pdf.trailer['/Root'] for k in ('/OpenAction','/AA','/AcroForm')):errors.append('Active Scalar3D PDF root')
 for locale in ('zh','en'):
  for name in PAGES:
   page=root/locale/'examples'/(name+'.html')
   if not page.is_file():errors.append('Missing Scalar3D page: '+locale+'/'+name);continue
   tree=HTMLTree(page.read_text()).root;main=tree.find('.//*[@role="main"]')
   nav=next((e for e in tree.iter('div') if 'wy-menu-vertical' in e.get('class','').split()),None)
   captions=[] if nav is None else [element_text(e) for e in nav.iter('span') if 'caption-text' in e.get('class','').split()]
   if captions.count('Example')!=1 or captions.index('Example')+1>=len(captions) or captions[captions.index('Example')+1]!='Usage':errors.append('Example must immediately precede Usage: '+locale+'/'+name)
   if main is None:errors.append('Missing example article');continue
   sections=list(main.iter('section'))
   styled=[section for section in sections if 'sw-example-page' in section.get('class','').split()]
   if len(styled)!=1 or styled[0] is not sections[0] or styled[0].find('h1') is None:errors.append('Example style is not attached to the article root: '+locale+'/'+name)
   toc=[nav for nav in main.iter('nav') if 'sw-page-toc' in nav.get('class','').split()]
   if not sections or sections[0].get('data-sw-section')!='sw-section-0' or len(toc)!=1 or len(list(toc[0].iter('a')))!=len(sections)-1:errors.append('Example native table of contents/anchors differ: '+locale+'/'+name)
   figs=[f for f in main.iter('figure') if 'sw-example-figure' in f.get('class','').split()]
   if name in CASES:
    if len(figs)!=7:errors.append('Missing seven Scalar3D figure types: '+locale+'/'+name)
    for f,key in zip(figs,FIGURES):
     imgs=list(f.iter('img'));caption=element_text(f.find('figcaption'))
     if len(imgs)!=1:errors.append('Invalid Scalar3D figure');continue
     img=imgs[0];expected=f'../_static/scalar3d/{name}/{key}.png'
     if img.get('src')!=expected:errors.append('Wrong Scalar3D figure source')
     if len(img.get('alt',''))<(25 if locale=='zh' else 60) or len(caption)<(35 if locale=='zh' else 80):errors.append('Missing localized Scalar3D explanation')
     view=f.find('div')
     if view is None or view.get('tabindex')!='0' or view.get('role')!='region':errors.append('Missing accessible Scalar3D scroll region')
     if not any(a.get('href')==expected for a in f.iter('a')):errors.append('Missing full resolution link')
   if name=='reproduce':
    buttons=[b for b in main.iter('button') if 'data-sw-package-download' in b.attrib]
    if len(buttons)!=1 or not buttons[0].get('data-manifest'):errors.append('Missing complete package button')
    downloads={a.get('href') for a in main.iter('a') if a.get('download')}
    if not {'../_static/scalar3d/downloads/REPORT.html','../_static/scalar3d/downloads/Scalar3D-CUDA-Report.pdf'}<=downloads:errors.append('Missing report downloads')
  for case in CASES:
   zh=HTMLTree((root/'zh/examples'/f'{case}.html').read_text()).root
   en=HTMLTree((root/'en/examples'/f'{case}.html').read_text()).root
   if len(zh.findall('.//section'))!=len(en.findall('.//section')):errors.append('Scalar3D bilingual heading parity')
 return errors

if __name__=='__main__':
 errors=check(sys.argv[1] if len(sys.argv)>1 else PROJECT/'_build/html')
 if errors:print('\n'.join(sorted(set(errors))));raise SystemExit(1)
 print('PASS: Scalar3D: 21 unchanged figures, report/PDF, strict asset allowlist, bilingual Example pages and navigation order.')
