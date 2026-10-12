#!/usr/bin/env python3
"""AST-only public API inventory/checker. No imports/build/GPU needed.
Usage: python check_v19_api_contracts.py SOURCE [--write JSON] [--check JSON] [--docs DOCS]
Annotations are recorded but doc signature checking compares names/kinds/defaults only.
"""
import argparse, ast, json, pathlib, re
MAP = {
 'starwave.scalar':('scalar.py','scalar'), 'starwave.vrz':('vrz.py','vrz'),
 'starwave.vrr':('_vrr/api.py','vrr'), 'starwave.vrr_reference':('_vrr/api.py','vrr_reference'),
 'starwave.vti':('vti.py','vti'), 'starwave.elastic':('elastic.py','elastic'),
 'starwave.scalar_born':('scalar_born.py','scalar_born'),
 'starwave.ScalarBorn':('scalar_born.py','ScalarBorn.__init__'),
 'starwave.ScalarBorn.forward':('scalar_born.py','ScalarBorn.forward'),
 'starwave.visco_gsls':('visco_gsls.py','visco_gsls'),
 'starwave.prepare_visco_gsls':('_visco_gsls_native.py','prepare'),
 'starwave.visco_gsls_native_status':('_visco_gsls_native.py','native_status'),
 'starwave.native_status':('_native.py','native_status'),
 'starwave.prepare_native':('_native.py','prepare_native'),
 'starwave.elastic_native_status':('elastic_runtime.py','elastic_native_status'),
 'starwave.prepare_elastic':('elastic_runtime.py','prepare_elastic'),
 'starwave.ScalarIllumination':('illumination.py','_EquationIllumination.__init__'),
 'starwave.ScalarIllumination.seal':('illumination.py','_EquationIllumination.seal'),
 'starwave.ScalarIllumination.reduce':('illumination.py','_EquationIllumination.reduce'),
 'starwave.ScalarIllumination.close':('illumination.py','_EquationIllumination.close'),
 'starwave.precondition_gradient':('illumination.py','precondition_gradient'),
 'starwave.common.vpvsrho_to_lambmubuoyancy':('common.py','vpvsrho_to_lambmubuoyancy'),
 'starwave.common.lambmubuoyancy_to_vpvsrho':('common.py','lambmubuoyancy_to_vpvsrho'),
 'starwave.wavelets.ricker':('_deepwave/wavelets.py','ricker'),
 'starwave.torch_backend.backend_capabilities':('torch_backend/_backends.py','backend_capabilities'),
}
for name in ('scalar','vrz','vti','elastic','scalar_born','visco_gsls'):
 MAP['starwave.torch_backend.'+name]=('torch_backend/'+name+'.py',name)
def unpack(n):
 a=n.args; vals=[]; pos=a.posonlyargs+a.args; defaults=[None]*(len(pos)-len(a.defaults))+a.defaults
 for i,(arg,d) in enumerate(zip(pos,defaults)):
  if arg.arg=='self': continue
  vals.append(dict(name=arg.arg,kind='POSITIONAL_ONLY' if i<len(a.posonlyargs) else 'POSITIONAL_OR_KEYWORD',default=ast.unparse(d) if d is not None else '<required>',annotation=ast.unparse(arg.annotation) if arg.annotation else None))
 if a.vararg: vals.append(dict(name=a.vararg.arg,kind='VAR_POSITIONAL',default='<required>',annotation=None))
 for arg,d in zip(a.kwonlyargs,a.kw_defaults): vals.append(dict(name=arg.arg,kind='KEYWORD_ONLY',default=ast.unparse(d) if d is not None else '<required>',annotation=ast.unparse(arg.annotation) if arg.annotation else None))
 if a.kwarg: vals.append(dict(name=a.kwarg.arg,kind='VAR_KEYWORD',default='<required>',annotation=None))
 return vals
def locate(tree,path):
 for part in path.split('.'):
  tree=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part)
 return tree
def inventory(root):
 base=root/'starwave'; result={}
 for public,(file,path) in MAP.items():
  n=locate(ast.parse((base/file).read_text()),path)
  result[public]=dict(source='starwave/'+file,line=n.lineno,parameters=unpack(n),return_annotation=ast.unparse(n.returns) if n.returns else None)
 init=ast.parse((base/'__init__.py').read_text()); exports=next(ast.literal_eval(n.value) for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__all__' for t in n.targets))
 fields=locate(ast.parse((base/'illumination.py').read_text()),'IlluminationFields')
 return dict(schema=1,source_commit='46d32877e3336d531520a696d5f2e074e6f4ef74',scope='AST signatures and declared exports; runtime semantics in Usage; not GPU acceptance',exports=exports,additional_direct_attributes=['common','wavelets','IGNORE_LOCATION'],dataclass_fields={'starwave.IlluminationFields':[{ 'name':n.target.id,'annotation':ast.unparse(n.annotation)} for n in fields.body if isinstance(n,ast.AnnAssign)]},apis=result)
def normalized(params): return [{k:p[k] for k in ('name','kind','default')} for p in params]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('source',type=pathlib.Path,nargs='?');ap.add_argument('--write',type=pathlib.Path);ap.add_argument('--check',type=pathlib.Path);ap.add_argument('--docs',type=pathlib.Path);a=ap.parse_args(); data=inventory(a.source) if a.source else json.loads(a.check.read_text()); errors=[]
 if a.write:a.write.write_text(json.dumps(data,indent=2)+'\n')
 if a.check:
  old=json.loads(a.check.read_text())
  for name,api in old['apis'].items():
   if normalized(api['parameters'])!=normalized(data['apis'][name]['parameters']):errors.append('SOURCE CONTRACT CHANGED '+name)
  if old['exports']!=data['exports']:errors.append('SOURCE EXPORTS CHANGED')
 if a.docs:
  for f in [a.docs/'usage.md',a.docs/'en/usage.md']:
   if not f.exists():continue
   seen=set()
   for kind,name,sig in re.findall(r'```\{py:(function|class|method)\}\s+([\w.]+)(\([^\n]+)',f.read_text()):
    if name not in data['apis']:
     if name.startswith('starwave_vrr.'):errors.append(str(f)+': obsolete namespace '+name)
     continue
    seen.add(name)
    body=re.search(r'```\{py:'+kind+r'\} '+re.escape(name)+r'[^\n]*\n(.*?)^```',f.read_text(),re.M|re.S).group(1)
    fields=re.findall(r'^:param (\w+): (.+)$',body,re.M)
    if [n for n,_ in fields]!=[p['name'] for p in data['apis'][name]['parameters']]:errors.append(str(f)+': parameter prose missing '+name)
    try: node=ast.parse('def temp'+sig+':\n pass').body[0]
    except SyntaxError as e:errors.append(str(f)+': unparsable '+name+': '+str(e));continue
    if normalized(unpack(node))!=normalized(data['apis'][name]['parameters']):errors.append(str(f)+': signature mismatch '+name)
   for name in [n for n in data['apis'] if not n.startswith('starwave.torch_backend.') or n.endswith('backend_capabilities')]:
    if name not in seen:errors.append(str(f)+': missing public signature '+name)
 print(json.dumps(dict(api_count=len(data['apis']),errors=errors),indent=2))
 raise SystemExit(bool(errors))
if __name__=='__main__':main()
