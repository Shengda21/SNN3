"""Recompute R3 from per-item records without loading models or using the network."""
from pathlib import Path
import argparse,importlib.util,json,os,shutil,subprocess,sys
import pandas as pd
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=ROOT/'recomputed/revision_round3')
 p.add_argument('--figures',action='store_true')
 a=p.parse_args();work=ROOT/'revision_round3';out=a.output.resolve()
 assert out!=work/'analysis','Keep recorded summaries unchanged.'
 out.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('r3_analysis',work/'code/r3_analyze.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.WORK=work;mod.OUT=out;mod.main()
 compared=[]
 for f in sorted((work/'analysis').glob('*.csv')):
  g=out/f.name
  if not g.exists():continue
  pd.testing.assert_frame_equal(pd.read_csv(f),pd.read_csv(g),check_dtype=False,rtol=1e-11,atol=1e-11)
  compared.append(f.name)
 assert len(compared)==11,compared
 assert json.loads((out/'summary.json').read_text())==json.loads((work/'analysis/summary.json').read_text())
 if a.figures:
  tree=out/'figure_rebuild';support=tree/'support';support.mkdir(parents=True,exist_ok=True)
  shutil.copytree(ROOT/'figure_source/published_revision3_data',support/'revision3_data',dirs_exist_ok=True)
  shutil.copytree(ROOT/'figure_source/published_revision2_data',support/'revision2_data',dirs_exist_ok=True)
  for name in ['contrasts_by_seed.csv','contrast_intervals.csv','bn_transfer_penalty_by_seed.csv','bn_transfer_penalty_intervals.csv','summary.json']:
   shutil.copy2(out/name,support/'revision3_data'/name)
  shutil.copy2(ROOT/'figure_source/make_revision3_figures.py',support/'make_revision3_figures.py')
  subprocess.run([sys.executable,str(support/'make_revision3_figures.py')],check=True)
 report={'status':'passed','scoring_paths':506,'csv_compared':compared,'figures_regenerated':a.figures,
         'scope':'Per-item offline recomputation; no GPU execution or new model selection.'}
 (out/'offline_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
