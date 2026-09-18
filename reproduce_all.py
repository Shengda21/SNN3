"""Recompute both the original study and R2.1 from the supplied raw records."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
import pandas as pd

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figures',action='store_true')
    args=ap.parse_args()
    r2=ROOT/'revision_round2'
    manifest=read(r2/'results/model_analysis_lock.json')
    for name,expected in manifest['execution_and_analysis_code_sha256'].items():
        assert hashlib.sha256((r2/'code'/name).read_bytes()).hexdigest()==expected,name
    for relative,matched in manifest['original_code_identities_match'].items():
        assert matched
    sources=read(r2/'protocol/source_identities.json')
    external_sources_missing = []
    for source in sources:
        path=ROOT/'experiment'/source['path']
        if not path.exists() and '/vendor/SmoothSpike/' in source['path']:
            external_sources_missing.append(source['path'])
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'],source['path']

    subprocess.run([sys.executable,str(ROOT/'reproduce.py')]+(['--figures'] if args.figures else []),check=True)
    subprocess.run([sys.executable,str(ROOT/'derive_adaptation_recovery.py')],check=True)
    out=ROOT/'recomputed/revision_round2'
    subprocess.run([sys.executable,str(r2/'code/r2_analyze.py'),'--root',str(r2),'--output',str(out)],check=True)
    compared=[]
    for reference in sorted((r2/'analysis').glob('*.csv')):
        a=pd.read_csv(reference);b=pd.read_csv(out/reference.name)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-11,atol=1e-11)
        compared.append(reference.name)
    assert read(out/'verification.json')==read(r2/'analysis/verification.json')

    if args.figures:
        support=ROOT/'recomputed/support';data=support/'revision2_data';data.mkdir(exist_ok=True)
        for source in out.glob('*.csv'):shutil.copy2(source,data/source.name)
        shutil.copy2(out/'verification.json',data/'verification.json')
        shutil.copy2(ROOT/'figure_source/revision2_panels.py',support/'revision2_panels.py')
        for name in ['make_revision2_figures.py','make_workflow_revision2.py']:
            shutil.copy2(ROOT/'figure_source'/name,support/name)
            subprocess.run([sys.executable,str(support/name)],check=True)
        for name in ['paper_quality_summary.csv','paper_quality_latency.csv',
                     'bn_metric_plane.csv','timing_quality_process_points.csv']:
            expected=pd.read_csv(ROOT/'figure_source/published_revision2_data'/name)
            actual=pd.read_csv(data/name)
            pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-11,atol=1e-11)

    report={'status':'passed','original':read(ROOT/'recomputed/verification.json'),
        'revision_round2':read(out/'verification.json'),'revision_tables_compared':compared,
        'included_execution_code_identities_match':True,
        'external_sources_not_distributed':external_sources_missing,
        'figures_regenerated':args.figures,
        'scope':'offline analysis from raw scores and calls; no model execution or network access'}
    (ROOT/'recomputed/combined_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
