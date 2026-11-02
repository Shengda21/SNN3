"""Explicit dual-GPU execution of the frozen revision-2 dependency graph."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from common import BUNDLE, environment, exclusive, read, require_frozen, sha, verify_prepared, write
import driver as d


class Pool:
    def __init__(self, root, devices):
        self.root, self.devices = Path(root), list(devices)
        for name in ('state','logs'):
            (self.root/name).mkdir(parents=True,exist_ok=True)
        self.pending, self.running, self.seen = [], {}, set()
        self.records, self.failures, self.maximum_active = {}, [], 0

    def add(self, name, command, finished, callback=lambda: None, preferred=None):
        if name in self.seen:
            return
        self.seen.add(name)
        self.pending.append({'name':name,'command':list(map(str,command)),
                             'finished':finished,'callback':callback,'preferred':preferred})

    def save(self, status='running'):
        write(self.root/'state/parallel_status.json',
              {'status':status,'updated':time.time(),'coordinator_pid':os.getpid(),
               'devices':self.devices,'queued':[task['name'] for task in self.pending],
               'running':{str(gpu):{'name':task['name'],'pid':task['process'].pid}
                          for gpu,task in self.running.items()},
               'tasks':self.records,'failures':self.failures,'maximum_active':self.maximum_active})

    def run(self):
        while self.pending or self.running:
            changed = False
            for gpu,task in list(self.running.items()):
                code = task['process'].poll()
                if code is None:
                    continue
                del self.running[gpu]
                task['log'].close()
                good = code == 0 and task['finished']()
                self.records[task['name']].update(finished=time.time(),returncode=code,
                                                   status='complete' if good else 'failed')
                if good:
                    task['callback']()
                else:
                    self.failures.append(task['name'])
                changed = True
            for task in list(self.pending):
                if task['finished']():
                    self.pending.remove(task)
                    self.records[task['name']]={'status':'retained_complete','finished':time.time()}
                    task['callback']()
                    changed = True
            for gpu in self.devices:
                if gpu in self.running:
                    continue
                task = next((task for task in self.pending
                             if task['preferred'] in (None,gpu)),None)
                if task is None:
                    continue
                stream = (self.root/'state'/f'physical_gpu_{gpu}.lock').open('a')
                try:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                if shutil.disk_usage(self.root).free/2**30 < 10:
                    stream.close()
                    raise RuntimeError('Less than 10 GiB free; existing results preserved')
                self.pending.remove(task)
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                           PYTHONDONTWRITEBYTECODE='1',REVISION2_COORDINATOR_PID=str(os.getpid()))
                log = (self.root/'logs'/f"parallel_{task['name']}.log").open('a')
                try:
                    process = subprocess.Popen(task['command'],env=env,stdout=log,
                        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        pass_fds=(stream.fileno(),),start_new_session=True)
                except BaseException:
                    log.close()
                    stream.close()
                    raise
                # The child inherits the actual locked descriptor until it exits.
                stream.close()
                task.update(process=process,log=log)
                self.running[gpu] = task
                self.records[task['name']]={'status':'running','physical_gpu':gpu,
                    'pid':process.pid,'started':time.time(),'command':task['command']}
                changed = True
            self.maximum_active = max(self.maximum_active,len(self.running))
            if changed:
                self.save()
            if self.pending or self.running:
                time.sleep(1)
        self.save('failed' if self.failures else 'complete')
        return self.failures


def complete(path):
    return path.is_file() and read(path).get('status') == 'complete'


def phase(root, name, cohort):
    root = verify_prepared(root)
    environment(root)
    require_frozen(root)
    assert os.getppid() == int(os.environ['REVISION2_COORDINATOR_PID'])
    assert read(root/'environment/model_smoke.json')['status'] == 'passed'
    if name == 'select':
        d.select(root,cohort)
    elif name == 'E3':
        from e3 import evaluate_sources
        evaluate_sources(root)
    elif name == 'timing':
        from measure import run
        run(root,d.command)
    else:
        raise ValueError(name)
    write(root/'state'/f'parallel_phase_{name}_{cohort or "all"}.json',{'status':'complete'})


def export(root):
    rows = read(root/'protocol/scoring_queue.json')
    for row in rows:
        path = root/'results/scoring'/row['id']
        assert complete(path/'complete.json') and (path/'test.npz').is_file()
        assert read(path/'complete.json')['row'] == row
    output = root/'export'
    output.mkdir(exist_ok=True)
    weights = []
    for folder in ('construction','models','bn_banks'):
        for path in sorted((root/folder).rglob('*.pt')):
            if any(part.startswith('SOFTWARE_') for part in path.parts):
                continue
            weights.append({'path':path.relative_to(root).as_posix(),
                            'bytes':path.stat().st_size,'sha256':sha(path)})
    with (output/'checkpoint_inventory.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['path','bytes','sha256'])
        writer.writeheader()
        writer.writerows(weights)
    report = {'status':'compute_complete_pending_verified_download',
              'scoring_rows':len(rows),'checkpoint_files':len(weights),
              'checkpoint_bytes':sum(row['bytes'] for row in weights),
              'local_git_commit':read(root/'environment/protocol_freeze.json')['local_git_commit'],
              'server_shutdown_executed':False,'full_outputs_download_verified':False,
              'public_release_claimed':False}
    write(output/'READY_FOR_DOWNLOAD.json',report)
    with zipfile.ZipFile(output/'reports_without_full_weights.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('environment','state','protocol','logs','construction','training',
                       'results','selection_eval','analysis','timing','bn_banks'):
            for path in sorted((root/folder).rglob('*')):
                relative = path.relative_to(root)
                if not path.is_file() or any(part.startswith('SOFTWARE_') for part in relative.parts):
                    continue
                if path.suffix in ('.lock','.partial') or (path.suffix=='.pt' and folder!='bn_banks'):
                    continue
                archive.write(path,relative.as_posix())
        for path in (output/'checkpoint_inventory.csv',output/'READY_FOR_DOWNLOAD.json'):
            archive.write(path,path.name)
    write(root/'state/COMPUTE_COMPLETE.json',report)


def execute(root, devices):
    root = verify_prepared(root)
    environment(root)
    require_frozen(root)
    assert read(root/'environment/model_smoke.json')['status'] == 'passed'
    assert (root/'protocol/a_mtt_screen_sources.json').is_file()
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version',
                                        '--format=csv,noheader'],text=True)
    write(root/'environment/dual_gpu_execution.json',{'devices':devices,'hardware':hardware,
        'scope':'independent whole trajectories; unchanged per-job batches, seeds, losses and endpoints',
        'timing':'GPU 0 in isolation after all quality work; no speed claim from mixed devices'})
    pool = Pool(root,devices)
    cohort_done, A_done = set(), [False]
    advanced = {'score':False,'timing':False}
    def cmd(script,*args):
        return [sys.executable,str(BUNDLE/'scripts'/script),'--root',str(root),*map(str,args)]
    def finish_timing():
        d.command(root,'cohort_analysis','cohort_analysis.py')
        d.command(root,'quality_cost_view','cost_view.py')
        export(root)
    def score_all():
        if advanced['score'] or cohort_done!={'C-conf','F-arch'} or not A_done[0]:
            return
        advanced['score'] = True
        d.scoring_queue(root)
        queue = root/'protocol/scoring_queue.json'
        rows = read(queue)
        sources = {}
        for row in rows:
            sources.setdefault(row['source_id'],[]).append(row)
        groups,loads = [[] for _ in devices],[0 for _ in devices]
        for source,items in sorted(sources.items(),key=lambda item:(-len(item[1]),item[0])):
            index = min(range(len(devices)),key=lambda index:(loads[index],index))
            groups[index].append(source)
            loads[index] += len(items)
        done_groups = set()
        def group_done(index):
            done_groups.add(index)
            if len(done_groups)==len(devices) and not advanced['timing']:
                advanced['timing'] = True
                marker = root/'state/parallel_phase_timing_all.json'
                pool.add('timing',cmd('parallel_run.py','--phase','timing'),
                         lambda:complete(marker),finish_timing,preferred=devices[0])
        for index,source_ids in enumerate(groups):
            group = root/'protocol'/f'score_source_group_{index}.json'
            d.save_locked(group,{'queue_sha256':sha(queue),'source_ids':source_ids,
                                'partition_rule':'greedy route count, stable source ID; no test outcome used'})
            group_rows = [row for row in rows if row['source_id'] in set(source_ids)]
            def ready(items=group_rows):
                return all(complete(root/'results/scoring'/row['id']/'complete.json')
                           and (root/'results/scoring'/row['id']/'test.npz').is_file() for row in items)
            pool.add(f'score_group_{index}',cmd('score.py','--source-group',group),ready,
                     lambda index=index:group_done(index),preferred=devices[index])
    def enqueue_jobs(jobs, callback):
        remaining = {job['id'] for job in jobs}
        if not remaining:
            callback()
            return
        def one_done(name):
            remaining.remove(name)
            if not remaining:
                callback()
        for job in jobs:
            path = root/'protocol/jobs'/f"{job['id']}.json"
            d.save_locked(path,job)
            marker = root/'training'/job['id']/'complete.json'
            def ready(marker=marker,job=job):
                return complete(marker) and read(marker)['identity']['job']==job and read(marker)['step']==2048
            pool.add(job['id'],cmd('train_job.py','--job',path),ready,
                     lambda name=job['id']:one_done(name))
    def formal_ready(cohort):
        jobs = read(root/'protocol'/f'{cohort}_formal_jobs.json')
        def finished():
            cohort_done.add(cohort)
            score_all()
        enqueue_jobs(jobs,finished)
    def screens_ready(cohort):
        marker = root/'state'/f'parallel_phase_select_{cohort}.json'
        pool.add(f'select_{cohort}',cmd('parallel_run.py','--phase','select','--cohort',cohort),
                 lambda:complete(marker),lambda:formal_ready(cohort))
    def constructed(cohort):
        jobs = d.screen_jobs(root,cohort)
        d.save_locked(root/'protocol'/f'{cohort}_screen_jobs.json',jobs)
        enqueue_jobs(jobs,lambda:screens_ready(cohort))
    for cohort in ('C-conf','F-arch'):
        marker = root/'construction'/cohort/'complete.json'
        pool.add('construct_'+cohort,cmd('train_job.py','--construct',cohort),
                 lambda marker=marker:complete(marker),lambda cohort=cohort:constructed(cohort))
    def E3_ready():
        selection = read(root/'results/e3/A-conf-BN-selection/selection.json')
        rows = read(root/'results/e3/A-conf-BN-selection/conditional_formal_tasks.json')
        assert bool(rows)==selection['conditional_retrain_needed']
        jobs = [{**row,'architecture':'resformer','T':None,'stage':'formal','roles':['mtt_joint'],
                 'policy_id':'A-conf-BN-selection'} for row in rows]
        if jobs:
            d.save_locked(root/'protocol/A-conf-BN-selection_formal_jobs.json',jobs)
        def finished():
            A_done[0] = True
            score_all()
        enqueue_jobs(jobs,finished)
    marker = root/'state/parallel_phase_E3_all.json'
    pool.add('A_E3',cmd('parallel_run.py','--phase','E3'),lambda:complete(marker),E3_ready)
    failures = pool.run()
    if failures:
        raise RuntimeError('Recorded failed tasks; no outcome-driven retries: '+','.join(failures))
    assert (root/'state/COMPUTE_COMPLETE.json').is_file()


def self_test():
    with tempfile.TemporaryDirectory(prefix='revision2_dispatch_') as temporary:
        root = Path(temporary)
        pool = Pool(root,['0','1'])
        retained = root/'retained.json'
        retained.write_text('{"status":"complete"}')
        pool.add('retained',[sys.executable,'-c','raise RuntimeError("must not repeat")'],lambda:complete(retained))
        for index in range(2):
            path = root/f'job{index}.json'
            code = 'import time,pathlib; time.sleep(2); pathlib.Path('+repr(str(path))+').write_text(\'{"status":"complete"}\')'
            pool.add(f'job{index}',[sys.executable,'-c',code],lambda path=path:complete(path))
        dependent = root/'dependent.json'
        def after():
            assert complete(root/'job0.json') and complete(root/'job1.json')
        assert not pool.run()
        after()
        assert pool.maximum_active==2 and pool.records['retained']['status']=='retained_complete'
        assert {row['physical_gpu'] for row in pool.records.values() if 'physical_gpu' in row}=={'0','1'}
        broken = Pool(root,['0','1'])
        broken.add('failure',[sys.executable,'-c','raise SystemExit(7)'],lambda:False,
                   lambda:dependent.write_text('incorrect'))
        assert broken.run()==['failure'] and not dependent.exists()
    print(json.dumps({'status':'passed','independent_GPU_slots':2,'completed_work_skipped':True,
                      'nonzero_failure_propagated_without_dependent_work':True,'software_only':True}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--phase',choices=['select','E3','timing'])
    parser.add_argument('--cohort',choices=['C-conf','F-arch'])
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.phase:
        phase(args.root,args.phase,args.cohort)
    else:
        assert args.root is not None
        devices = args.gpus.split(',')
        assert devices and all(device.isdigit() for device in devices) and len(devices)==len(set(devices))
        with exclusive(verify_prepared(args.root)):
            execute(args.root,devices)
