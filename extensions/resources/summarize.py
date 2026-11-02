"""Summarize actual per-board receipts while retaining invalid energy observations."""
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for path in sorted(a.receipts.rglob('receipt.json')):
  r=json.loads(path.read_bytes())
  if 'process_wall_seconds' not in r:continue
  rows.append({'stage':r['stage']['id'],'status':r['status'],'returncode':r['returncode'],'wall_seconds':r['process_wall_seconds'],'board_energy_J':r['board_energy_J'],'energy_valid':r['energy_valid'],'max_sample_gap_seconds':r['max_sample_gap_seconds'],'device_peak_memory_MiB':r['device_peak_memory_MiB'],'receipt':str(path.relative_to(a.receipts))})
 assert rows
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'stages':len(rows),'energy_valid_stages':sum(r['energy_valid'] for r in rows),'energy_invalid_stages_preserved':sum(not r['energy_valid'] for r in rows),'valid_energy_total_J':sum(r['board_energy_J'] for r in rows if r['energy_valid']),'wall_seconds_total':sum(r['wall_seconds'] for r in rows),'scope':'board power including idle during process; not whole-system or isolated-host power'},indent=2))
if __name__=='__main__':main()
