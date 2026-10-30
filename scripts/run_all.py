"""Run selected notebooks independently, retain real outputs and report every failure."""
import argparse,os,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('notebooks',nargs='*');p.add_argument('--full',action='store_true');p.add_argument('--timeout',type=int,default=1800);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
paths=[root/name for name in args.notebooks] if args.notebooks else sorted((root/'notebooks').glob('*.ipynb'))
env=os.environ.copy();env['TUTORIAL_FULL']='1' if args.full else '0'
report=[]
for path in paths:
    print('\nRUN',path.name,flush=True)
    result=subprocess.run([sys.executable,str(root/'scripts/execute_notebook.py'),str(path.resolve()),'--timeout',str(args.timeout)],cwd=root,env=env)
    report.append({'notebook':path.name,'exit_code':result.returncode,'mode':'full' if args.full else 'quick'})
out=root/'artifacts'/'batch_execution.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
raise SystemExit(1 if any(row['exit_code'] for row in report) else 0)
