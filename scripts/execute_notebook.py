"""Execute one notebook in a real Jupyter kernel and retain only real outputs."""
import argparse,os,time,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import sys
parser=argparse.ArgumentParser();parser.add_argument('notebook');parser.add_argument('--timeout',type=int,default=1800);args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
# Private workspace kernel spec uses the interpreter running this script.
kdir=root/'artifacts'/'kernels'/'tutorials';kdir.mkdir(parents=True,exist_ok=True)
(kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Tutorial runtime','language':'python'}))
ksm=KernelSpecManager(kernel_dirs=[str(kdir.parent)])
path=(root/args.notebook).resolve();book=nbformat.read(path,as_version=4);start=time.time()
# Never retain stale results from a previous run after a later-cell failure.
for cell in book.cells:
    if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
client=NotebookClient(book,timeout=args.timeout,kernel_name='tutorials',resources={'metadata':{'path':str(root)}},allow_errors=False)
from jupyter_client import KernelManager
client.km=KernelManager(kernel_name='tutorials',kernel_spec_manager=ksm)
status='passed'
try:client.execute()
except BaseException as error:
    status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
    raise
finally:
    if status=='passed':
        book.cells=[cell for cell in book.cells if not (cell.cell_type=='markdown' and cell.source.startswith('**Execution status:**'))]
    nbformat.write(book,path)
    (root/'artifacts'/(path.stem+'_execution.json')).write_text(json.dumps({'notebook':path.name,'status':status,'seconds':time.time()-start,'full':os.environ.get('TUTORIAL_FULL')=='1'},indent=2))
print('PASS',path.name,round(time.time()-start,1),'seconds')
