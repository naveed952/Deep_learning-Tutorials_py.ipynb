"""Public-data loading, bounding boxes and evaluation shared by detection tutorials."""
import shutil, zipfile, urllib.request
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import to_tensor
from tutorial_support.common import DATA, OUT, QUICK, SEED

def penn_root():
    root=DATA/'PennFudanPed'
    if not (root/'PNGImages').exists():
        archive=DATA/'PennFudanPed.zip'
        if not (archive.exists() and zipfile.is_zipfile(archive)):
            url='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
            temporary=archive.with_suffix('.download')
            try:
                with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as target:
                    shutil.copyfileobj(response,target)
                if not zipfile.is_zipfile(temporary):raise ValueError('Downloaded file is not a ZIP archive')
                temporary.replace(archive)
            except Exception as error:
                raise RuntimeError('Dataset download failed. Download PennFudanPed.zip from '+url+' with your browser and place it in '+str(DATA)+', then rerun. Certificate checking remains enabled.') from error
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                dest=(DATA/item.filename).resolve()
                if not dest.is_relative_to(DATA.resolve()):raise ValueError('Unsafe archive path')
            z.extractall(DATA)
    return root

class PennFudan(Dataset):
    def __init__(self):
        self.root=penn_root();self.images=sorted((self.root/'PNGImages').glob('*.png'))
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        path=self.images[i];image=Image.open(path).convert('RGB')
        mask=np.asarray(Image.open(self.root/'PedMasks'/(path.stem+'_mask.png')))
        boxes=[]
        for label in np.unique(mask):
            if label==0:continue
            ys,xs=np.where(mask==label)
            boxes.append([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        target={'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64),'image_id':torch.tensor(i)}
        return to_tensor(image),target

def split_ids(dataset):
    ids=np.random.default_rng(SEED).permutation(len(dataset)).tolist()
    # Same fixed split for all detectors. Quick uses disjoint subsets of it.
    train,val,test=ids[:150],ids[150:160],ids[160:]
    return (train[:16],val[:4],test[:4]) if QUICK else (train,val,test)

def collate(batch):return tuple(zip(*batch))
def loaders(dataset):
    return tuple(DataLoader(Subset(dataset,ids),batch_size=2,shuffle=(i==0),num_workers=0,collate_fn=collate) for i,ids in enumerate(split_ids(dataset)))

def train_detector(model,loader,device,epochs):
    model.to(device);optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    history=[]
    for epoch in range(epochs):
        model.train();totals={};count=0
        for images,targets in loader:
            images=[im.to(device) for im in images];targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
            losses=model(images,targets);loss=sum(losses.values())
            if not torch.isfinite(loss):raise ValueError('Non-finite detection loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            for k,v in losses.items():totals[k]=totals.get(k,0)+float(v.detach())
            count+=1
        row={k:v/count for k,v in totals.items()};history.append(row);print('Epoch',epoch+1,row)
    return history

@torch.no_grad()
def evaluate_detector(model,loader,device,person_only=False):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric=MeanAveragePrecision(box_format='xyxy',iou_type='bbox',class_metrics=True)
    model.to(device).eval()
    for images,targets in loader:
        outputs=model([im.to(device) for im in images]);predictions=[]
        for out in outputs:
            keep=(out['labels']==1) if person_only else torch.ones_like(out['labels'],dtype=torch.bool)
            predictions.append({k:out[k][keep].detach().cpu() for k in ('boxes','scores','labels')})
        metric.update(predictions,[{k:v.cpu() for k,v in t.items() if k in ('boxes','labels')} for t in targets])
    result=metric.compute()
    return {k:float(result[k]) for k in ['map','map_50','map_75','mar_100']}

def show_prediction(image,pred,threshold=.5):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(8,5));ax.imshow(image.permute(1,2,0).cpu())
    for box,score,label in zip(pred['boxes'],pred['scores'],pred['labels']):
        if float(score)<threshold:continue
        x1,y1,x2,y2=box.detach().cpu().tolist();ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='red'))
        ax.text(x1,y1,f'{int(label)} {float(score):.2f}',color='white',bbox={'facecolor':'black','alpha':.6})
    ax.axis('off');plt.show()

def export_yolo(dataset):
    import yaml
    root=DATA/'penn_yolo';train,val,test=split_ids(dataset)
    for name,ids in [('train',train),('val',val),('test',test)]:
        (root/'images'/name).mkdir(parents=True,exist_ok=True);(root/'labels'/name).mkdir(parents=True,exist_ok=True)
        for i in ids:
            image,target=dataset[i];h,w=image.shape[1:];src=dataset.images[i]
            shutil.copy2(src,root/'images'/name/src.name)
            lines=[]
            for x1,y1,x2,y2 in target['boxes'].tolist():
                lines.append(f'0 {(x1+x2)/(2*w)} {(y1+y2)/(2*h)} {(x2-x1)/w} {(y2-y1)/h}')
            (root/'labels'/name/(src.stem+'.txt')).write_text('\n'.join(lines))
        # Explicit manifest avoids accidentally including stale files from a previous full run.
        (root/(name+'.txt')).write_text('\n'.join(str((root/'images'/name/dataset.images[i].name).resolve()) for i in ids))
    config={'path':str(root.resolve()),'train':str((root/'train.txt').resolve()),'val':str((root/'val.txt').resolve()),'test':str((root/'test.txt').resolve()),'names':{0:'person'}}
    path=root/'dataset.yaml';path.write_text(yaml.safe_dump(config));return path
