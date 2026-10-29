"""Offline API checks. Synthetic inputs test code paths, NOT model accuracy."""
from pathlib import Path
import sys,json,ast,copy,gc
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tutorial_support.common import *
import torch
from torch import nn
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
DEVICE=torch_setup();keras=keras_setup()

def definitions(filename,names):
    book=json.loads((root/'notebooks'/filename).read_text(encoding='utf-8'))
    for cell in book['cells']:
        if cell['cell_type']!='code':continue
        tree=ast.parse(''.join(cell['source']) if isinstance(cell['source'],list) else cell['source'])
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),globals())

def check_loss(losses):
    loss=sum(losses.values());assert torch.isfinite(loss)
    loss.backward();print({k:float(v.detach()) for k,v in losses.items()})

def main():
    definitions('05_CNN_CIFAR10.ipynb',{'build_cnn'})
    for kw in [{},{'kernel':5},{'filters':(32,64,128)},{'dense':128},{'regularized':True}]:
        m=build_cnn(**kw);result=m.train_on_batch(np.random.rand(2,32,32,3).astype('float32'),np.array([0,1]));assert np.isfinite(result).all()
    print('PASS CNN five configurations and gradient steps');del m
    definitions('07_Transfer_Learning.ipynb',{'TransferNet','CustomCNN','epoch','fit'})
    # Disable external weight downloads only in this offline verification process.
    for name in ['vgg16','resnet50']:
        original=getattr(models,name)
        setattr(models,name,lambda weights=None,_f=original,**kw:_f(weights=None,**kw))
    for name in ['VGG16','ResNet50']:
        m=TransferNet(name).to(DEVICE);x=torch.rand(2,3,64,64)
        before=sum(p.numel() for p in m.parameters() if p.requires_grad)
        loss=nn.functional.cross_entropy(m(x),torch.tensor([0,1]));loss.backward()
        m.zero_grad();m.unfreeze_last_block();m.train()
        assert sum(p.numel() for p in m.parameters() if p.requires_grad)>before
        assert all(not b.training for b in m.base.modules() if isinstance(b,nn.BatchNorm2d))
        loss=nn.functional.cross_entropy(m(x),torch.tensor([0,1]));loss.backward();assert torch.isfinite(loss)
        print('PASS transfer frozen and fine-tuned',name);del m;gc.collect()
    definitions('08A_Custom_Faster_RCNN.ipynb',{'make_detector'})
    targets=[{'boxes':torch.tensor([[10.,10.,70.,100.]]),'labels':torch.tensor([1])} for _ in range(2)]
    images=[torch.rand(3,128,128) for _ in range(2)]
    for width in [32,64]:
        m=make_detector(width).train();check_loss(m(images,targets));m.eval()
        with torch.no_grad():p=m(images)
        assert all({'boxes','scores','labels'}<=set(x) for x in p)
        print('PASS custom Faster R-CNN',width);del m;gc.collect()
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric=MeanAveragePrecision();gt=[{'boxes':torch.tensor([[0.,0.,10.,10.]]),'labels':torch.tensor([1])}]
    metric.update([{**gt[0],'scores':torch.tensor([.9])}],gt);assert abs(float(metric.compute()['map'])-1)<1e-6
    print('PASS AP evaluator exact-match example')
    definitions('09_SSD_YOLO.ipynb',{'TinySSD','TinyYOLO'})
    assert TinyYOLO()(torch.rand(2,3,64,64)).shape==(2,75,13,13)
    for cls,box in TinySSD()(torch.rand(2,3,64,64)):assert cls.shape[1]==9 and box.shape[1]==12
    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.ssd import SSDClassificationHead
    m=ssd300_vgg16(weights=None,weights_backbone=None)
    with torch.no_grad():features=m.backbone(torch.zeros(1,3,300,300))
    m.head.classification_head=SSDClassificationHead([v.shape[1] for v in features.values()],m.anchor_generator.num_anchors_per_location(),2)
    m.train();check_loss(m(images,targets));print('PASS SSD replaced head and training loss')
    (OUT/'offline_smoke.json').write_text(json.dumps({'status':'passed','scope':'CNN training variants, VGG/ResNet freeze/unfreeze gradients, custom Faster R-CNN losses/inference, exact-match AP, toy SSD/YOLO shapes, real SSD head/backward','data':'synthetic','pretrained_weights':False},indent=2))
if __name__=='__main__':main()

