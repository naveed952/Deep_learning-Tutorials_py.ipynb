"""Small shared utilities; model and training explanations live in each notebook."""
from pathlib import Path
import os, random, json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'; OUT = ROOT/'artifacts'
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
QUICK = os.environ.get('TUTORIAL_FULL', '0') != '1'
SEED = 42

def seed():
    random.seed(SEED); np.random.seed(SEED)

def save_result(name, result):
    path = OUT/(name+'.json')
    path.write_text(json.dumps(result, indent=2, default=str))
    return path

def curves(histories, metric='loss'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,3))
    for name,h in histories.items():
        h = h.history if hasattr(h,'history') else h
        ax.plot(range(1,len(h[metric])+1),h[metric],label=name+' train')
        if 'val_'+metric in h: ax.plot(range(1,len(h['val_'+metric])+1),h['val_'+metric], '--',label=name+' val')
    ax.set(xlabel='Epoch', ylabel=metric); ax.legend(fontsize=7); plt.tight_layout(); plt.show()

def sample_image():
    # Public-domain sample bundled with scikit-learn; no personal photo uploaded.
    from sklearn.datasets import load_sample_image
    from PIL import Image
    path = DATA/'sample_flower.jpg'
    if not path.exists(): Image.fromarray(load_sample_image('flower.jpg')).save(path)
    return path

def torch_setup():
    import torch
    seed(); torch.manual_seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def keras_setup():
    # Keras 3 retains the course Sequential/Dense/Conv2D APIs on PyTorch.
    # Set the backend before importing Keras; no duplicate TensorFlow install needed.
    os.environ.setdefault('KERAS_BACKEND','torch')
    import keras
    torch_setup();keras.utils.set_random_seed(SEED)
    return keras
