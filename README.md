# Deep Learning tutorials with worked explanations

Ten executable notebooks covering the nine course tutorials (Tutorial 8 has two versions). Each notebook explains the objective, data, model, training, evaluation and assigned experiments.

## Start here

1. Open a notebook using its Colab link below.
2. Select **Runtime → Change runtime type → GPU** for the image-model tutorials if available.
3. Run the cells in order. The first code cell fetches the support files and installs missing packages in Colab.
4. Read the explanation before each code cell and inspect its actual outputs.

| Course tutorial | Notebook | Colab |
|---|---|---|
| 1 Perceptron | [01](notebooks/01_Perceptron.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/01_Perceptron.ipynb) |
| 2 MLP | [02](notebooks/02_MLP.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/02_MLP.ipynb) |
| 3 ANN on MNIST | [03](notebooks/03_ANN_MNIST.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/03_ANN_MNIST.ipynb) |
| 4 Data augmentation | [04](notebooks/04_Data_Augmentation.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/04_Data_Augmentation.ipynb) |
| 5 CNN on CIFAR-10 | [05](notebooks/05_CNN_CIFAR10.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/05_CNN_CIFAR10.ipynb) |
| 6 Pretrained classification | [06](notebooks/06_Pretrained_Classification.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/06_Pretrained_Classification.ipynb) |
| 7 Transfer learning | [07](notebooks/07_Transfer_Learning.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/07_Transfer_Learning.ipynb) |
| 8 Custom detector | [08A](notebooks/08A_Custom_Faster_RCNN.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/08A_Custom_Faster_RCNN.ipynb) |
| 8 Pretrained detector | [08B](notebooks/08B_Pretrained_Detection.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/08B_Pretrained_Detection.ipynb) |
| 9 SSD and YOLO | [09](notebooks/09_SSD_YOLO.ipynb) | [Open](https://colab.research.google.com/github/naveed952/Deep_learning-Tutorials_py.ipynb/blob/main/notebooks/09_SSD_YOLO.ipynb) |

## Local setup

Python 3.12 or newer is recommended. Install PyTorch appropriate for your CPU/GPU using its official instructions if the default wheel is unsuitable.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dl-tutorials --display-name "Deep Learning Tutorials"
```

Open the repository in VS Code/Jupyter, select that kernel and Run All. Notebooks run from either the repository root or `notebooks/`.

Keras 3 uses the **PyTorch backend** here, preserving the Sequential/Dense/Conv2D training APIs without a second large framework installation. Tutorial 4 uses modern Keras preprocessing layers instead of legacy ImageDataGenerator. Tutorials 6–7 implement the handout's requested PyTorch extension. Tutorial 8B uses Torchvision instead of the handout's TensorFlow Hub loading route. These differences are explained in the notebooks.

## Quick and larger training runs

Quick mode is the default: it executes small real-data experiments, not final benchmarks. Tutorial 4 still saves exactly 40 JPEGs; Tutorial 6 still runs all five pretrained models. Set `TUTORIAL_FULL=1` **before starting a new kernel** for larger datasets and longer training. In Colab, insert this before the project setup cell:

```python
import os
os.environ['TUTORIAL_FULL'] = '1'
```

Downloads require internet and can take time: CIFAR-10, Penn–Fudan and especially VGG16/AlexNet weights are large. GPU acceleration is recommended for the full image-model experiments. Results vary by seed, device and package version.

## What each task implements

- **1:** custom step perceptron, -1/+1 and 0/1 labels, sigmoid + BCE training, editable manual measurements.
- **2:** (10,10) baseline, nine architecture/rate combinations, loss curves, validation selection, test report, worked backpropagation and numerical gradient check.
- **3:** MNIST baseline, small/deep architectures, ReLU/tanh/sigmoid, SGD/RMSprop/Adam, L2/dropout, early stopping, curves and predictions.
- **4:** rotation, shear, zoom, flips and brightness; exactly 40 saved augmented JPEGs and an inspection grid.
- **5:** CIFAR-10 CNN, kernel/filter/Dense comparisons, regularized variant, parameter calculation and a convolution-gradient example.
- **6:** top-five predictions for VGG16, ResNet50, AlexNet, ResNet101 and MobileNet; dataset transfer exercise continues in 7.
- **7:** frozen and fine-tuned VGG16/ResNet50, custom CNN baseline, fixed splits, reduced fine-tuning rate, weight decay and early stopping.
- **8A:** small custom backbone inside a complete Faster R-CNN, width comparison, real annotations, training losses, AP evaluation, RoI-pooling and box-offset examples.
- **8B:** pretrained ResNet50 Faster R-CNN, MobileNet Faster R-CNN and VGG16 SSD; boxed predictions and actual mAP computation on annotated data.
- **9:** teaching SSD/YOLO head implementations and shape checks, pretrained SSD adaptation/training, YOLO label conversion/training and held-out evaluation. The toy heads are explicitly distinguished from complete trainable detectors.

## Data and outputs

Iris and the public sample photograph come from scikit-learn. MNIST/CIFAR-10 use their standard loaders. Detection uses [Penn–Fudan](https://www.cis.upenn.edu/~jshi/ped_html/), with boxes derived from its instance masks. Full detection split: 150 train, 10 validation, 10 test; quick mode: disjoint 16/4/4 subsets. This is a small instructional benchmark, not a claim of robust deployment performance. Use the original dataset terms when redistributing images.

Downloads, model weights and generated artifacts are ignored by Git. Only source, explanations, and verified notebook outputs are intended for upload. `artifacts/` stores generated images, metrics and optional training files locally. No personal images or course PDFs are uploaded.

## Assignment and exam preparation

Read [the assignment checklist](ASSIGNMENT_CHECKLIST.md) before submission. Every notebook ends with task answers, interpretation guidance and numerical revision questions. See [third-party notices](THIRD_PARTY_NOTICES.md) for the sample-photo attribution.

## Verification

```bash
python scripts/validate_notebooks.py
python scripts/execute_notebook.py notebooks/01_Perceptron.ipynb
```

See [VALIDATION.md](VALIDATION.md) for the actual execution status. An unexecuted notebook is not a reported training result. Test data is reserved for final evaluation; validation data controls model selection.

## Useful references

- [Keras backend configuration](https://keras.io/getting_started/)
- [Torchvision pretrained weights](https://docs.pytorch.org/vision/stable/models.html)
- [Official detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)

Third-party software/datasets retain their own licenses. 
