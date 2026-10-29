# Validation status

Checked on 30 October 2026. Quick runs use real public data with a reduced budget; they are not full-training benchmarks.

| Notebook | End-to-end execution | Additional verification |
|---|---|---|
| 01 Perceptron | Passed | Both label encodings; sigmoid model; manual prediction |
| 02 MLP | Passed | Nine architecture/rate experiments; finite-difference gradient check |
| 03 ANN | Passed in quick mode | Eight model variants, validation selection, held-out MNIST test |
| 04 Augmentation | Passed | Exactly 40 JPEGs generated; grid stored in notebook output |
| 05 CNN | Not yet verified end-to-end | All five model variants passed a synthetic-data training step |
| 06 Pretrained classification | Not yet verified end-to-end | Syntax/schema checked; official weight downloads required |
| 07 Transfer learning | Not yet verified end-to-end | VGG16 and ResNet50 frozen/fine-tuned gradient paths checked without pretrained weights |
| 08A Custom Faster R-CNN | Public dataset download failed certificate validation in Python | Both backbone widths passed synthetic training losses/backpropagation/inference |
| 08B Pretrained detection | Public dataset download failed certificate validation in Python | Syntax/schema checked; dataset and weight downloads required |
| 09 SSD/YOLO | Not yet verified end-to-end | Teaching-head shapes and real SSD replacement-head loss/backpropagation checked |

All ten notebook schemas and all code-cell syntax passed validation. The AP evaluator returns AP=1 on an exact-match toy example. These checks verify code paths, **not learned detection accuracy**.

The complete implementations and experiment loops are included. No pretrained predictions, detector scores or training results have been invented for unexecuted notebooks. Their notebook outputs are intentionally empty. Use the Colab links and Run All to produce those results; full training remains to be executed.

## Reproduce checks

```bash
python scripts/validate_notebooks.py
python scripts/smoke_models.py
python scripts/execute_notebook.py notebooks/01_Perceptron.ipynb
```

The offline model check uses random synthetic inputs and disables pretrained downloads only inside its own process. It does not alter the tutorial implementations. The execution script saves actual notebook outputs and a local status JSON in `artifacts/`.

## Dataset download troubleshooting

Download PennFudanPed.zip from the official University of Pennsylvania address linked in the README and place it in `data/` if Python's certificate store rejects the server. The loader accepts a valid pre-downloaded ZIP. Do not disable certificate verification. The larger image tutorials also need internet for CIFAR-10 and pretrained weights; VGG16 alone is approximately 528 MiB.

## Observed runtime

Python 3.14, CPU, PyTorch 2.14.0, Torchvision 0.29.2, Keras 3.15.1 with PyTorch backend, scikit-learn 1.9.1. The repository requirements support installing dependencies separately; GPU/other versions can produce different timings and results.
