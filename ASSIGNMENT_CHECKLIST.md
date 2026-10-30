# Assignment review checklist

This checklist maps the course tasks to evidence. It does not assign or predict marks. The teacher's rubric remains authoritative.

## Current readiness

- Tutorials 1 and 2: executed with results and written interpretation.
- Tutorial 3: eight real MNIST quick-run experiments, figures and interpretation saved. A longer run is still needed to demonstrate early stopping or convergence.
- Tutorial 4: 40 JPEGs generated and shown in a grid; source attribution included.
- Tutorials 5–9 (including both Tutorial 8 versions): implementations and explanations are present, but end-to-end public-data/model-weight runs remain outstanding. Do not present these as fully trained experiments.

## Evidence to retain before submission

| Tutorial | Required evidence | Where to find it |
|---|---|---|
| 1 | Two label encodings, sigmoid extension, accuracy, manual prediction | Notebook 01 outputs and task answers |
| 2 | Baseline, architecture/rate table, loss curves, selected test report, gradient calculation | Notebook 02 |
| 3 | Parameter counts, optimizer/activation comparisons, training/validation curves, predictions, explanation of overfitting | Notebook 03; longer run if required by the rubric |
| 4 | Original image identification, five transformation types, 40 saved variants | Notebook 04 and local artifacts/tutorial04_augmented |
| 5 | CNN summary, filter/Dense variations, measured results, regularization comparison | Run Notebook 05; offline checks alone are insufficient |
| 6 | Five models and five ranked labels per model, preprocessing explanation | Run Notebook 06 |
| 7 | Frozen/fine-tuned VGG and ResNet plus custom baseline, split sizes, learning rates, metrics | Run Notebook 07 |
| 8A | Dataset/annotations, backbone configuration, four losses, validation selection and test AP | Run Notebook 08A |
| 8B | Three pretrained model examples, boxed images and same-dataset AP comparison | Run Notebook 08B |
| 9 | SSD/YOLO head shapes, labels, real training logs, held-out results and prediction image | Run Notebook 09 |

## Explain every result in four sentences

1. **Method:** which data, preprocessing, architecture and training settings were used?
2. **Observation:** what did the actual table or curve show, with the relevant number?
3. **Interpretation:** why might that have happened, and what alternative explanation exists?
4. **Limitation:** what small sample size, short training budget, seed variation or model difference limits the conclusion?

Example from Tutorial 2: “I compared three MLP architectures at three learning rates using a fixed validation split. The five-neuron model at 0.001 achieved 96.67% validation accuracy and 96.67% held-out test accuracy. The larger networks did not improve validation accuracy in this run, so extra capacity was unnecessary here. There were only 30 test examples; one changed prediction would move accuracy by 3.33 percentage points.”

## Before teacher review

- Restart the kernel and Run All; confirm that there are no error cells.
- Report whether the notebook ran in quick or full mode and give actual sample counts/epochs.
- Keep test data out of training, preprocessing fitting and hyperparameter selection.
- Report measured values; leave an unfinished experiment labelled unfinished.
- Include your own brief observations after rerunning and inspecting the curves.
- Preserve third-party attributions and follow your class policy on AI-assisted work.
- Do not promise a model is suitable for deployment based on these small coursework experiments.

## Revision route before the exam

Start with notebook 01's update rule, then notebook 02's backward pass. Continue with notebook 03's losses, regularization and parameter counts; notebook 05's convolution sizes; notebook 07's freezing and fine-tuning; and notebooks 08–09's detection losses, IoU, AP and output shapes. Each notebook ends with task answers and short numerical questions.

## Running the outstanding work

Use the README Colab links for GPU access, or install the requirements locally. Run all locally with:

```bash
python scripts/run_all.py
```

For the larger training budgets:

```bash
python scripts/run_all.py --full
```

To run only one selected notebook:

```bash
python scripts/run_all.py notebooks/05_CNN_CIFAR10.ipynb
```

The batch runner uses a fresh kernel for each notebook, records every failure and exits unsuccessfully if any notebook failed. Successful execution is separate from learning useful accuracy: check both the run status and the measured results.
