# Siamese Augmentation Strategies (SiAug)

[**Paper**](https://openreview.net/pdf?id=xkmhsBITaCw) | [**OpenReview**](https://openreview.net/forum?id=xkmhsBITaCw) | [**ArXiv**](https://arxiv.org/abs/2301.12636)

This repository contains the implementation code for our paper: <br>
[Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays](https://openreview.net/pdf?id=xkmhsBITaCw)

- Authors: Rogier van der Sluijs\*, Nandita Bhaskhar\*, Daniel Rubin, Curtis Langlotz, Akshay Chaudhari
- \*- co-first authors
- Published at Medical Imaging with Deep Learning (MIDL)

## tl;dr

Tailored augmentation strategies for image-only Siamese representation learning can outperform supervised baselines with zero-shot learning, linear probing and fine-tuning for chest X-ray classification. We systematically assess the effect of various augmentations on the quality and robustness of the learned representations. We train and evaluate Siamese Networks for abnormality detection on chest X-Rays across three large datasets (MIMIC-CXR, CheXpert and VinDr-CXR). We investigate the efficacy of the learned representations through experiments involving linear probing, fine-tuning, zero-shot transfer, and data efficiency. Finally, we identify a set of augmentations that yield robust representations that generalize well to both out-of-distribution data and diseases, while outperforming supervised baselines using just zero-shot transfer and linear probes by up to 20%.

## Installation

To contribute to _siaug_, you can install the package in editable mode:

```python
pip install -e .
pip install -r requirements.txt
pre-commit install
pre-commit
```

Make sure to update the `.env` file according to the setup of your cluster and placement of your project folder on disk. Also, run `accelerate config` to generate a config file, and copy it from `~/cache/huggingface/accelerate/default_config.yaml` to the project directory. Finally, create symlinks from the `data/` folder to the datasets you would want to train on.

## Technology Stack

This repository does not currently contain a single section that documents the full stack end to end. The main technologies used in the project are:

- Python as the main implementation language.
- PyTorch and Torchvision for model definition, training, datasets and image transforms.
- Hydra for configuration management. In this project Hydra composes the final run configuration from `configs/train.yaml` together with experiment, model, dataloader, optimizer, criterion, logger and path configs, and lets you override any field from the command line.
- Hugging Face Accelerate for device placement, mixed precision, distributed training and unified experiment tracking APIs.
- TensorBoard for local metric tracking and visualization.
- Weights & Biases (W&B) as an optional remote experiment tracker.
- Timm for image backbones.
- Transformers and Diffusers for model components used in some experiments.
- Kornia and project-local augmentations for image augmentation pipelines.
- Safetensors for model checkpoint serialization.
- Pyrootutils for resolving the project root and environment-aware paths.
- Lightning metrics / TorchMetrics-related utilities for evaluation metrics.
- Polars and Zarr for data handling utilities used by the project.
- Pytest and Pre-commit for testing and repository hygiene.

## Helpful Docs

Project-specific working documentation has been moved to the companion documentation repository.

## SimSiam Architecture In This Project

The main representation-learning model in this repository is a SimSiam-style siamese network implemented in `siaug/models/simsiam.py`.

At a high level, the training pipeline works as follows:

1. One input image is transformed into two different augmented views.
2. Both views are passed through the same encoder with shared weights.
3. The encoder output of each branch is passed through a projector MLP.
4. The projected representation of each branch is passed through a predictor MLP.
5. The loss encourages the predictor output from one branch to match the projected representation from the other branch.

In symbols:

```text
image
  -> aug1 -> encoder -> projector -> predictor -> p1
  -> aug2 -> encoder -> projector -> predictor -> p2

loss = 0.5 * D(p1, stopgrad(z2)) + 0.5 * D(p2, stopgrad(z1))
```

Where:

- `encoder` is the image backbone, created through `timm`. In the NIH representation-learning experiment this is `resnet50`.
- `projector` is an MLP that maps encoder features into the self-supervised representation space.
- `predictor` is an MLP that predicts the target representation from the opposite branch.
- `D` is the negative cosine similarity loss.
- `stopgrad` means gradients are stopped on the target projection branch, which is a core part of SimSiam and helps avoid collapse.

Important implementation details in this repository:

- The two views are created by the dataset transform pipeline, for example through `siaug.augmentations.ToSiamese`.
- The encoder weights are shared across both branches. This is what makes the network siamese.
- The model returns `p1`, `p2`, `z1`, and `z2`, and the loss is computed from these tensors.
- The learned representation used later for downstream tasks is the encoder representation, not the predictor output.

### Class imbalance

For the NIH dataset you can compute class weights to mitigate label imbalance.
Run the helper script below and copy the resulting weights into
`configs/criterion/focal_pos_weight.yaml` or override them via the command line.

```bash
python scripts/compute_nih_pos_weights.py /path/to/Data_Entry_2017_v2020.csv \
    /path/to/images --list_path /path/to/train_val_list.txt
```

## Training

Currently, we support two modes of training: pretraining and linear evaluation.

### Representation learning

To learn a new representation, you can use the `train_repr.py` script.

```python
# Train and log to WandB
accelerate launch siaug/train_repr.py experiment=experiment_name logger=wandb

# Train and log locally to TensorBoard
accelerate launch siaug/train_repr.py experiment=experiment_name logger=tensorboard

# Resume from checkpoint
accelerate launch siaug/train_repr.py ... resume_from_ckpt=/path/to/accelerate/ckpt/dir

# Run a fast_dev_run
accelerate launch siaug/train_repr.py ... fast_dev_run=True max_epoch=10 log_every_n_steps=1 ckpt_every_n_epochs=1
```

To monitor local TensorBoard metrics while training is running, start TensorBoard in a separate terminal:

```python
tensorboard --logdir logs
```

Then open the local URL printed by TensorBoard, typically `http://localhost:6006`.

### Linear evaluation

To train a linear classifier on top of a frozen backbone, use the `train_lcls.py` script.

```python
# Train a linear classifier on top of a frozen backbone
accelerate launch siaug/train_lcls.py \
    experiment=experiment_name \
    model.ckpt_path=/path/to/model.safetensors

# Train a linear classifier on top of a random initialized backbone
accelerate launch siaug/train_lcls.py model.ckpt_path=None

# Use ImageNet pretrained weights
accelerate launch siaug/train_lcls.py +model.pretrained=True
```

If you use `logger=tensorboard`, you can monitor the training metrics in real time with:

```python
tensorboard --logdir logs
```

### Zero Shot Evaluation

To evaluate a model on a downstream task without fine-tuning, use the `siaug/eval.py` script.

```python
python siaug/eval.py experiment=eval_chex_resnet +checkpoint_folder=/path/to/model/checkpoints/folder +save_path=/path/to/save/resulting/pickle/files
```

## Contact Us

<a name="contact"></a>
This repository is being developed at the Stanford's MIMI Lab. Please reach out to `sluijs [at] stanford [dot] edu` and `nanbhas [at] stanford [dot] edu` if you would like to use or contribute to `siaug`.

## Citation

If you find our paper and/or code useful, please use the following BibTex for citation:

```bib
@article{sluijsnanbhas2023_siaug,
  title={Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays},
  author={Rogier van der Sluijs and Nandita Bhaskhar and Daniel Rubin and Curtis Langlotz and Akshay Chaudhari},
  year={2023},
  journal={Medical Imaging with Deep Learning (MIDL)},
}
```
