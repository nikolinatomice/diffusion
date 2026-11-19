# diffusion

Tiny Denoising Diffusion Probabilistic Model (DDPM) CIFAR-10 with TensorBoard logging.

## Setup

```bash
# conda
conda create -n diffusion python=3.11 -y
conda activate diffusion

# or venv (macOS/Linux), named diffusion
python3 -m venv ~/.venvs/diffusion
source ~/.venvs/diffusion/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python train.py                  # downloads CIFAR-10 to ./data
tensorboard --logdir runs
```s

Resume: `python train.py --run-name <name> --resume runs/<name>/checkpoint.pt`


Note: CUDA only. Mac trains on CPU.
Trained for 500 epochs.
