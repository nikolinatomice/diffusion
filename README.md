# diffusion

Tiny DDPM on CIFAR-10 with TensorBoard logging.

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
```

Smoke test (random data, no download):

```bash
python train.py --fake-data --epochs 2 --sample-every 1 --max-steps-per-epoch 5 \
                --batch-size 16 --num-workers 0 --T 50
```

Resume: `python train.py --run-name <name> --resume runs/<name>/checkpoint.pt`

## Files

- `model.py`: U-Net, sinusoidal time embedding with transformer blocks
- `diffusion.py`: schedule, `add_noise`, DDPM sampling
- `data.py`: CIFAR-10 loader
- `train.py`: training loop, EMA, checkpoints, TensorBoard

## TensorBoard

- Scalars: loss, grad norm, LR, loss by timestep bucket
- Images: per-epoch denoising preview (noisy / predicted x0), periodic EMA samples, denoising trajectories
- Histograms: weights

Note: CUDA only. Mac trains on CPU.
