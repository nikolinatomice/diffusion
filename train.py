"""Train a tiny DDPM on CIFAR-10 with TensorBoard logging.

    python train.py                       # default run
    python train.py --fake-data --epochs 2 --sample-every 1   # quick smoke test
    tensorboard --logdir runs             # in another terminal
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import get_dataloader
from diffusion import Diffusion
from model import DiffusionUNet


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=3, choices=[1, 3])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fake-data", action="store_true", help="random data, for smoke tests")
    # model
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=256)
    # diffusion
    p.add_argument("--T", type=int, default=1000)
    # optimisation
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch (debug aid)")
    # logging
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50, help="steps between scalar logs")
    p.add_argument("--sample-every", type=int, default=5, help="epochs between full DDPM sampling")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint.pt")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

class EMA:
    """Exponential moving average of the weights. Sampling from the EMA model gives
    noticeably cleaner images than sampling from the raw, noisy-SGD weights."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # warm-up so early EMA isn't junk
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            e.lerp_(p, 1.0 - d)
        for e, b in zip(self.shadow.buffers(), model.buffers()):
            e.copy_(b)


def to_img(x):
    """[-1, 1] -> [0, 1] for display."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def denoise_preview(model, diffusion, x_clean, ts, device, seed=0):
    """One-shot denoising at several fixed noise levels.

    Returns a grid where, for every image, row 1 = noisy input x_t and
    row 2 = the model's predicted clean image x0 (columns = timesteps in `ts`).
    Cheap (one forward pass), so we log it every epoch.
    """
    n, K = x_clean.shape[0], len(ts)
    x_rep = x_clean.repeat_interleave(K, dim=0)
    t = torch.tensor(ts, device=device).repeat(n)

    g = torch.Generator(device=device).manual_seed(seed)  # same noise every call
    noise = torch.randn(x_rep.shape, device=device, generator=g)
    noisy, _ = diffusion.add_noise(x_rep, t, noise)

    pred = diffusion.predict_x0(noisy, t, model(noisy, t))

    C, H, W = x_clean.shape[1:]
    rows = torch.stack([noisy.view(n, K, C, H, W), pred.view(n, K, C, H, W)], dim=1)
    return make_grid(to_img(rows.reshape(-1, C, H, W)).cpu(), nrow=K, padding=2)


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, step, args):
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # ---------- run dir + tensorboard ----------
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    writer.add_text("config", "```\n" + json.dumps(vars(args), indent=2) + "\n```", 0)

    # ---------- data ----------
    loader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        channels=args.channels,
        num_workers=args.num_workers,
        fake_data=args.fake_data,
    )

    # ---------- diffusion / model / optim ----------
    diffusion = Diffusion(T=args.T, device=device)

    model = DiffusionUNet(
        in_channels=args.channels,
        out_channels=args.channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
    ).to(device)
    ema = EMA(model, args.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    writer.add_scalar("model/params_millions", n_params / 1e6, 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / args.warmup_steps)
    )

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    # ---------- fixed things we log every epoch for comparability ----------
    x_fixed, _ = next(iter(loader))
    x_fixed = x_fixed[:4].to(device)
    preview_ts = [10, 100, 250, 500, 750, 999]
    preview_ts = [min(t, args.T - 1) for t in preview_ts]

    writer.add_image("data/real_batch", make_grid(to_img(x_fixed.cpu()), nrow=4), 0)

    # ---------- training ----------
    n_buckets = 4  # loss is logged separately for low / mid / high noise levels
    bucket_names = [f"t_{i * args.T // n_buckets}-{(i + 1) * args.T // n_buckets - 1}" for i in range(n_buckets)]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_steps = 0
        bucket_sum = torch.zeros(n_buckets, device=device)
        bucket_cnt = torch.zeros(n_buckets, device=device)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for i, (x, _) in enumerate(pbar):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break

            x = x.to(device, non_blocking=True)

            # random timestep per image
            t = torch.randint(0, args.T, (x.shape[0],), device=device)

            # forward process + noise prediction
            noisy_x, noise = diffusion.add_noise(x, t)
            pred = model(noisy_x, t)

            loss_per_sample = F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2, 3))
            loss = loss_per_sample.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model, global_step)

            # bookkeeping (kept on-GPU to avoid a sync every step)
            epoch_loss_sum += loss.detach()
            epoch_steps += 1
            bucket = (t * n_buckets) // args.T
            bucket_sum.index_add_(0, bucket, loss_per_sample.detach())
            bucket_cnt.index_add_(0, bucket, torch.ones_like(loss_per_sample))

            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            global_step += 1

        # ---------- end of epoch logging ----------
        epoch_loss = (epoch_loss_sum / max(epoch_steps, 1)).item()
        writer.add_scalar("epoch/loss", epoch_loss, epoch + 1)
        for name, s, c in zip(bucket_names, bucket_sum.tolist(), bucket_cnt.tolist()):
            if c > 0:
                writer.add_scalar(f"epoch_loss_by_timestep/{name}", s / c, epoch + 1)

        # cheap: one-shot denoising preview (noisy row / predicted-x0 row)
        writer.add_image(
            "denoise_preview/noisy_then_predicted_x0",
            denoise_preview(ema.shadow, diffusion, x_fixed, preview_ts, device),
            epoch + 1,
        )

        print(f"Epoch {epoch + 1}/{args.epochs}  loss: {epoch_loss:.4f}  steps: {global_step}")

        # expensive: full 1000-step sampling from pure noise (+ intermediate steps)
        is_last = epoch + 1 == args.epochs
        if (epoch + 1) % args.sample_every == 0 or is_last:
            g = torch.Generator(device=device).manual_seed(args.seed)  # same start noise each time
            shape = (args.n_samples, args.channels, args.image_size, args.image_size)
            samples, snaps = diffusion.sample(ema.shadow, shape, n_snapshots=8, generator=g)

            writer.add_image("samples/final", make_grid(to_img(samples).cpu(), nrow=4), epoch + 1)

            # rows = samples, columns = noisy -> clean
            traj = torch.stack(snaps, dim=1)[:8]  # [8, S, C, H, W]
            C, H, W = traj.shape[2:]
            writer.add_image(
                "samples/trajectory_noisy_to_clean",
                make_grid(to_img(traj.reshape(-1, C, H, W)), nrow=len(snaps), padding=2),
                epoch + 1,
            )
            save_image(to_img(samples), run_dir / "samples" / f"epoch_{epoch + 1:04d}.png", nrow=4)

            for name, p in model.named_parameters():
                writer.add_histogram(f"weights/{name}", p, epoch + 1)

        save_checkpoint(run_dir / "checkpoint.pt", model, ema, optimizer, scheduler, epoch + 1, global_step, args)
        writer.flush()

    writer.close()
    print(f"Done. Logs + checkpoint in {run_dir}")


if __name__ == "__main__":
    main()
