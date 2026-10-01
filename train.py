import torch
import torch.nn.functional as F


# ============================================================
# Diffusion settings
# ============================================================

T = 1000

betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


# ============================================================
# Add noise
# ============================================================

def add_noise(x, t):
    alpha_bar = alpha_bars[t].to(x.device)

    sqrt_alpha_bar = torch.sqrt(alpha_bar)[:, None, None, None]
    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar
    )[:, None, None, None]

    noise = torch.randn_like(x)

    noisy_x = (
        sqrt_alpha_bar * x
        + sqrt_one_minus_alpha_bar * noise
    )

    return noisy_x, noise


# ============================================================
# Model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DiffusionUNet(
    in_channels=1,
    out_channels=1,
    base_channels=32,
    time_dim=128,
).to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)


# ============================================================
# Training
# ============================================================

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    for x, _ in train_loader:

        x = x.to(device)

        # Random timestep for every image
        t = torch.randint(
            0,
            T,
            (x.shape[0],),
            device=device,
        )

        # Add noise
        noisy_x, noise = add_noise(x, t)

        # Predict the noise
        predicted_noise = model(
            noisy_x,
            t,
        )

        # Diffusion loss
        loss = F.mse_loss(
            predicted_noise,
            noise,
        )

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"Loss: {loss.item():.4f}"
    )