# %%
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.datasets import Flowers102
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
# %%
# HYPERPARAMS
T = 50
img_size = 64
batch_size = 64
embedding_dims = 32

# reload the data with new transforms?
RELOAD = True


# %%
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# %%

transforms = torch.nn.Sequential(
  torchvision.transforms.RandAugment(2, 5),
  torchvision.transforms.RandomResizedCrop(img_size),
  torchvision.transforms.RandomHorizontalFlip(),
).to(device)

# %%
train_flowers = Flowers102(
    root="/data/ml_data", download=True, transform=torchvision.transforms.ToTensor(), split="train"
)
test_flowers = Flowers102(
    root="/data/ml_data", download=True, transform=torchvision.transforms.ToTensor(), split="test"
)
# val_flowers = Flowers102(
#     root="data/ml_data", download=True, transform=transforms, split="val"
# )
# %%
# make one big torch dataset from it
if RELOAD:
    X = [f[0] for f in train_flowers]
    X += [f[0] for f in test_flowers]
    Y = [f[1] for f in train_flowers]
    Y += [f[1] for f in test_flowers]
    X = transforms(torch.stack(X).to(device))
    Y = torch.tensor(Y).to(device)
    # normalize X
    X = (X - X.mean()) / X.std()
    torch.save(X, "/data/ml_data/flowers-102/X.pt")
    torch.save(Y, "/data/ml_data/flowers-102/Y.pt")
else:
    X = torch.load("/data/ml_data/flowers-102/X.pt")
    Y = torch.load("/data/ml_data/flowers-102/Y.pt")

# %%
def sinusoidal_embedding(x, emb_dims, max_freq):
    frequencies = torch.exp(
        torch.linspace(0, torch.math.log(max_freq), emb_dims // 2, device=x.device)
    )
    angular_speeds = 2.0 * torch.math.pi * frequencies
    embeddings = torch.concat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], axis=3
    )
    return embeddings


# %%
class conv_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_c)
        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        self.swish = torch.nn.SiLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.swish(x)
        return x


class encoder_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = torch.nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0
        )
        self.conv = conv_block(out_c * 2, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(embedding_dims + 3, 64)
        self.e2 = encoder_block(64, 128)
        self.b = conv_block(128, 256)
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)
        self.outputs = torch.nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, inputs, noise_level):
        enc = sinusoidal_embedding(noise_level, embedding_dims, 1e3)
        enc = enc.view(-1, embedding_dims, 1, 1)
        enc = enc.repeat(1, 1, img_size, img_size)
        s1, p1 = self.e1(torch.cat([inputs, enc], axis=1))
        s2, p2 = self.e2(p1)
        b = self.b(p2)
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)
        outputs = self.outputs(d2)
        return outputs


# %%
class DenoisingDiffusionModel(torch.nn.Module):
    def __init__(self, T, minval, maxval):
        super().__init__()
        self.T = T  # maximum diffusion time
        startangle = torch.acos(torch.tensor(maxval, dtype=torch.float32))
        endangle = torch.acos(torch.tensor(minval, dtype=torch.float32))
        steps = torch.linspace(startangle, endangle, T)
        noise_level = torch.sin(steps)
        signal_level = torch.cos(steps)
        self.register_buffer("noise_level", noise_level.view(-1, 1, 1, 1))
        self.register_buffer("signal_level", signal_level.view(-1, 1, 1, 1))

    def diffusion_loss(self, model, X):
        # we sample from a N(0, 1)
        eps = torch.randn_like(X, device=X.device)
        # sample a time
        t = torch.randint(0, self.T, (X.shape[0],), device=X.device)
        X_noisy = self.signal_level[t] * X + self.noise_level[t] * eps
        # model is supposed to predict noise as close as possible to eps
        eps_pred = model(X_noisy, self.noise_level[t]**2)
        return F.l1_loss(eps_pred, eps)

    def sample(self, model: torch.nn.Module, n_samples: int, dims: tuple, X0=None):
        with torch.no_grad():
            device = next(model.parameters()).device
            if X0 is None:
                img_noisy = torch.randn(n_samples, *dims, device=device)
            else:
                img_noisy = X0
            for t in reversed(range(self.T)):
                signal_level, noise_level = self.signal_level[t], self.noise_level[t]
                eps_pred = model(
                    img_noisy,
                    (noise_level ** 2).repeat(n_samples, 1, 1, 1),
                )
                img_pred = (img_noisy - noise_level * eps_pred) / signal_level

                # noise it again for next iteration
                next_signal_level, next_noise_level = (
                    self.signal_level[t - 1],
                    self.noise_level[t - 1],
                )
                eps_new = torch.randn_like(img_pred, device=device)
                img_noisy = next_signal_level * img_pred + next_noise_level * eps_new
            return img_pred  # that way we drop the last noise


# %%
def get_slice(X, batch_size):
    idxs = torch.randperm(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        yield X[idxs[i : i + batch_size]]


def train(model, optimizer, diffusion_model, X, n_epochs=50):
    bar = tqdm(range(n_epochs))
    for epoch in bar:
        losses = []
        for x in get_slice(X, batch_size):
            optimizer.zero_grad()
            loss = diffusion_model.diffusion_loss(model, x)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        bar.set_description(f"loss: {torch.tensor(losses).mean().item():.3f}")


# %%
model = UNet().to(device)
# model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
diffusion_model = DenoisingDiffusionModel(T, .01, 0.95).to(device)
# n params in model
sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
train(model, optimizer, diffusion_model, X, n_epochs=100)

# %%
# sample and plot 5
output = diffusion_model.sample(model, 5, dims=X.shape[1:])
# get back to [0,1]
output = output - output.min()
output /= output.max()
# %%
# plot all
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i in range(5):
    axs[i].imshow(output[i].permute(1, 2, 0).cpu().numpy())

# %%
