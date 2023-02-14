# %%
import matplotlib.pyplot as plt
import torch
import tqdm
from sklearn.datasets import make_swiss_roll
import numpy as np

# %%
class DenoisingDiffusionModel(torch.nn.Module):
    def __init__(self, T, minval, maxval):
        super().__init__()
        self.T = T  # maximum diffusion time
        self.register_buffer("beta", torch.linspace(minval, maxval, T))
        self.register_buffer("alpha", 1 - self.beta.unsqueeze(1))
        self.register_buffer("alphaprod", torch.cumprod(self.alpha, dim=0))

    def diffusion_loss(self, model, X):
        # we sample from a N(0, 1)
        eps = torch.randn_like(X, device=X.device)
        # sample a time
        t = torch.randint(0, self.T, (X.shape[0],), device=X.device)
        alpha = self.alphaprod[t]
        # noise up X
        X_noisy = torch.sqrt(alpha) * X + torch.sqrt(1 - alpha) * eps
        # model is supposed to predict noise as close as possible to eps
        eps_pred = model(X_noisy, t)
        return torch.nn.functional.mse_loss(eps_pred, eps)

    def sample(self, model, n_samples, ndim = 2, X0 = None):
        with torch.no_grad():
            device = next(model.parameters()).device
            if X0 is None:
              X = torch.randn(n_samples, ndim, device=device)
            else:
              X = X0
            for t in reversed(range(self.T)):
                alphaprod = self.alphaprod[t]
                alpha = self.alpha[t]
                eps_theta = model(X, torch.tensor([t] * n_samples, device=device))
                X_mean = X - (1 - alpha) / torch.sqrt(1 - alphaprod) * eps_theta
                X_mean /= torch.sqrt(alpha)
                X = X_mean + torch.randn_like(X_mean) * torch.sqrt(1 - alpha)
                if t % (self.T // 10) == 0:
                    plt.scatter(X[:, 0], X[:, 1])
                    plt.show()
            return X_mean  # that way we drop the last noise


# %%
class PositionalEncoding(torch.nn.Module):
    """The classic positional encoding from the original Attention papers"""

    def __init__(
        self,
        d_model: int = 128,
        maxlen: int = 1024,
        min_freq: float = 1e-4,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Args:
            d_model (int, optional): embedding dimension of each token. Defaults to 128.
            maxlen (int, optional): maximum sequence length. Defaults to 1024.
            min_freq (float, optional): use the magic 1/10,000 value! Defaults to 1e-4.
            device (str, optional): accelerator or nah. Defaults to "cpu".
            dtype (_type_, optional): torch dtype. Defaults to torch.float32.
        """
        super().__init__()
        pos_enc = self._get_pos_enc(d_model=d_model, maxlen=maxlen, min_freq=min_freq)
        self.register_buffer(
            "pos_enc", torch.tensor(pos_enc, dtype=dtype, device=device)
        )

    def _get_pos_enc(self, d_model: int, maxlen: int, min_freq: float):
        position = np.arange(maxlen)
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_enc = position[:, None] * freqs[None]
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def forward(self, x):
        return self.pos_enc[x]


class DiscreteTimeResidualBlock(torch.nn.Module):
    """Generic block to learn a nonlinear function f(x, t), where
    t is discrete and x is continuous."""

    def __init__(self, d_model: int, maxlen: int = 512):
        super().__init__()
        self.d_model = d_model
        self.emb = PositionalEncoding(d_model=d_model, maxlen=maxlen)
        self.lin1 = torch.nn.Linear(d_model, d_model)
        self.lin2 = torch.nn.Linear(d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.act = torch.nn.GELU()

    def forward(self, x, t):
        return self.norm(x + self.lin2(self.act(self.lin1(x + self.emb(t)))))


class BasicDiscreteTimeModel(torch.nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lin_in = torch.nn.Linear(2, d_model)
        self.lin_out = torch.nn.Linear(d_model, 2)
        self.blocks = torch.nn.ModuleList(
          [DiscreteTimeResidualBlock(d_model=d_model) for _ in range(n_layers)]
        )

    def forward(self, x, t):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x, t)
        return self.lin_out(x)

# %%
def train(EPOCHS, X, T=100, minval=1e-5, maxval=5e-4):
    DDM = DenoisingDiffusionModel(T=T, minval=minval, maxval=maxval)
    model = BasicDiscreteTimeModel(d_model=128, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bar = tqdm.tqdm(range(EPOCHS))
    for epoch in bar:
        model.train()
        loss = DDM.diffusion_loss(model, X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.set_description(f"Loss: {loss.item():.2f}")
    return DDM, model


# %%
# make swiss roll
X = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)[0][:, [0, 2]] / 10
X = torch.tensor(X).float()
n_samples = 1000
# X = torch.tensor([[i,n_samples-i] for i in range(n_samples)]).float()
# X = (X - X.mean(axis=0)) / X.std(axis=0)
# X0 = torch.tensor([[i,i] for i in range(n_samples)]).float()
# X0 = (X0 - X0.mean(axis=0)) / X0.std(axis=0)
plt.scatter(*X.T)
# %%
DDM, model = train(1000, X, T=100, minval=1e-5, maxval=5e-3)

# %%
X_hat = DDM.sample(model, n_samples)#, X0=X)
plt.scatter(*X_hat.T)
plt.show()
# %%
