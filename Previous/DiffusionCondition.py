
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('snr', alphas_bar / torch.sqrt(1. - alphas_bar))
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels, get_eps=False):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        if get_eps:
            return xt_prev_mean, var, eps
        else:
            return xt_prev_mean, var

    def forward(self, x_T, labels, tracking_mode=False):
        x_t = x_T
        tracking_images_array = []
        tracking_images_array2 = []
        x_hat_tensor = torch.zeros([x_T.shape[0], (self.T // 250), 3, 32, 32], dtype=x_T.dtype, device=x_T.device)
        x_t_tensor = torch.zeros_like(x_hat_tensor)
        tracking_tensors_array = x_t_tensor
        time = 0

        # if tracking_mode:
            # x_hat_tensor = torch.zeros([x_T.shape[0], self.T, 3, 32, 32], dtype=x_T.dtype, device=x_T.device)
            # x_t_tensor = torch.zeros_like(x_hat_tensor)
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * time_step
            if tracking_mode:
                mean, var, eps = self.p_mean_variance(x_t=x_t, t=t, labels=labels, get_eps=True)
                alpha_bar_t = extract(self.alphas_bar, t, x_t.shape)
                x_hat = 1 / torch.sqrt(alpha_bar_t) * (x_t - torch.sqrt(1 - alpha_bar_t) * eps)
                # x_hat_tensor[:, time_step] = x_hat
                # x_t_tensor[:, time_step] = x_t
            else:
                mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            noise = torch.randn_like(x_t) if time_step > 0 else 0
            x_t = mean + torch.sqrt(var) * noise
            if time_step % 250 == 0:
                tracking_images_array.append(torch.clip(x_t, -1, 1))
                tracking_tensors_array[:, time] = x_t
                time += 1
                print("Sampling time", time_step, "finished.")
            if time_step < 500:
                tracking_images_array2.append(torch.clip(x_t, -1, 1))
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        tracking_images_array.append(torch.clip(x_t, -1, 1))
        return (tracking_images_array, tracking_images_array2, tracking_tensors_array, x_hat_tensor, x_t_tensor, self.snr, self.alphas_bar) if tracking_mode else torch.clip(x_0, -1, 1)

