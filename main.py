import torch
from diffusers import StableDiffusionPipeline
import os
import torch.nn as nn

model_id = "CompVis/stable-diffusion-v1-4"  # 원하는 모델 ID로 변경
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

def simulate_BM(x, gamma_list):
    """
    Simulate a single Brownian motion path.

    Args:
        x (torch.Tensor): Input data tensor.
        gamma_list (torch.Tensor): List of SNR values.

    Returns:
        tuple: W_gamma, delta_W, z_gamma tensors.
    """
    gamma_list = gamma_list.clone().detach().to(x.device)
    delta_gamma = torch.diff(gamma_list).to(x.device)
    delta_W = torch.randn((len(delta_gamma),) + x.shape, device=x.device) * torch.sqrt(delta_gamma).view(-1, 1, 1, 1).to(x.device)
    W_initial = torch.randn_like(x) * torch.sqrt(gamma_list[0]).to(x.device)
    W_gamma = torch.cat([W_initial.unsqueeze(0), torch.cumsum(delta_W, dim=0)], dim=0).to(x.device)
    z_gamma = gamma_list.view(-1, 1, 1, 1).to(x.device) * x + W_gamma
    return W_gamma, delta_W, z_gamma

def get_intermediate_pointwise_mutual_info(img, label, model_num, t, w=1.0, pipeline=None):
    """
    Calculate the intermediate PMI using a pre-trained Stable Diffusion model.

    Args:
        img (torch.Tensor): Input image tensor.
        label (int): Label for conditioning.
        model_num (int): Model number identifier.
        t (int): Intermediate timestep.
        w (float, optional): Weight parameter. Defaults to 1.0.
        pipeline (StableDiffusionPipeline): Loaded Stable Diffusion pipeline.

    Returns:
        tuple: Standard integral and total PMI values.
    """
    if pipeline is None:
        raise ValueError("Stable Diffusion pipeline must be provided")

    with torch.no_grad():
        device = img.device
        model = pipeline.unet  # Access the UNet component
        model.eval().to(device)

        # Initialize integrals
        standard_integral = 0.0
        ito_integral = 0.0

        # DDPM parameters
        betas = torch.linspace(1e-4, 0.028, 4000).double().to(device)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0).to(device)

        # Compute alpha_bar from timestep t
        alphas_bar_from_t = torch.cumprod(alphas[t:], dim=0).to(device)
        # Signal-to-Noise Ratio (SNR)
        snrs = (alphas_bar_from_t / (1. - alphas_bar_from_t))

        # Simulate Brownian Motion from timestep t
        W_gamma, delta_Ws, z_gammas = simulate_BM(img, torch.flip(snrs, [0]))
        W_gamma = torch.flip(W_gamma, [0])
        delta_Ws = torch.flip(delta_Ws, [0])
        delta_Ws = torch.cat((torch.zeros((1, *img.shape), device=device), delta_Ws), dim=0)
        z_gammas = torch.flip(z_gammas, [0])

        # Scale z_gamma to get x_t
        alphas_bar_from_t_reshaped = alphas_bar_from_t[:, None, None, None].to(torch.float32)
        z_gammas = z_gammas.to(torch.float32)
        x_ts = z_gammas * (1 - alphas_bar_from_t_reshaped) / torch.sqrt(alphas_bar_from_t_reshaped)

        size = 4000 - t

        # Process in batches to limit memory usage
        batch_size = 1000
        for i in range((size - 1) // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size - 1, size - 1)

            # Slice tensors appropriately
            snr = snrs[start_idx:end_idx + 1]
            x_t = x_ts[start_idx:end_idx + 1]
            delta_W = delta_Ws[start_idx:end_idx + 1]

            original_t_values = torch.arange(start_idx + t, end_idx + t + 1, 1).to(device)

            # Prepare labels
            labels_tensor = torch.full((end_idx - start_idx + 1,), label, dtype=torch.int).to(device)

            # Inference with conditional UNet
            # Stable Diffusion's UNet typically expects timestep embeddings
            # and optional conditioning. Here, we'll assume label is used for conditioning.
            
            # Generate timestep embeddings
            timesteps = original_t_values
            # Note: Adjust the timestep embedding if necessary based on the model's implementation

            # Forward pass (conditional)
            # Ensure the UNet's forward method aligns with how it's called here
            # For diffusers, UNet's forward might require additional parameters
            # Adjust accordingly
            # Example for diffusers' UNet:
            # latent model expects inputs in the form (batch, channels, height, width)
            # and timesteps as integers or embeddings

            # Conditional prediction
            # In diffusers, the UNet's forward method signature might look like:
            # def forward(self, sample, timestep, encoder_hidden_states)
            # Adjust based on actual implementation

            # Here, we'll assume unconditional for simplicity
            # Modify if you have a conditional setup
            # If using classifier-free guidance, handle both conditional and unconditional passes

            # Example for unconditional:
            eps = model(x_t, timesteps).sample  # Replace with actual attribute/method
            # If using conditional:
            # concatenated_x = torch.cat([x_t, x_t], dim=0)
            # concatenated_timesteps = torch.cat([timesteps, timesteps], dim=0)
            # conditional_eps = model(concatenated_x, concatenated_timesteps).sample
            # Assuming labels are handled inside the model or need to be concatenated similarly

            # Placeholder for non-conditional prediction
            nonEps = eps  # Replace with actual unconditional prediction if available

            # Combine predictions
            eps_comb = (1. + w) * eps - w * nonEps

            # Compute coefficients
            alpha_bar = alphas_bar[start_idx + t : end_idx + t + 1]
            alpha_bar_reshaped = alpha_bar[:, None, None, None].to(torch.float32)

            alpha_bar_from_t_slice = alphas_bar_from_t[start_idx:end_idx + 1]
            alpha_bar_from_t_reshaped = alpha_bar_from_t_slice[:, None, None, None].to(torch.float32)

            coeff1 = torch.sqrt(1.0 / alpha_bar_from_t_reshaped)
            coeff2 = (1 - alpha_bar_from_t_reshaped) / torch.sqrt(1 - alpha_bar_reshaped)

            # MMSE estimates of x_t
            x_hat_cond = coeff1 * (x_t - coeff2 * eps_comb)
            x_hat_uncond = coeff1 * (x_t - coeff2 * nonEps)

            # Reshape input image for difference calculation
            img_reshaped = img.unsqueeze(0)  # Shape: (1, C, H, W)

            # Compute differences
            diff_unconditional = img_reshaped - x_hat_uncond
            diff_tensor = img_reshaped - x_hat_cond

            # Compute squared L2 norms
            squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
            squared_l2_tensor = (diff_tensor ** 2).sum(dim=(1, 2, 3))

            # Difference in squared norms
            difference = squared_l2_unconditional - squared_l2_tensor

            # Compute SNR differences
            snr_diff = -1 * (snr[1:] - snr[:-1])

            # Accumulate standard integral
            standard_integral += torch.sum(difference[1:] * snr_diff, dim=0)

            # Compute Ito integral
            difference = x_hat_cond - x_hat_uncond
            ito_integral += (difference * delta_W).sum()

    return 0.5 * standard_integral, 0.5 * standard_integral + ito_integral

# Prepare your image and label
# For demonstration, we'll use a random tensor. Replace this with your actual image tensor.
# Ensure the image tensor is normalized as per Stable Diffusion's requirements.
img = torch.randn(3, 512, 512).to("cuda")  # Example for a 512x512 image

label = 0  # Replace with your actual label
model_num = 0  # Not used in this context
t = 1000  # Example timestep
w = 1.0  # Weight parameter

# Compute PMI
standard_integral, total_pmi = get_intermediate_pointwise_mutual_info(
    img=img,
    label=label,
    model_num=model_num,
    t=t,
    w=w,
    pipeline=pipeline
)

print("Standard Integral:", standard_integral.item())
print("Total PMI:", total_pmi.item())
