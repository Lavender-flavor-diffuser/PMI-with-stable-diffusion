import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# 중간 이미지 텐서를 저장하는 함수 추가
def save_intermediate(intermediate, path, count):
    torch.save(intermediate, os.path.join(path, f"intermediate_{count:05}.pt"))

# 최종 이미지 텐서를 저장하는 함수 추가
def save_final_tensor(tensor, path, count):
    torch.save(tensor, os.path.join(path, f"final_{count:05}.pt"))

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def simulate_BM(x, gamma_list):
    # Given a data and list of snr, simulates a single Brownian motion path
    # z_gamma = gamma * x + W_gamma, W_gamma ~ N(0, gamma)

    gamma_list = gamma_list.clone().detach().to(x.device)
    delta_gamma = torch.diff(gamma_list).to(x.device)
    delta_W = torch.randn((len(delta_gamma),) + x.shape, device=x.device) * torch.sqrt(delta_gamma).view(-1, 1, 1, 1).to(x.device)
    W_initial = torch.randn_like(x) * torch.sqrt(gamma_list[0]).to(x.device)
    W_gamma = torch.cat([W_initial.unsqueeze(0), torch.cumsum(delta_W, dim=0)], dim=0).to(x.device)
    z_gamma = gamma_list.view(-1, 1, 1, 1).to(x.device) * x + W_gamma
    return W_gamma, delta_W, z_gamma

def get_intermediate_pointwise_mutual_info(latent, prompt, t, w=1.0):
    with torch.no_grad():
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Stable Diffusion components
        model_id = "CompVis/stable-diffusion-v1-4"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        
        # Initialize two terms
        standard_integral = 0.0
        ito_integral = 0.0

        # Get alphas and betas from scheduler
        betas = torch.linspace(1e-4, 0.028, 1000).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim = 0).to(device)

        # alpha bar calculated from t (intermediate timestep)
        alphas_bar_from_t = torch.cumprod(alphas[t:], dim = 0).to(device)
        # snr when we consider input at timestep t
        snrs = alphas_bar_from_t / (1 - alphas_bar_from_t)

        # Simulate BM from t taking account of new snr values above
        W_gamma, delta_Ws, z_gamma = simulate_BM(latent, torch.flip(snrs, [0]))
        W_gamma = torch.flip(W_gamma, [0])
        delta_Ws = torch.flip(delta_Ws, [0])
        delta_Ws = torch.cat((torch.zeros((1,) + latent.shape).to(device), delta_Ws), dim=0)
        z_gamma = torch.flip(z_gamma, [0])

        # Compute x_t from z_gamma
        alphas_bar_from_t_reshaped = alphas_bar_from_t[:, None, None, None].to(torch.float32)
        z_gamma = z_gamma.to(torch.float32)
        x_ts = z_gamma * (1 - alphas_bar_from_t_reshaped) / torch.sqrt(alphas_bar_from_t_reshaped)

        num_timesteps = 1000
        size = num_timesteps - t
        max_batch_size = 10  # Adjust as needed

        for i in range((size - 1) // max_batch_size + 1):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size - 1, size - 1)
            # Truncate appropriately
            batch_snr = snrs[start_idx : end_idx + 1]
            batch_x_t = x_ts[start_idx : end_idx + 1]
            batch_delta_W = delta_Ws[start_idx : end_idx + 1]

            # Original t values
            original_t_values = torch.arange(start_idx + t, end_idx + t + 1, 1).to(device)

            batch_size = batch_x_t.shape[0]

            # Get text embeddings
            text_inputs = tokenizer(
                [prompt] * batch_size,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

            # For unconditional generation, create empty text embeddings
            uncond_input = tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

            # Inference
            # Conditional prediction
            eps = unet(
                batch_x_t, original_t_values, encoder_hidden_states=text_embeddings
            ).sample
            # Unconditional prediction
            nonEps = unet(
                batch_x_t, original_t_values, encoder_hidden_states=uncond_embeddings
            ).sample
            eps_comb = (1. + w) * eps - w * nonEps
            
            # Calculate coefficients
            alpha_bar_batch = alphas_bar[start_idx + t : end_idx + t + 1]
            alpha_bar_batch_reshaped = alpha_bar_batch[:, None, None, None, None].to(torch.float32)
            
            alpha_bar_from_t_batch = alphas_bar_from_t[start_idx : end_idx + 1]
            alpha_bar_from_t_batch_reshaped = alpha_bar_from_t_batch[:, None, None, None].to(torch.float32)
            
            coeff_lambda_inverse = torch.sqrt(1 / alpha_bar_from_t_batch_reshaped)
            coeff_sigma_square = (1 - alpha_bar_from_t_batch_reshaped)/ torch.sqrt(1 - alpha_bar_batch_reshaped)
            
            # Calculate mmse estimates of x_t
            x_hat_uncond = coeff_lambda_inverse * (batch_x_t + coeff_sigma_square * nonEps)
            x_hat_cond = coeff_lambda_inverse * (batch_x_t + coeff_sigma_square * eps_comb)
            
            # Calculating standard integral
            latent_reshaped = latent[None, :, :, :]
            
            diff_unconditional = latent_reshaped - x_hat_uncond
            diff_conditional = latent_reshaped - x_hat_cond

            squared_l2_unconditional = (diff_unconditional ** 2).sum(dim=(1, 2, 3))
            squared_l2_conditional = (diff_conditional ** 2).sum(dim=(1, 2, 3))
            
            integrand_in_standard_integral = squared_l2_unconditional - squared_l2_conditional 
            snr_diff = -1 * (batch_snr[1:] - batch_snr[:-1]) # multiply -1 to make positive
            snr_diff = snr_diff.unsqueeze(1)
            riemann_sum = integrand_in_standard_integral[1:] * snr_diff
            
            standard_integral += torch.sum(riemann_sum)

            # Calculating ito integral
            integrand_in_ito_integral = x_hat_cond[:, :, :, :] - x_hat_uncond[:, :, :, :]
            ito_integral += (integrand_in_ito_integral * batch_delta_W).sum()
        return 0.5 * standard_integral, 0.5 * standard_integral + ito_integral
    
def get_est(Img, prompt, t, iter=5):
    result = 0
    for _ in range(iter):
        _, est = get_intermediate_pointwise_mutual_info(Img, prompt, t)
        result += est
    return result/iter

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="Ann Graham Lotz",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20, ## 나중에 20으로 수정해야함
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--log_every_t",
        type=int,
        help="determine the intermediate save period (Default : 1)",
        default=1
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    all_intermediates = []  # To store all intermediate tensors

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, intermediate = sampler.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                            log_every_t=opt.log_every_t
                        )
                        all_intermediates.append(intermediate)  # Store intermediate tensors

                        # All tensors are assumed to be on CPU
                        est_cpu = [tensor.cpu().numpy() for tensor in intermediate['x_inter']]

                        # Convert to NumPy array
                        est_array = np.array(est_cpu)

                        print(f"est_array의 shape: {est_array.shape}")

                        ## Decoding Latent Representations to Images (이전까지는 latent space 상에서의 tensor)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        # Save intermediate tensors
                        save_intermediate(intermediate, sample_path, base_count)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # Additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # To image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    # PMI Calculation : time(51) * batch_size(--n_sample에 의해 조정) * Channel(4) * Height(64) * Width(64)
    est_list_list = []  # Stores PMI for all samples

    for iter_num in range(opt.n_iter):        
        # Retrieve the intermediate dict for the current sample
        # 
        intermediate_dict = all_intermediates[iter_num]
        
        # Access the list of intermediate tensors
        # Replace 'x_inter' with the correct key if it's different
        x_inter_list = intermediate_dict['x_inter']
        
        for sample_num in range(opt.n_samples):
            est_list = []
            # Iterate over each intermediate tensor
            for time_idx, img_tensor in enumerate(x_inter_list):
                print(f'current_time : {time_idx}')
                if(time_idx == 0) : continue 
                # Calculate the corresponding timestep
                t = 1000 - time_idx * (1000 // opt.ddim_steps)
                
                # Call get_est with the correct arguments
                est = get_est(img_tensor[sample_num], opt.prompt, t).cpu()
                est_list.append(est)
        
            # Convert est_list to a NumPy array and append to est_list_list
            est_array = np.array(est_list)
            est_list_list.append(est_array)
            
            # Save PMI for the current sample
            torch.save(est_array, f'outputs/PMI_query_{opt.prompt}_iter_num_{iter_num}_sample_num_{sample_num}.pt')
            print(f"PMI index {sample_num} finished.")

        # Convert the entire PMI list to a tensor and save
        est_list_list_tensor = torch.tensor(est_list_list)
        torch.save(est_list_list_tensor, f'outputs/PMI_query_{opt.prompt}_iter_num_{iter_num}.pt')
        print("Saved PMI indices.")

if __name__ == "__main__":
    main()