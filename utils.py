import os
import torch
import math
import numpy as np
import torchvision
from torch.utils.data import Dataset
from diffusers import PriorTransformer
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def setup_logging(run_name):
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "eval"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "one_shot"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir

def save_images(images, obtained_images, original_images, added_images, path, **kwargs):
    input_images = (original_images * 0.5 + added_images * 0.5).type(torch.uint8)
    B, C, H, W = original_images.shape
    
    all_images = torch.stack([
        original_images, 
        added_images, 
        input_images, 
        images, 
        obtained_images
    ], dim=1).view(-1, C, H, W)
    
    grid = torchvision.utils.make_grid(all_images, nrow=5, padding=2, pad_value=255, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)
    
    from PIL import Image
    im = Image.fromarray(ndarr)
    im.save(path)

class EmbeddingDataset(Dataset):
    def __init__(self, data_dir, partition='train', scale_factor=17.2):
        self.scale_factor = scale_factor
        self.embeds1 = torch.load(os.path.join(data_dir, f"{partition}_dataset1_embeds.pt"))
        self.embeds2 = torch.load(os.path.join(data_dir, f"{partition}_dataset2_embeds.pt"))
        
        self.embeds1 = self.embeds1 * self.scale_factor
        self.embeds2 = self.embeds2 * self.scale_factor
            
    def __getitem__(self, index):
        return self.embeds1[index], self.embeds2[index]
    
    def __len__(self):
        return len(self.embeds1)

class ColdDiffusionEmbeds:
    def __init__(self, max_timesteps=250, alpha_start=0., alpha_max=0.5, device="cuda"):
        self.max_timesteps = max_timesteps
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps

    def mix_embeds(self, emb_1, emb_2, t):
        w1 = (1. - self.alteration_per_t * t).unsqueeze(1)
        w2 = (self.alteration_per_t * t).unsqueeze(1)
        return emb_1 * w1 + emb_2 * w2

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, superimposed_emb, alpha_init=0.5):
        n = len(superimposed_emb)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()
        with torch.no_grad():
            x_t = superimposed_emb.clone()

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_emb = model(hidden_states=x_t, timestep=t, proj_embedding=superimposed_emb).predicted_image_embedding
                other_emb = (superimposed_emb - (1. - alpha_init) * predicted_emb) / alpha_init
                x_t = x_t - self.mix_embeds(predicted_emb, other_emb, t) + self.mix_embeds(predicted_emb, other_emb, t-1)
        
        model.train()
        other_emb = (superimposed_emb - (1 - alpha_init) * x_t) / alpha_init
        return x_t, other_emb

def get_prior_model():
    return PriorTransformer(
        num_embeddings=1,
        embedding_dim=768,
        num_attention_heads=12,
        num_layers=12,
        project_dim=768,
    )

def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added

def decode_batch_to_images(embeds, decoder, scale_factor):
    embeds_unscaled = embeds / scale_factor
    images = decoder(image_embeds=embeds_unscaled.to(torch.float16), num_inference_steps=25, output_type="pt").images
    images_lpips = (images * 2) - 1
    images_uint8 = (images * 255).clamp(0, 255).type(torch.uint8)
    return images_lpips, images_uint8