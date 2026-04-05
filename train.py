import os
import torch
import math
import wandb
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from utils import (
    setup_logging,
    EmbeddingDataset,
    ColdDiffusionEmbeds,
    get_prior_model,
    get_decoder_pipeline,
    encode_pil_images_to_embeddings,
    run_image_evaluation,
    compute_embedding_metrics_over_testset,
)

def train(args):
    base_dir = setup_logging(args.run_name)
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    device = accelerator.device

    train_dataset = EmbeddingDataset(
        data_dir=args.data_dir,
        image_dir1=args.image_dir1,
        image_dir2=args.image_dir2,
        partition="train",
    )

    test_dataset = EmbeddingDataset(
        data_dir=args.data_dir,
        image_dir1=args.image_dir1,
        image_dir2=args.image_dir2,
        partition="test",
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    model = get_prior_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * args.epochs),
    )

    diffusion = ColdDiffusionEmbeds(alpha_max=args.alpha_max, device=device)
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)

    ema_model = EMAModel(model.parameters(), inv_gamma=1.0, power=0.75, decay=0.9999)
    ema_model.to(device)

    if accelerator.is_main_process:
        wandb.init(project="Face-DM", name=args.run_name, config=vars(args))

    global_step = 0
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    for epoch in range(args.epochs):
        model.train()

        for e1, e2, _, _ in train_dataloader:
            t = diffusion.sample_timesteps(e1.shape[0], max_t=init_timestep)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_embeds(e1, e2, t)
                superimposed = e1 * (1.0 - args.alpha_init) + e2 * args.alpha_init

                predicted_emb = model(
                    hidden_states=x_t,
                    timestep=t,
                    proj_embedding=superimposed,
                ).predicted_image_embedding.squeeze(1)

                loss = F.mse_loss(predicted_emb, e1)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

            global_step += 1

            if global_step % 100 == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0

                with torch.no_grad():
                    for v_e1, v_e2, _, _ in test_dataloader:
                        val_t = diffusion.sample_timesteps(v_e1.shape[0], max_t=init_timestep)
                        val_x_t = diffusion.mix_embeds(v_e1, v_e2, val_t)
                        val_sup = v_e1 * (1.0 - args.alpha_init) + v_e2 * args.alpha_init

                        val_pred = model(
                            hidden_states=val_x_t,
                            timestep=val_t,
                            proj_embedding=val_sup,
                        ).predicted_image_embedding.squeeze(1)

                        v_loss = F.mse_loss(val_pred, v_e1)
                        v_loss = accelerator.gather(v_loss).mean()
                        val_loss += v_loss.item()
                        val_steps += 1

                avg_val_loss = val_loss / max(val_steps, 1)

                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "val_loss": avg_val_loss,
                            "step": global_step,
                            "epoch": epoch,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

                model.train()

        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)

            torch.save(
                unet.state_dict(),
                os.path.join(base_dir, "checkpoints", "prior_active.pt"),
            )

            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())

            torch.save(
                unet.state_dict(),
                os.path.join(base_dir, "checkpoints", "prior_ema.pt"),
            )

            ema_model.restore(unet.parameters())

def eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    dataset = EmbeddingDataset(
        data_dir=args.data_dir,
        image_dir1=args.image_dir1,
        image_dir2=args.image_dir2,
        partition="test",
    )
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = get_prior_model()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.load_state_dict(
        torch.load(
            os.path.join(base_dir, "checkpoints", "prior_ema.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    diffusion = ColdDiffusionEmbeds(alpha_max=args.alpha_max, device=device)

    metrics = compute_embedding_metrics_over_testset(
        args, test_dataloader, model, diffusion, mode="Regular"
    )

    img_report = run_image_evaluation(
        args, base_dir, test_dataloader, model, diffusion, mode="Regular"
    )

    metrics_report = (
        "--- Regular Evaluation Metrics (Entire Test Set) ---\n"
        f"MSE Target: {metrics['mse_target']:.6f}\n"
        f"MSE Deduced: {metrics['mse_deduced']:.6f}\n"
        f"MSE Mean: {metrics['mse_mean']:.6f}\n"
        f"Cosine Similarity Target: {metrics['cos_target']:.6f}\n"
        f"Cosine Similarity Deduced: {metrics['cos_deduced']:.6f}\n"
        f"Cosine Similarity Mean: {metrics['cos_mean']:.6f}\n"
        f"{img_report}"
    )

    print(f"\n{metrics_report}")
    with open(os.path.join(base_dir, "results", "final_eval.txt"), "w") as f:
        f.write(metrics_report)

def one_shot_eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    dataset = EmbeddingDataset(
        data_dir=args.data_dir,
        image_dir1=args.image_dir1,
        image_dir2=args.image_dir2,
        partition="test",
    )
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = get_prior_model()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.load_state_dict(
        torch.load(
            os.path.join(base_dir, "checkpoints", "prior_ema.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    diffusion = ColdDiffusionEmbeds(alpha_max=args.alpha_max, device=device)

    metrics = compute_embedding_metrics_over_testset(
        args, test_dataloader, model, diffusion, mode="OneShot"
    )

    img_report = run_image_evaluation(
        args, base_dir, test_dataloader, model, diffusion, mode="OneShot"
    )

    metrics_report = (
        "--- One-Shot Evaluation Metrics (Entire Test Set) ---\n"
        f"MSE Target: {metrics['mse_target']:.6f}\n"
        f"MSE Deduced: {metrics['mse_deduced']:.6f}\n"
        f"MSE Mean: {metrics['mse_mean']:.6f}\n"
        f"Cosine Similarity Target: {metrics['cos_target']:.6f}\n"
        f"Cosine Similarity Deduced: {metrics['cos_deduced']:.6f}\n"
        f"Cosine Similarity Mean: {metrics['cos_mean']:.6f}\n"
        f"{img_report}"
    )

    print(f"\n{metrics_report}")
    with open(os.path.join(base_dir, "results", "one_shot_eval.txt"), "w") as f:
        f.write(metrics_report)

def test_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting decoder sanity check on {device}...")

    from PIL import Image

    dataset = EmbeddingDataset(
        data_dir=args.data_dir,
        image_dir1=args.image_dir1,
        image_dir2=args.image_dir2,
        partition="test",
    )

    _, _, path_1, path_2 = dataset[0]

    real_img1 = Image.open(path_1).convert("RGB")
    real_img2 = Image.open(path_2).convert("RGB")

    print("Loading matched UnCLIP pipeline...")
    pipeline = get_decoder_pipeline(device)

    print("Encoding the real images with the pipeline's own encoder...")
    with torch.no_grad():
        reencoded = encode_pil_images_to_embeddings([real_img1, real_img2], pipeline, device)
        emb_1 = reencoded[0:1]
        emb_2 = reencoded[1:2]

    print("Decoding the re-encoded embeddings...")
    with torch.no_grad():
        dec_img_1 = pipeline(
            prompt=[""],
            image_embeds=emb_1.to(dtype=pipeline.unet.dtype),
            noise_level=0,
            num_inference_steps=20,
        ).images[0]

        dec_img_2 = pipeline(
            prompt=[""],
            image_embeds=emb_2.to(dtype=pipeline.unet.dtype),
            noise_level=0,
            num_inference_steps=20,
        ).images[0]

    def process_to_64(img):
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
        img = img.to(device)
        return TF.resize(img, [64, 64], antialias=True)

    real_1_t = process_to_64(real_img1)
    dec_1_t = process_to_64(dec_img_1)
    real_2_t = process_to_64(real_img2)
    dec_2_t = process_to_64(dec_img_2)

    row_grid = torch.cat([real_1_t, dec_1_t, real_2_t, dec_2_t], dim=2)

    base_dir = os.path.join("experiments", args.run_name, "results")
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, "decoder_single_image_sanity_check_64x64.png")

    save_image(row_grid, save_path)
    print(f"Saved sanity-check row to: {save_path}")

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to precomputed embeddings directory', required=True)
    parser.add_argument("--image_dir1", type=str, required=True, help="Path to the folder containing the first dataset's original images")
    parser.add_argument("--image_dir2", type=str, required=True, help="Path to the folder containing the second dataset's original images")
    parser.add_argument('--run_name', help='Name of the experiment', required=True)
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Max alpha weight', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Init alpha weight', required=False)
    parser.add_argument('--batch_size', default=1024, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=1e-4, help='Learning rate', type=float, required=False)

    args = parser.parse_args()
    train(args)
    one_shot_eval(args)
    eval(args)
    test_decoder(args)


if __name__ == '__main__':
    launch()
