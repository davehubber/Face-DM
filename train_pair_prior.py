import os
import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from utils_pair_prior import (
    setup_logging,
    EmbeddingDataset,
    make_pair_state,
    make_average_condition,
    maybe_drop_condition,
    permutation_invariant_pair_loss,
    get_pair_prior_model,
    get_train_noise_scheduler,
    get_inference_scheduler,
    compute_pair_embedding_metrics_over_testset,
    run_pair_image_evaluation,
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

    model = get_pair_prior_model(num_layers=args.num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * args.epochs),
    )

    train_scheduler = get_train_noise_scheduler(args)

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
            e1 = e1.to(device)
            e2 = e2.to(device)
            target_pair = make_pair_state(e1, e2)
            cond_avg = make_average_condition(e1, e2, alpha=args.cond_alpha)

            timesteps = torch.randint(
                0,
                train_scheduler.config.num_train_timesteps,
                (e1.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(target_pair)
            x_t = train_scheduler.add_noise(target_pair, noise, timesteps)

            with accelerator.accumulate(model):
                cond_for_model = maybe_drop_condition(cond_avg, args.cond_drop_prob)

                pred_pair = model(
                    hidden_states=x_t,
                    timestep=timesteps,
                    proj_embedding=cond_for_model,
                ).predicted_image_embedding

                loss, _ = permutation_invariant_pair_loss(pred_pair, target_pair)

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
                        v_e1 = v_e1.to(device)
                        v_e2 = v_e2.to(device)
                        v_target_pair = make_pair_state(v_e1, v_e2)
                        v_cond_avg = make_average_condition(v_e1, v_e2, alpha=args.cond_alpha)

                        v_timesteps = torch.randint(
                            0,
                            train_scheduler.config.num_train_timesteps,
                            (v_e1.shape[0],),
                            device=device,
                            dtype=torch.long,
                        )
                        v_noise = torch.randn_like(v_target_pair)
                        v_x_t = train_scheduler.add_noise(v_target_pair, v_noise, v_timesteps)

                        v_pred_pair = model(
                            hidden_states=v_x_t,
                            timestep=v_timesteps,
                            proj_embedding=v_cond_avg,
                        ).predicted_image_embedding

                        v_loss, _ = permutation_invariant_pair_loss(v_pred_pair, v_target_pair)
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

        if (epoch + 1) % 5 == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)

            torch.save(
                unwrapped.state_dict(),
                os.path.join(base_dir, "checkpoints", "prior_active.pt"),
            )

            ema_model.store(unwrapped.parameters())
            ema_model.copy_to(unwrapped.parameters())

            torch.save(
                unwrapped.state_dict(),
                os.path.join(base_dir, "checkpoints", "prior_ema.pt"),
            )

            ema_model.restore(unwrapped.parameters())


def evaluate(args):
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

    model = get_pair_prior_model(num_layers=args.num_layers)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.load_state_dict(
        torch.load(
            os.path.join(base_dir, "checkpoints", "prior_ema.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    sample_scheduler = get_inference_scheduler(args)

    metrics = compute_pair_embedding_metrics_over_testset(
        args=args,
        test_dataloader=test_dataloader,
        model=model,
        scheduler=sample_scheduler,
    )

    img_report = run_pair_image_evaluation(
        args=args,
        base_dir=base_dir,
        test_dataloader=test_dataloader,
        model=model,
        scheduler=sample_scheduler,
    )

    metrics_report = (
        "--- Gaussian Pair Prior Evaluation Metrics (Entire Test Set) ---\n"
        f"MSE Target: {metrics['mse_target']:.6f}\n"
        f"MSE Deduced: {metrics['mse_deduced']:.6f}\n"
        f"MSE Mean: {metrics['mse_mean']:.6f}\n"
        f"Cosine Similarity Target: {metrics['cos_target']:.6f}\n"
        f"Cosine Similarity Deduced: {metrics['cos_deduced']:.6f}\n"
        f"Cosine Similarity Mean: {metrics['cos_mean']:.6f}\n"
        f"{img_report}"
    )

    print(f"\n{metrics_report}")
    with open(os.path.join(base_dir, "results", "gaussian_pair_prior_eval.txt"), "w") as f:
        f.write(metrics_report)



def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to precomputed embeddings directory")
    parser.add_argument("--image_dir1", type=str, required=True, help="Path to original images for dataset 1")
    parser.add_argument("--image_dir2", type=str, required=True, help="Path to original images for dataset 2")
    parser.add_argument("--run_name", type=str, required=True, help="Experiment name")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_layers", type=int, default=12)

    parser.add_argument("--cond_alpha", type=float, default=0.5, help="Weight used to build the conditioning average")
    parser.add_argument("--cond_drop_prob", type=float, default=0.0, help="Probability of dropping the average conditioning during training for CFG")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale at sampling/eval time")

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta; 0.0 gives deterministic sampling")
    parser.add_argument("--eval_seed", type=int, default=0)

    args = parser.parse_args()

    train(args)
    evaluate(args)


if __name__ == "__main__":
    launch()
