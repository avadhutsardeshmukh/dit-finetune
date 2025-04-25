import torch
import argparse
import finetuning_ddpm, finetuning_dit
#####################################
# The main function for finetuneing #
#####################################
def main(args):
    torch.manual_seed(args.random_seed)
    if (args.model == 'DiT'):
        losses = finetuning_dit.finetune_dit(
                full_fine_tune=False,
                image_size=args.image_size,
                part_type=args.part_type, 
                num_epochs=args.epochs,
                lr=args.learning_rate, 
                batch_size=args.batch_size,
                grad_acc_steps=args.gradient_accumulation_steps,
                ckpt_every=args.save_checkpoints_every,
                log_samples_every=args.save_samples_every
            )
    elif (args.model == 'DDPM'):
        print("Finetuning a DDPM pipeline")
        losses = finetuning_ddpm.finetune_ddpm(
                image_size=args.image_size,
                part_type=args.part_type, 
                num_epochs=args.epochs,
                lr=args.learning_rate, 
                batch_size=args.batch_size,
                grad_acc_steps=args.gradient_accumulation_steps,
                ckpt_every=args.save_checkpoints_every,
                log_samples_every=args.save_samples_every
            )
    print(losses)

if __name__ == "__main__":
    models = ['DiT', 'StableDiffusion', 'DDPM']
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=models, default='DiT')
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--part_type", type=str, default='hazelnut')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_checkpoints_every", type=int, default=1)
    parser.add_argument("--save_samples_every", type=int, default=1)
    args = parser.parse_args()
    main(args)



