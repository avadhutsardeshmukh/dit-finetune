import numpy as np
import argparse
import tqdm
import torch, torchvision
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiTTransformer2DModel, AutoencoderKL
from matplotlib import pyplot as plt
from PIL import Image
from torch_utils import dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_diffusion(diffusion_model, vae_model, vae_latent_size, noise_scheduler, num_classes, num_samples_per_class):
    """
    Sample from a (trained) latent diffusion model. 
    :param diffusion_model : The pretrained latent diffusion model
    :param vae_model : The pretrained VAE model to map images to latent space
    :param vae_latent_size : Size of the VAE latent - also the input size of diffusion model
    :param noise_scheduler : The noise scheduler used by the diffusion model
    :param num_classes : Number of different class labels for conditional generation
    :param num_samples_per_class : The no of samples to be generated for each class

    :return A grid image (PIL) of the generated samples
    """
    labels = [l for l in range(num_classes) for _ in range(num_samples_per_class)]  
    labels = torch.tensor(labels, device = device).long()
    z = torch.randn(
            num_samples_per_class,
            vae_model.config.latent_channels,
            vae_latent_size,
            vae_latent_size,
            device=device
            )
    #Replicate the noise vectors for all class labels
    z = z.repeat(num_classes, 1,1,1)

    #Sampling loop with the DDIMSampler
    for i, t in tqdm.tqdm(enumerate(noise_scheduler.timesteps), total=20):
        model_input = noise_scheduler.scale_model_input(z,t)
        timestep = t[None]
        #timestep = timestep.long()
        with torch.no_grad():
            #with torch.autocast(device_type='cpu'):
            noise_pred = diffusion_model(model_input,timestep,labels,return_dict=False)[0]
        mean_noise_pred = noise_pred[:,:4,:,:]
        z = noise_scheduler.step(mean_noise_pred, t, z).prev_sample
        print('One dnoising step completed')

    print('Sampling done. Got latents {} of shape {}'.format(z, z.shape))
    #Make a grid with num_samples_per_class images in each row, one row per class
    print(vae_model.config.scaling_factor)
    x = vae_model.decode(z/vae_model.config.scaling_factor).sample
    print('Finally decoded input of shape ', x.shape)
    grid = torchvision.utils.make_grid(x,nrow=num_classes)
    im = grid.permute(1,2,0).cpu().clip(-1,1)*0.5 + 0.5
    im = Image.fromarray(np.array(im*255).astype(np.uint8))
    return im


def finetune(
        image_size=256,
        #noise_scheduler_model = 'google/ddpm-celebahq-256',
        part_type='hazelnut', 
        num_epochs=1,
        lr=1e-5, 
        batch_size=4,
        grad_acc_steps=2,
        wandb_project='defectDiT_finetune',
        ckpt_every=100,
        ckpt_dir='.',
        log_samples_every=10):
    """
    Fine tune a latent diffusion DiT model with a defect images of a particular part type from DefectSpectrum data. 
    :param image_size : The input image size (default 256x256)
    :param part_type : The selected part type from DefectSpectrum (e.g, zipper, pill,.. Default hazelnut)
    :param lr : Learning rate (default 1e-5, we don't use LR decay)
    :param batch_size : Batch size (default = 4 to make it memory efficient)
    :param grad_acc_steps : Number steps for which gradient is accumulated before updating wts - to account for small batch
    :param wandb_project : The id of wandb project where intermediate models and outputs are logged (not used currently)
    :param ckpt_every : Frequency of checkpointing the model (no of steps)
    :param ckpt_dir : currently the models are saved locally, so this parameter specifies the directory
    :param log_samples_every : Frequency of saving generating samples (no of steps) 

    :return A list of average loss values per epoch
    """

    #Initialize the wandb project to log the samples and checkpoints during training
    #TODO Enable this if it works in colab, otherwise save locally
    #wandb.init(wandb_project, config=locals())

    #A fast scheduler to trade-off fidelity with sampling speed
    scheduler=DDIMScheduler.from_pretrained('google/ddpm-celebahq-256')
    scheduler.set_timesteps(num_inference_steps=20)

    #The base DiT model. This is a latent diffusion model - i.e. it operates in the latent space of a VAE
    diffusion_model = DiTTransformer2DModel.from_pretrained(
            'facebook/DiT-XL-2-256', subfolder='transformer')
    # The VAE
    vae_model = AutoencoderKL.from_pretrained(
            'facebook/DiT-XL-2-256', subfolder='vae')
    #TODO How to get this value from the model (is it vae.config.norm_num_groups)?
    vae_latent_size=image_size//8

    #Define the dataloader
    db, dl = dataloader(image_size=image_size,batch_size=batch_size,part_type=part_type)
    classes = db.classes
    num_classes = len(db.classes)
    print("Prepared the dataloader with size", len(dl))

    #Use a very small learning rate, since we have a very small dataset and a small batch size (gradients may be noisy)
    optimizer = torch.optim.AdamW(diffusion_model.parameters(),lr=lr)
    #Loss history, for posterior analysis or debugging
    losses=[]

    #Fine tuning loop
    for epoch in range(num_epochs):
        print("Training epoch", epoch)
        for step, batch in tqdm.tqdm(enumerate(dl), total=len(dl)):
            train_images = batch[0].to(device)
            train_labels = batch[1].to(device)

            #Get the latent representations of images from the VAE model - Diffusion will operate in this space
            #We are not finetuning the VAE model (its generic, learns reps for any image). so, no_grad()
            with torch.no_grad():
                image_latents = vae_model.encode(train_images).latent_dist.sample()
                image_latents = image_latents * vae_model.config.scaling_factor
            print("Got vae latents with shape ", image_latents.shape)

            #Standard Gaussian noise to be added to each clean image
            noise = torch.randn(image_latents.shape).to(device)
            #Sample a timestep t uniformly for each real image in the batch
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, 
                                      (batch_size,),
                                      device=image_latents.device).long()
            #Add the noise, scaled with appropriate variance for that timestep (according to scheduler), to corresponding images
            noisy_latents = scheduler.add_noise(image_latents, noise, timesteps)

            #Predict the added noise from the VAE latent vectors of clean images
            #The DiT model outputs two tensors - the predicted noise  and the diagonal covariance matrix
            #Both of shape (patchXpatchXchannel), stacked together. We separate the mean predicted noise 
            with torch.autocast(device_type='cpu'):
                ##Mixed precision training for memory efficiency
                noise_pred = diffusion_model(noisy_latents, timesteps,train_labels,return_dict=False)
                noise_pred = noise_pred[0]
                print("Noise predictions with shape ", noise_pred.shape)
                mean_noise_pred = noise_pred[:,:4,:,:]
                #Gradient descent on the error between the true added noise and predicted noise
                loss = F.mse_loss(mean_noise_pred, noise)
            
            losses.append(loss.item())
            loss.backward(loss)
            
            #Accumulate gradients for some steps - because small batch size might mean very small gradients
            if (step % (grad_acc_steps-1) == 0):
                optimizer.step()
                optimizer.zero_grad()

            #Save some sample generations every 'log_samples_every' steps
            #Generate a batch of 8 images per class and save as a grid
            if (step+1)%log_samples_every == 0:
                num_samples_per_class=1
                im = sample_diffusion(
                        diffusion_model, vae_model, 
                        vae_latent_size, scheduler, 
                        num_classes, num_samples_per_class
                        )
                save_path = f"samples_step_{step}_epoch_{epoch}.png"
                torchvision.save_image(im, save_path)
                #wandb.log({'Sample generations': wandb.Image(im)})
            # Save a checkpoint every 'ckpt_every' steps
            if (step+1)%ckpt_every == 0:
                checkpoint_path=f"{ckpt_dir}/checkpoint_step_{step+1}"
                checkpoint = {
                        "model" : diffusion_model.state_dict(),
                        "optimizer" : optimizer.state_dict()
                        }
                torch.save(checkpoint, checkpoint_path)

    print(f"Epoch {epoch}, Average Loss {sum(losses[-len(dataloader):])/len(dataloader)}")
    return losses

#######################
# The main function   #
#######################
def main(args):
    torch.manual_seed(args.random_seed)
    if (args.model == 'DiT'):
        losses = finetune(
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
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--part_type", type=str, default='hazelnut')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_checkpoints_every", type=int, default=1)
    parser.add_argument("--save_samples_every", type=int, default=1)
    args = parser.parse_args()
    main(args)



