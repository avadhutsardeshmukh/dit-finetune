import numpy as np
import argparse
import tqdm
import torch, torchvision
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from PIL import Image
from torch_utils import dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_ddpm(diffusion_model, image_size, noise_scheduler, num_classes, num_samples_per_class):
    """
    Sample from a (trained) DDPM diffusion model. 
    :param diffusion_model : The pretrained diffusion model
    :param image_size : Size of the training images
    :param noise_scheduler : The noise scheduler used by the diffusion model
    :param num_classes : Number of different class labels for conditional generation
    :param num_samples_per_class : The no of samples to be generated for each class

    :return A grid image (PIL) of the generated samples
    """
    labels = [l for l in range(num_classes) for _ in range(num_samples_per_class)]  
    labels = torch.tensor(labels, device = device).long()
    z = torch.randn(
            num_samples_per_class,
            3,
            image_size,
            image_size,
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
        #mean_noise_pred = noise_pred[:,:4,:,:]
        z = noise_scheduler.step(noise_pred, t, z).prev_sample
        print('One dnoising step completed')

    print('Sampling done. Got latents {} of shape {}'.format(z, z.shape))
    #Make a grid with num_samples_per_class images in each row, one row per class    
    grid = torchvision.utils.make_grid(z,nrow=num_classes)
    im = grid.permute(1,2,0).cpu().clip(-1,1)*0.5 + 0.5
    im = Image.fromarray(np.array(im*255).astype(np.uint8))
    return im


def finetune_ddpm(
        image_size=128,
        #noise_scheduler_model = 'google/ddpm-celebahq-256',
        part_type='hazelnut', 
        num_epochs=1,
        lr=1e-5, 
        batch_size=2,
        grad_acc_steps=2,
        wandb_project='defectDDPM_finetune',
        ckpt_every=100,
        ckpt_dir='.',
        log_samples_every=10):
    """
    Fine tune a latent diffusion DiT model with a defect images of a particular part type from DefectSpectrum data. 
    :param image_size : The input image size (default 128x128)
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

    #The base DDPM model.
    diffusion_model = UNet2DModel.from_pretrained(
        'RichardMX/conditional_ddpm-butterflies-128'
        )
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

            #Standard Gaussian noise to be added to each clean image
            noise = torch.randn(train_images.shape).to(device)
            #Sample a timestep t uniformly for each real image in the batch
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, 
                                      (batch_size,),
                                      device=train_images.device).long()
            #Add the noise, scaled with appropriate variance for that timestep (according to scheduler), to corresponding images
            noisy_images = scheduler.add_noise(train_images, noise, timesteps)

            #Predict the added noise from the clean images
            #The DiT model outputs two tensors - the predicted noise  and the diagonal covariance matrix
            #Both of shape (patchXpatchXchannel), stacked together. We separate the mean predicted noise 
            with torch.autocast(device_type=device):
                ##Mixed precision training for memory efficiency
                noise_pred = diffusion_model(noisy_images, timesteps,train_labels,return_dict=False)[0]
                print("Noise predictions with shape ", noise_pred.shape)
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
                im = sample_ddpm(
                        diffusion_model,  
                        image_size, scheduler, 
                        num_classes, num_samples_per_class
                        )
                save_path = f"ddpm_samples_step_{step}_epoch_{epoch}.png"
                torchvision.save_image(im, save_path)
                #wandb.log({'Sample generations': wandb.Image(im)})
            # Save a checkpoint every 'ckpt_every' steps
            if (step+1)%ckpt_every == 0:
                checkpoint_path=f"{ckpt_dir}/ddpm_checkpoint_step_{step+1}"
                checkpoint = {
                        "model" : diffusion_model.state_dict(),
                        "optimizer" : optimizer.state_dict()
                        }
                torch.save(checkpoint, checkpoint_path)

    print(f"Epoch {epoch}, Average Loss {sum(losses[-len(dataloader):])/len(dataloader)}")
    return losses