import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch, torchvision
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from datasets import load_dataset
import os
from typing import Any, Callable, cast, Optional, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def has_file_allowed_extension(filename: str, extensions):
    """Checks if a file is an allowed extension. Helper function for the dataset loader
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

class ImageFolderSelectSubset(ImageFolder):
    """A custom data loader.
    It selects specified number of samples from each class. 
    If a class does not have those samples then all available samples are selected.
    Args:
        max_samples_per_class (int) : Maximum number of samples to be collected per class
        For each class, the number of samples will be min(available_samples, max_samples_per_class)
    """
    def __init__(
        self,
        root,    
        extensions = IMG_EXTENSIONS,
        transform = None,
        target_transform = None,
        is_valid_file = None,
        allow_empty = False,
        max_samples_per_class = None,
    ) -> None:
        #super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        classes, class_to_idx = self.find_classes(self.root)
        #
        self.max_samples_per_class = max_samples_per_class
        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            max_samples_per_class = self.max_samples_per_class
        )
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
    directory,
    class_to_idx = None,
    extensions = None,
    is_valid_file = None,
    allow_empty = False,
    max_samples_per_class = None,
    ) :
        """Generates a list of samples of a form (path_to_sample, class).
        See :class:`DatasetFolder` for details.
        We override this method to select only max_samples_per_class samples for each class.
        Returns : list[tuple[str, int]]
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        #samples_per_class = 1
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            #Keeps track of the number of samples (file_paths) collected for current class.
            samples_per_class = 0            
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)                        
                        ###Check if max_samples has been collected for this class.
                        samples_per_class +=1
                        if (max_samples_per_class is not None 
                        and samples_per_class >= max_samples_per_class):                            
                            #If yes, then stop collecting samples. Move to next class.
                            break

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)        

        return instances

def show_images(im_batch):
    #Unnormalize from [-1, 1] to [0,1]
    im_batch = im_batch*0.5 + 0.5 
    grid = torchvision.utils.make_grid(im_batch)
    grid = grid.detach().cpu().permute(1,2,0) * 255
    grid_im = Image.fromarray(np.array(grid).astype(np.uint8))
    return grid_im

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    print("Now pil_image is of size ", pil_image.size)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    print("And now it is  ", pil_image.size)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    print("Are crop_x and crop_y 0? ", crop_x, crop_y)
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def dataloader(image_size, batch_size, part_type, max_samples_per_type):
    """ Data loader for defectspectrum
    Args :
        image_size (int) : The input image size
        batch_size (int) : Batch size
        part_type (str) : The product or part type from DefectSpectrum
        max_samples_per_type : Maximum no of images to be loaded per defect type of the part
    Retruns :
        dataset (torchvision.datasets.ImageFolder or ImageFolderSelectSubset) : The dataset object
        dataloader (torch.utils.DataLoader) : A dataloader for the dataset 
     """
    preprocess = transforms.Compose(
        [
            transforms.CenterCrop(800),
            transforms.Resize((image_size,image_size)),
            transforms.ColorJitter(brightness=(0.5,1.5),contrast=(3),saturation=(0.3,1.5),hue=(-0.1,0.1)),
            transforms.RandomRotation([45,120], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]), # From [0,1] to [-1,1] Normalization
        ]
    )
    #Direct access from hf
    #dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
            #DefectSpectrum/Defect_Spectrum")
    #In case the dataset is locally stored
    dataset = ImageFolderSelectSubset(
            os.path.join('Defect_Spectrum/DS-MVTec',
            part_type, 'image'),
            transform=preprocess,
            max_samples_per_class = max_samples_per_type
            )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader


