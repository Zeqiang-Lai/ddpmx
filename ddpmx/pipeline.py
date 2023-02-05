import torch
import torch.nn as nn
from typing import Optional
from PIL import Image
from tqdm import tqdm

class DDPMPipeline:
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    @property
    def device(self):
        return next(iter(self.unet.parameters())).device

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
    ):
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        image = torch.randn(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise model_output
            
            ts = torch.full((batch_size,), t, device=self.device).long()
            
            model_output = self.unet(image, ts)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = self.numpy_to_pil(image)
        return image
