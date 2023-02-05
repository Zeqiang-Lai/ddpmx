from pathlib import Path
from tqdm import tqdm
import torch
from ddpmx.model import UNet
from ddpmx.pipeline import DDPMPipeline
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

cifar10_cfg = {
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": (1, 2, 2, 2),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.1,
}


net = UNet(**cifar10_cfg).cuda()

ckpt = 'ckpt/diffusion_models_converted/ema_diffusion_cifar10_model/model-790000.ckpt'
net.load_state_dict(torch.load(ckpt))
net.eval()


scheduler = DDPMScheduler()
pipeline = DDPMPipeline(net, scheduler)

save_dir = Path('results/ddpm1000')
save_dir.mkdir(exist_ok=True, parents=True)
bs = 500

for k in tqdm(range(100)):
    imgs = pipeline(batch_size=bs, num_inference_steps=1000)
    for i in range(bs):
        imgs[i].save(save_dir / f'{k*bs+i}.png')