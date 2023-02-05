# DDPMx

Original DDPM model with diffuser integration in PyTorch.

- Test original DDPM model using latest techniques, e.g., DDIM, with [diffusers](https://github.com/huggingface/diffusers)

## Usage

Pytorch pretrained model comes from [pytorch_diffusion](https://github.com/pesser/pytorch_diffusion).

```
python generate.py
sh test_fid.sh
```


## Results

Evaluating 50k samples with
[torch-fidelity](https://github.com/toshas/torch-fidelity) gives


| Dataset            | EMA | Framework  | Model            | FID      |
| ------------------ | --- | ---------- | ---------------- | -------- |
| CIFAR10 Train      | no  | PyTorch    | `cifar10`        | 12.13775 |
|                    |     | TensorFlow | `tf_cifar10`     | 12.30003 |
|                    | yes | PyTorch    | `ema_cifar10`    | 3.21213  |
|                    |     | TensorFlow | `tf_ema_cifar10` | 3.245872 |
| CIFAR10 Validation | no  | PyTorch    | `cifar10`        | 14.30163 |
|                    |     | TensorFlow | `tf_cifar10`     | 14.44705 |
|                    | yes | PyTorch    | `ema_cifar10`    | 5.274105 |
|                    |     | TensorFlow | `tf_ema_cifar10` | 5.325035 |


Results with different samplers.

| Dataset            | Sampler   | Framework | Model         | FID      |
| ------------------ | --------- | --------- | ------------- | -------- |
| CIFAR10 Train      | ddim20    | PyTorch   | `ema_cifar10` | 11.08184 |
|                    | ddim100   | PyTorch   | `ema_cifar10` | 5.515337 |
|                    | ddim50    | PyTorch   | `cifar10`     | 13.45145 |
|                    | ddim100   | PyTorch   | `cifar10`     | 11.05178 |
|                    | ddpm1000  | PyTorch   | `ema_cifar10` | 3.171527 |
|                    | ddpm1000  | PyTorch   | `cifar10`     | 14.28527 |
|                    | ddpm1000x | PyTorch   | `ema_cifar10` | 4.741088 |
| CIFAR10 Validation | ddim20    | PyTorch   | `ema_cifar10` | 13.22313 |
|                    | ddim100   | PyTorch   | `ema_cifar10` | 7.609758 |
|                    | ddim50    | PyTorch   | `cifar10`     | 15.63806 |
|                    | ddim100   | PyTorch   | `cifar10`     | 13.22416 |
|                    | ddpm1000  | PyTorch   | `ema_cifar10` | 5.245864 |
|                    | ddpm1000  | PyTorch   | `cifar10`     | 12.15665 |
|                    | ddpm1000x | PyTorch   | `ema_cifar10` | 6.823921 |


## Reference

- [diffusion](https://github.com/hojonathanho/diffusion): Official DDPM implementation in tensorflow.
- [pytorch_diffusion](https://github.com/pesser/pytorch_diffusion): Pytorch translation of original tensorflow version.