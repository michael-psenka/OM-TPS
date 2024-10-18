# MNIST Diffusion
![60 epochs training from scratch](assets/demo.gif "60 epochs training from scratch")

Only simple depthwise convolutions, shorcuts and naive timestep embedding, there you have it! A fully functional denosing diffusion probabilistic model while keeps ultra light weight **4.55MB** (the checkpoint has 9.1MB but with ema model double the size).

## Training
Install packages
```bash
pip install -r requirements.txt
```
Start default setting training 
```bash
python train_mnist.py
```
Feel free to tuning training parameters, type `python train_mnist.py -h` to get help message of arguments.

## Interpolation
```bash
python om_interpolation.py
```

The following hyperparams should give decent interpolations on MNIST:

```bash
python om_interpolation.py --steps 1500 --batch_size 32 --truncate_v_gradient --v_scale 0.08 --kernel_var 2.1 --disable_logging --lr 5e-2
```

## Reference
A neat blog explains how diffusion model works(must read!): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

The Denoising Diffusion Probabilistic Models paper: https://arxiv.org/pdf/2006.11239.pdf 

A pytorch version of DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch

# CelebaHQ installation

1. Download the CelebaHQ dataset from the following website: https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only. Place it in a directory of choice. The structure should look like: `/path/to/celebaHQ/train` and `/path/to/celebaHQ/valid`.
2. When creating the celeba dataloader from `data/dataloaders.py`, change the `root` parameter to the path of the celebaHQ dataset, so a call would look like the following:

```python
celeba_dataloader = CelebaDataLoader(batch_size=32, root_dir="/path/to/celebaHQ")
```

