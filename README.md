### Simple comparison of Onsanger Machlup actions

Select action, tune parameters and launch. 

It is important to run CUDA_VISIBLE_DEVICES=5 ./sample.sh

CUDA_VISIBLE_DEVICES= ./sample.sh runs on CPU

The problem could be path in evaluators

Take a look into ddpm.py function om_interpolate. There you should have a guide. 