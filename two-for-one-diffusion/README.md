# Fast Folding Proteins and Tetrapeptides


## Requirements

***

## Sampling and evaluation
### Pretrained models 
Trained models per protein can be found in the [saved_models](./saved_models/) folder. 



For each protein, we provide a ```model-best.pt``` file with the checkpoint corresponding to the best validation loss as well as an ```args.pickle``` file containing the arguments used for that run.

### Sampling
Sampling for each protein can be done in either the i.i.d. setting or the dynamics setting using the [sample script](./sample.py). As an example, we show simple commands for obtaining chignolin samples in both settings. We refer to ```python sample.py --help``` for more sampling options and an explanation of all individual arguments. 



## Training the diffusion model
The DFF model was trained from scratch using the [training script](./main_train.py). The underlying code for the DDPM and graph transformer can be found in the [models folder](./models/). For training options, please check:

```bash
python main_train.py --help
```

***

## External code-sources used to create this codebase
1. This codebase builds heavily on [Two for One Diffusion](https://github.com/microsoft/two-for-one-diffusion) (PyTorch) by Microsoft. We use many of the same pretrained models, sampling and evaluation code, and infrastructure. 
2. We also build on [MDGen](https://github.com/bjing2016/mdgen) for Markov State Model (MSM) analysis and tetrapeptide infrastructure.

***
## Citation
If you use this code in your research, please cite our paper.

```bibtex
@inproceedings{raja2025action,
  title={Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional},
  author={Raja, Sanjeev and {\v{S}}{\'\i}pka, Martin and Psenka, Michael and Kreiman, Tobias and Pavelka, Michal and Krishnapriyan, Aditi S},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
  organization={PMLR}
}