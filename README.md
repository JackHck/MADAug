# [ICCV 2023] MADAug: When to Learn What: Model-Adaptive Data Augmentation Curriculum

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]([https://arxiv.org/abs/2309.04747])


## Introduction
Model-Adaptive Data Augmentation (MADAug) that jointly trains an augmentation policy network to teach the model “when to learn what”. In this paper, we study two fundamental problems towards developing a data-and-model-adaptive data augmentation policy that determines a curriculum of “when to learn what” to train a model: (1) when to apply data augmentation in training? (2) what data augmentations should be applied to each training sample at different training stages?



## Getting Started
Code supports Python 3.


## Install requirements

```shell
pip install -r requirements.txt
```

## Run data augmentation 
Script to search for the dynamic augmentation policy and train task model for is located in `main.sh`. Pass the dataset name as the argument to call the script.

For example, to use the dynamic augmentation policy for classifying the reduced_cifar10 dataset:

```shell
bash main.sh reduced_cifar10
```
## References & Opensources
Part of our implementation is adopted from the Fast AutoAugment and DADA repositories.
- Fast AutoAugment (https://github.com/kakaobrain/fast-autoaugment)
- DADA (https://github.com/VDIGPKU/DADA)
- AdaAug (https://github.com/jamestszhim/adaptive_augment)

## Citation
If you find MADAug helpful in your research, please consider citing: 
```bibtex   
@article{hou2023learn,
  title={When to Learn What: Model-Adaptive Data Augmentation Curriculum},
  author={Hou, Chengkai and Zhang, Jieyu and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2309.04747},
  year={2023}
}
```


