# MADAug
MADAug： When to Learn What: Model-Adaptive Data Augmentation Curriculum

### Introduction
Model-Adaptive Data Augmentation (MADAug) that jointly trains an augmentation policy network to teach the model “when to learn what”.
**Our paper is accepted by ICCV 2023.**

### Getting Started
Code supports Python 3.


### Install requirements

```shell
pip install -r requirements.txt
```

### Run data augmentation 
Script to search for the dynamic augmentation policy and train task model for is located in `main.sh`. Pass the dataset name as the argument to call the script.

For example, to use the dynamic augmentation policy for classifying the reduced_cifar10 dataset:

```shell
bash main.sh reduced_cifar10
```
### References & Opensources
Part of our implementation is adopted from the Fast AutoAugment and DADA repositories.
- Fast AutoAugment (https://github.com/kakaobrain/fast-autoaugment)
- DADA (https://github.com/VDIGPKU/DADA)
- AdaAug (https://github.com/jamestszhim/adaptive_augment)
