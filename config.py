OPS_NAMES = ['ShearX',
            'ShearY',
            'TranslateX',
            'TranslateY',
            'Rotate',
            'AutoContrast',
            'Invert',
            'Equalize',
            'Solarize',
            'Posterize',
            'Contrast',
            'Color',
            'Brightness',
            'Sharpness',
            'Cutout',
            'Flip',
            'Identity']


def get_warmup_config(dset):
    # multiplier, epoch
    config = {'svhn': (2, 2),
            'cifar10': (2, 5),
            'cifar100': (4, 5),
            'mnist': (1, 1),
            'imagenet': (2, 3)}
    if 'svhn' in dset:
        return config['svhn']
    elif 'cifar100' in dset:
        return config['cifar100']
    elif 'cifar10' in dset:
        return config['cifar10']
    elif 'mnist' in dset:
        return config['mnist']
    elif 'imagenet' in dset:
        return config['imagenet']
    else:
        return config['imagenet']


def get_search_divider(model_name):
    # batch size is too large if the search model is large
    # the divider split the update to multiple updates
    if model_name == 'wresnet40_2':
        return 32
    elif model_name == 'resnet50':
        return 128
    else:
        return 16
