# deep-learning-project

## Install
1. Install the python3 version of [miniconda](https://docs.conda.io/en/latest/miniconda.html). Follow the installation instructions for your platform.
2. Use conda to create a virtual environment named `cs236605-hw`. From the root directory, run
```
conda env create -f environment.yml
```

3. Activate the new environment by running
```
conda activate cs236605-hw
```
Note: If there is any update in the `environment.yml` file. You can use
```
conda env update
```
## Training
Train ResNet generator and discriminator with hinge loss: python main.py --model resnet --loss hinge

Train ResNet generator and discriminator with wasserstein loss: python main.py --model resnet --loss wasserstein

Train DCGAN generator and discriminator with cross-entropy loss: python main.py --model dcgan --loss bce

## Main prolem they solved.
they generate examples that are much more diverse than the conventional weight normalization and achieve better or comparative inception scores.

## Why don't use on generator?

## The task we complete
1. redo the DCGAN-like model (with spectral normalization)
2. redo the ResNet-like model (with spectral normalization)
3. add GAN hack on the model
4. apply a new application gan with spectral normalization
