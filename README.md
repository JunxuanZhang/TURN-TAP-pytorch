# TURN-TAP-pytorch
This is a pytorch implementation of [TURN TAP: Temporal Unit Regression Network for Temporal Action Proposals](https://arxiv.org/abs/1703.06189). This code is for research purpose and suggestions are welcome.

# Enviorment
Pytorch 0.4.0

CUDA 8.0

Python 2.7.6

# References
The tensorflow implementation code provided by the authors: https://github.com/jiyanggao/TURN-TAP/tree/master/turn_codes

# Prepare the features
DenseFlow Features in the Google Drive：[val set](https://drive.google.com/file/d/1-6dmY_Uy-H19HxvfK_wUFQCYHmlPzwFx/view?usp=sharing), [test set](https://drive.google.com/file/d/1Qm9lIJQFm5s6hDSB_2k1tj8q2tnabflJ/view?usp=sharing)

# Setup
use git to clone this repository

$ git clone --recursive https://github.com/JunxuanZhang/TURN-TAP-pytorch/

Then create two necessary folders

$ mkdir features results

Move the downloaded features to the 'features' folder

# Training and evaluation
To train and evaluate the TURN model, run the 'main.py' script

$ python main.py

if you want to continue training from the specfic checkpoint, use '--resume' option

$ python main.py --resume CHECKPOINT_PATH

