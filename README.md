# A 3D-Aware Conditional Diffusion Model for Gaze Redirection
This repository is the code base of a Master's Thesis "A 3D-Aware Conditional Diffusion Model for Gaze Redirection"

* Author: YeonJoo Cho
* Examiner: Prof. Dr. Andreas Bulling
* Supervisor: Chuhan Jiao, M.sc.

### Abstract
This thesis proposes a novel approach for gaze redirecton, by reformulating the task of gaze redirection as an image-to-image translation using conditional diffusion models.
Specifically, we adopt viewpoint-conditioned diffusion models that have previously been utilized in novel view synthesis tasks and can leverage 3D transformations as conditions.

Taking advantage of this model, we introduce explicit 3D gaze rotations derived from gaze labels, and latent facial parameters from an existing framework designed for 3D face reconstruction, to condition the model learning. 

Our conceptual idea of interpreting the gaze redirection task as a 3D-aware conditional image generation process demonstrate substantial potentials for effective gaze redirection, backed up with  further ideas for refinement and development.

## Requirements
Necessary packages that needs to be installed are outlined in the following file.  
Simply run:
```
pip install -r requirements.txt
```

This work includes several external codeworks, and the following suggested file hierarchies is the default setting for simple training and testing.
At the same time, as all available configuration parameters are defined in config.py, it can be freely adjusted and managed to a desirable directory.

## Dataset
Our work uses the [GazeCapture](https://gazecapture.csail.mit.edu/) as the main dataset for training and evaluation.
A strict license must be agreed to download the dataset [here](https://gazecapture.csail.mit.edu/download.php). 

As we pre-process the datase analogously to STED, please follow the instructions of [this repository](https://github.com/swook/faze_preprocess) to pre-process the dataset.  
Then, the preprocessed the preprocessed 'GazeCapture.h5' file and 'gazecapture_split.json' should be located under '/datasets/GazeCapture/'.

## Pre-trained Models
#### 1.Self-Learning Transformations for Improving Gaze and Head Redirection (STED)

STED is the state-of-the-art framework for gaze redirection. We use this as our baseline for evaluation. 
Also, we use their pre-trained estimator for the training and evaluation phase.

The pre-trained checkpoints can be downloaded from the
official [repository](https://github.com/zhengyuf/STED-gaze), along with the two external estimators.

Below is a shortcut to download the files: 

- [STED](https://drive.google.com/file/d/1PGb1GKy31WE692rvk_iBYQdeO_OK9BRi/view?usp=sharing)
- [VGG gaze estimator for training](https://drive.google.com/file/d/1amWI-1mrVIRLgUntnvBwuAj3Nn9ktiq9/view?usp=sharing)
- [ResNet gaze estimator for evaluation](https://drive.google.com/file/d/1P4PnRMDhb37NXnezYosiwqCQrEguD2kd/view?usp=sharing)

These checkpoints should be located under '/models/pretrained/'.

#### 2.DECA: Detailed Expression Capture and Animation 

DECA is a framework for single image to 3D face reconstruction.
We specifically use the DECA encoder to produce our latent facial parameters. 

The pretrained model can be collected in the [offical repository](https://github.com/yfeng95/DECA) of DECA.
As, their architecture includes the [FLAME](https://flame.is.tue.mpg.de/index.html) model, the pretrained weights for the FLAME model should also be downloaded.
This requires additional user registration from the FLAME's repository. 
Then, the "FLAME 2020" version model is specifically required.

Moreover, the '/data' folder from the original repository MUST be downloaded and be strictly located under our project as '/models/pretrained/decalib/data/', with an exception.

## Model Selection
This work has several versions of models. The main XUNet diffusion model is wrapped as Network(), with other models needed in the training phase, in network.py.
- xunet.py: is the vanilla model that uses only pose embeddings for conditions.
- xunet_deca.py: is the model using pose embeddings and latent code as conditions. The best observed checkpoint is provided in 'projects/cho/ckpt/org/'


Model versions for further experiments and imporvemnts are provided in 'projects/cho/models/'. However, we do not provide all of them here, as it is yet ideas for further improvements and distrubs the execution flow. These file are the examples of the variation.
- xunet_deca_v1.py: is another variation of concatenating the latent code.
- xunet_deca_v2.py: is another variation of stacking the latent code.


  
## Training

To train our proposed model, simply run this command:

```
python train_ddp_main.py  --transfer=False --use_deca=False
```
While other configuration can be set in config.py, the option to transfer checkpoints from previous training and the option to use the latent codes are specified with the command.
Above command denotes no transfer of previous cheackpoints and onlt use the pose embedings as conditions, without the use of latent codes.

Tensorboard logging is also supported. 
For a plausible sampling result, training will take at least 7 days on 4 GPUs.

## Sampling & Evaluation

To sample redirected images from the pretrained model, run:

```
python sampling_main.py 
```
For sampling, there are several options to be passed:
| Command | Description |
| --- | --- |
| --gpu_id | the gpu to use|
| --max_w | the maximum value of weight to test the classifier-free guidance. It supports 1.0 intervals. |
|--use_deca|whether the  model to be tested includes the use of latent codes or not.|
|--save_gif|save the results as .gif|
|--sample_only|as default, we provide the sampled image with the evaluations from the STED model. To disable baseline evaluation, this should be set as False.|
|--save_dir _name|The directory name to save the sampled images. As different checkpoints output different results, we provide this option to avoid confusion. The results wll be saved as '/sampling_res/<save_dir_name>'|
  
## Side Notes
Our first extensive experiments with ETH-XGaze dataset is not included here, as the results are noisy.
However, to exploit the better image quaility of ETH-XGaze and an effort for cross-data evaluation of our current model,  
it can be tested upon querying test_eth_xgaze() that returns a dataloader. 
As it the preprocessd ETH-XGaze dataset has face masks to remove the backgroung image, this option can be passed to the function as test_eth_xgaze(remove_bg =True).
Upon testing with the current checkpoint, the results are misleading.
However, we leave the structure as it can be found useful in improved model version of future works.
