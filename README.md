# Bayesian KalmanNet: Quantifying Uncertainty in Deep Learning Augmented Kalman Filter

This branch includes the source code used in our paper:

Yehonatan Dahan, Guy Revach, Jindrich Dunik, and Nir Shlezinger. "[Bayesian KalmanNet: Quantifying Uncertainty in Deep Learning Augmented Kalman Filter](https://arxiv.org/abs/2309.03058)." (2024).


## Abstract

Recent years have witnessed a growing interest in tracking algorithms that augment Kalman Filters (KFs) with Deep Neural Networks (DNNs). By transforming KFs into trainable deep learning models, one can learn from data to reliably track a latent state in complex and partially known dynamics. However, unlike classic KFs, conventional DNN-based systems do not naturally provide an uncertainty measure, such as error covariance, alongside their estimates, which is crucial in various applications that rely on KF-type tracking. This work bridges this gap by studying error covariance extraction in DNN-aided KFs. We begin by characterizing how uncertainty can be extracted from existing DNN-aided algorithms and distinguishing between approaches by their ability to associate internal features with meaningful KF quantities, such as the Kalman Gain (KG) and prior covariance. We then identify that uncertainty extraction from existing architectures necessitates additional domain knowledge not required for state estimation. Based on this insight, we propose Bayesian KalmanNet, a novel DNN-aided KF that integrates Bayesian deep learning techniques with the recently proposed KalmanNet and transforms the KF into a stochastic machine learning architecture. This architecture employs sampling techniques to predict error covariance reliably without requiring additional domain knowledge, while retaining KalmanNet's ability to accurately track in partially known dynamics. Our numerical study demonstrates that Bayesian KalmanNet provides accurate and reliable tracking in various scenarios representing partially known dynamic systems.

## Overview

This repository consists of following Python scripts:
* `Main.py` the interface for applying both training and test for the different State-Spaces presented in our paper.
* `config.ini` configuration for running `Main.py` script. Further details about the parameters could be found in `config.md` [file](https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL/blob/main/config.md).
* `GSSFiltering/dnn.py` defines deep neural network (dnn) architectures: KalmanNet and Split-KalmanNet.
* `GSSFiltering/filtering.py` handles the filtering algorithms for the dnns and the extended kalman filter.
* `GSSFiltering/model.py` defines the State-Space model's parameters.
* `GSSFiltering/tester.py` handles the testing method.
* `GSSFiltering/trainer.py` handles the training method.


## Requirements

All required packages are listed in [requirement.txt](https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL/blob/develop/requirements.txt) file.


## Getting Started

To simply run the code, define the desired configuration in `config.ini` and execute `Main.py`.


## Notes
+ The Recurrent Kalman Network (RKN) comparison in the paper was done with respect to [RKN](https://github.com/ALRhub/rkn_share).  
+ This code is based on the [Split-KalmanNet](https://github.com/geonchoi/Split-KalmanNet) code.  
