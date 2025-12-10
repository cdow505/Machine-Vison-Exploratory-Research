# Machine-Vison-Exploratory-Research
A repository containing the jupyter notebook files I used during my CS 4732 exploratory research project to evaluate the effects of different CNN architectures.
Note: not all of this code was written by me. I based some of my code off of other sources and tutorials, which can be found at the bottom of this readme.


I don't have all of the files from my intial optimization phase, but below I've provided an index for the ones that I do have, mapping the exact experiment that each Jupyter Notebook filename refers to.
_______________________________________________________________________________________________________________
Optimization Phase:
A 7-layer CNN with an early stopper patience of 3, a 3x3 padded kernel, no scheduler, and no loss weights or weighted random sampler:
MachineVisionCnnNotebookBaseline

A 15-layer CNN with an early stopper patience of 5, a padded 3x3 kernel, no scheduler, and no loss weights or weighted random sampler.
MachineVisionCnnNotebookEnhancedBaseline

A 7-Layer CNN with an early stopper patience of 3, a padded 3x3 kernel,  loss weights, and a shceduler with a patience of 3:
MachineVisionCnnNotebook

A 14 layer CNN with an early stopper patience of 3, a padded 3x3 kernel, no adaptive pooling layer, a scheduler with a patience of 3, and non-improved loss weights
MachineVisionCnnNotebookEnhanced

A 15 layer CNN with an early stopper patience of 5, a padded 3x3 kernel, no scheduler, and non-improved loss weights.
MachineVisionCnnNotebookEnhancedAdaptivePool

A 7-layer CNN with an early stopper patience of 5, a scheduler with a patience of 2, a padded 3x3 kernel, no loss weights, and a weighted random sampler
MachineVisionCnnNotebookKernel

Note: this is not all of my optimization phase notebooks. I did some experiments before I started to save each experiment as a separate notebook file, so I don't have access to those.
_______________________________________________________________________________________________________________
Experimentation Phase:

15-layer CNNs with an early stopper patience of 5, improved loss weights, and no scheduler:

Padded 3x3: MachineVisionCnnNotebookEnhancedAdaptivePool2

Padded 5x5: MachineVisionCnnNotebookEnhancedAdaptivePool5by5

Padded 7x7: MachineVisionCnnNotebookEnhancedAdaptivePool7by7

Non-Padded 3x3: MachineVisionCnnNotebookLargeNoPad3x3

Non-Padded 5x5: MachineVisionCnnNotebookLargeNoPad5x5

Non-Padded 7x7:MachineVisionCnnNotebookLargeNoPad7x7

_________________________________________
7-Layer CNNs with an early stopper patience of 5, improved loss weights, and a scheduler with a patience of 3:

Padded 3x3: MachineVisionCnnNotebookWeights

Padded 5x5: MachineVisionCnnNotebookWeights5by5

Padded 7x7: MachineVisionCnnNotebookWeights7by7

Non-Padded 3x3: MachineVisionCnnNotebookNoPad3x3

Non-Padded 5x5: MachineVisionCnnNotebookNoPad5x5

Non-Padded 7x7:MachineVisionCnnNotebookNoPad7x7
________________________
Pre-trained resnet-50 Model runs:
Model just using intial Resnet-50 intialization with 5 classes and no additional training:

MachineVisionCNNResnet50NoTrain


Model using intial Resnet-50 intialization with 5 classes, but trained for additional epochs until validation loss stopped improving with an early stopper patience of 5, and a scheduler with a patience of 3:

MachineVisionCnResnet50Trained

_______________________________________________________________________________________________________________

References:
1. Another example of a CNN used for pedestrian detection: 
Szarvas, Mate & Yoshizawa, A. & Yamamoto, M. & Ogata, J.. (2005). Pedestrian detection with convolutional neural networks. 224 - 229. 10.1109/IVS.2005.1505106.
https://www.researchgate.net/publication/4171908_Pedestrian_detection_with_convolutional_neural_networks

This source is a research paper that applies a CNN to detect obstacles and pedestrians, and compares its performance to other machine learning models such as a Support vector machine. Although this source isn’t directly applicable to my classification problem, it still applies to the autonomous vehicle problem in general, and it helps me to understand what to expect for the performance of a CNN in general in terms of time complexity. 

2. Pytorch documentation: https://docs.pytorch.org/docs/stable/index.html

This source is the documentation for Pytorch, the python neural network library that I  utilize for my implementation. Naturally, if I want to know more about Pytorch and how to utilize it effectively, I have to have a thorough understanding of the documentation. Primarily, I used the torch.nn module, which has subfunctions for applying convolutions and setting up the various layers of the CNN so I don’t have to build a model from scratch. However, I also make use of torchvision and a few other torch-based modules, as I’ve listed in my implementation details section. 

3. A Basic tutorial about how to build an extremely basic CNN: https://www.geeksforgeeks.org/deep-learning/building-a-convolutional-neural-network-using-pytorch/

This document is a tutorial about how to build a very basic CNN using pytorch. It includes code for a few basic image transportation, how to use dataloadsers, and the definition of class labels for a CIFAR-10 dataset, along with a CNN architecture definition for two convolutional layers with RELU and max pulling, along with three fully connected layers. In general, it’s a good resource to help me understand exactly how CNN architectures work, and how they can be applied in a real world example to create a CNN model to classify images.

4.The dataset I’m using https://www.kaggle.com/datasets/alincijov/self-driving-cars/data

This source is the Kaggle dataset that I’m using. The dataset itself is a public domain dataset so I can use it in my research without consequence, and notably it’s originally a dataset intended for object detection as opposed to classification, however it does contain classification labels so it can be used for classification. As I’ve mentioned previously, the dataset is extremely unbalanced in a distribution similar to what one would expect to actually see on the road, so I believe it’s valuable to use to attempt to classify images.
5. A guide to early stopping, my implementation is based on this: https://medium.com/@piyushkashyap045/early-stopping-in-deep-learning-a-simple-guide-to-prevent-overfitting-1073f56b493e

This source is a guide that explains the technique of early stopping, and explains somewhat how it can be used in practice to stop a model in the midst of training if validation loss is stagnated. When I manually implemented my early stopping class and methodologies, I used the pre-existing early stopping implementation in this article as reference and based my code on it.

6. Discussion about best metrics to use for evaluating imbalanced classification: https://www.researchgate.net/post/Which-is-the-best-metric-to-evaluate-unbalanced-classification-model

This source is a discussion about the various evaluation metrics commonly used when it comes to evaluating classification models with extremely unbalanced data. It is primarily because of the posts on here and the general discussion that I have elected to focus on precision, recall, and f1 scores for my evaluation. 

7. Torch weighted random sampler documentation: 
https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler

This source is a link to the documentation of pytorch’s weighted random sampler module, which I make use of in order to help deal with drastic class imbalance.

8 Article about learning rate decay scheduling. I’m using the ReduceLROnPlateau reduction: https://machinelearningmastery.com/a-gentle-introduction-to-learning-rate-schedulers/

This source is an article about learning rate schedulers, and the different approaches generally used for learning rate decay. It provides insight into the different functions generally used in learning rate decay and why it’s helpful, and I make use of the ReducecLROnPleateu method that it mentions.

9. Pytorch CrossEntropyLoss module documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
This source is the documentation for the cross entropy loss function that I make use of in my models, and includes details about weighted cross entropy loss being useful for imbalanced classification.

10. Resnet-50 pre-trained model: https://huggingface.co/microsoft/resnet-50
This source is the resnet-50 pre-trained model that I used both with raw weight initialization and with additional training to compare my model results to, in order to help me get a frame of reference on how well or poorly my model was optimized.

