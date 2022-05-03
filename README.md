<center> <h1>Classification of MNIST and Fashion-MNIST images using SNNs</h1> </center>
 

## Abstract

Spiking neural networks (SNN) are a new approach to artificial intelligence, which try to mimic biological 
neural networks by using neural dynamics from neuroscience, such as LIF and ALIF. This project has thus 
allowed us to highlight the potential and efficiency of SNNs in artificial intelligence and specifically 
in image classification. Indeed, the image databases, MNIST and Fashion-MNIST, were classified with this 
type of network, with respective accuracies of 96.19% and 81.94%, which is competitive with multilayer 
perceptron networks (MLP) (95.3% for MNIST and 91.4% for Fashion-MNIST). Moreover, it was possible to 
demonstrate that the ALIF dynamics is more efficient than the LIF dynamics for image classification. Other 
parameters also have an important effect on the accuracy of the models: the presence of recurrent connections 
decreases the accuracy by an average of 45%, the use of a periodic input signal decreases it by 15% 
and the number of neurons in the network has no significant effect on the accuracy. Finally, ALIF was able 
to outperform its state-of-the-art MLP counterpart for the MNIST dataset. For Fashion-MNIST, although the 
SNN models did not outperform the state-of-the-art results, several avenues of research were revealed that 
would significantly improve the results, such as extending the training, training the beta parameters, and 
reducing the number of neurons.


## Prediction Pipeline

The prediction pipeline for the spiking neural network is as follows:
<p align="center"> <img width="900" height="150" src="https://github.com/JeremieGince/SNNImageClassification/blob/main/figures/PipelineNet_Schm.png?raw=true"> </p>

in the previous figure, the x variable is the input image, the Sx variable is the image transformed
into spikes, the z variable is the spikes generated by the SNN, and the y variable is the output of the
readout layer. Finally, the classification probability px are computed by the softmax function.


## Results

The results of the experiments are presented in the following figures.

### MNIST
<p align="center"> <img width="1200" height="500" src="https://github.com/JeremieGince/SNNImageClassification/blob/main/figures/MNIST_precision_128N.png?raw=true"> </p>



### Fashion-MNIST
<p align="center"> <img width="1200" height="500" src="https://github.com/JeremieGince/SNNImageClassification/blob/main/figures/FMNIST_precision.png?raw=true"> </p>

### Legend
- REC : recurrent connections flag;
- P : periodical signal flag; 
- H or HN : number of hidden neurons;
- I : number of training iterations;
- B : beta training flag.



## Requirements
- ```pip install -r requirements.txt```
- ```pip install git+https://github.com/JeremieGince/PythonBasicTools```



# License
[MIT License](LICENSE.md)


# Citation
```
@article{Gince_LamontagneCaron_SNNImgClassification_2022,
  title={SNN Image Classification},
  author={Gince, Jérémie and Lamontagne-Caron, Rémi},
  year={2022},
  publisher={Université Laval},
  url={https://github.com/JeremieGince/SNNImageClassification},
}
```
