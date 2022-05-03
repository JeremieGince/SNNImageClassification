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
<p align="center"> <img width="900" height="200" src="https://github.com/JeremieGince/SNNImageClassification/blob/main/figures/PipelineNet_Schm.png?raw=true"> </p>

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


## Code Example
Code example for the classification of the MNIST dataset using an SNN with the ALIF dynamics.

```python
import os
import pprint
from typing import Any, Dict

import psutil

from src.datasets.datasets import DatasetId, get_dataloaders
from src.modules.snn import LoadCheckpointMode, SNN
from src.modules.spike_funcs import SpikeFuncType
from src.modules.spiking_layers import LayerType
from src.modules.training import hash_params, save_params


def train_with_params(params: Dict[str, Any], data_folder="tr_results", verbose=False):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)

	dataloaders = get_dataloaders(
		dataset_id=params["dataset_id"],
		batch_size=256,
		to_spikes_use_periods=params["to_spikes_use_periods"],
		nb_workers=psutil.cpu_count(logical=False),
	)
	network = SNN(
		inputs_size=28 * 28,
		output_size=10,
		n_hidden_neurons=params["n_hidden_neurons"],
		spike_func=params["spike_func"],
		hidden_layer_type=params["hidden_layer_type"],
		use_recurrent_connection=params["use_recurrent_connection"],
		checkpoint_folder=checkpoint_folder,
		learn_beta=params.get("learn_beta", True),
	)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	network.fit(
		dataloaders["train"],
		dataloaders["val"],
		nb_epochs=params.get("nb_epochs", 15),
		load_checkpoint_mode=LoadCheckpointMode.LAST_EPOCH,
		force_overwrite=False,
		verbose=verbose,
	)
	network.load_checkpoint(LoadCheckpointMode.BEST_EPOCH)
	return dict(
		network=network,
		accuracies={k: network.compute_classification_accuracy(dataloaders[k]) for k in dataloaders},
		checkpoints_name=checkpoints_name,
	)


if __name__ == '__main__':
	results = train_with_params(
		{
			"dataset_id": DatasetId.MNIST,
			"to_spikes_use_periods": True,
			"n_hidden_neurons": 128,
			"spike_func": SpikeFuncType.FastSigmoid,
			"hidden_layer_type": LayerType.ALIF,
			"use_recurrent_connection": True,
			"learn_beta": True,
			"nb_epochs": 30,
		},
		verbose=True,
	)
	pprint.pprint(results, indent=4)
```



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
