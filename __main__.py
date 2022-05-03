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
		learn_beta=params.get("learn_beta", False),
	)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	network.fit(
		dataloaders["train"],
		dataloaders["val"],
		nb_epochs=params.get("nb_epochs", 15),
		load_checkpoint_mode=LoadCheckpointMode.LAST_EPOCH,
		force_overwrite=True,
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
