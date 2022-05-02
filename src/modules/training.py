import hashlib
import itertools
import logging
import os
import pickle
import pprint
from typing import Any, Dict, List

import pandas as pd
import psutil
import tqdm

from src.datasets.datasets import DatasetId, get_dataloaders
from src.modules.snn import LoadCheckpointMode, SNN
from src.modules.spike_funcs import SpikeFuncType
from src.modules.spiking_layers import LayerType


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return:
	"""
	return {
		"dataset_id": [
			# DatasetId.MNIST,
			DatasetId.FASHION_MNIST
		],
		"to_spikes_use_periods": [
			True,
			False
		],
		# "as_timeseries": [True, False],
		# "n_steps": [100, 1_000, ],
		"n_hidden_neurons": [
			# 64,
			# [64, 64],
			128,
			# [32, 32],
			# 32
		],
		"spike_func": [SpikeFuncType.FastSigmoid, ],
		"hidden_layer_type": [
			# LayerType.LIF,
			LayerType.ALIF,
		],
		"use_recurrent_connection": [
			False,
			True
		],
		"learn_beta": [
			True,
			False
		],
		"nb_epochs": [
			# 15,
			30,
		],
	}


def get_meta_name(params: Dict[str, Any]):
	meta_name = f""
	for k, v in params.items():
		meta_name += f"{k}-{v}_"
	return meta_name[:-1]


def hash_params(params: Dict[str, Any]):
	"""
	Hash the parameters to get a unique and persistent id.
	:param params:
	:return:
	"""
	return int(hashlib.md5(get_meta_name(params).encode('utf-8')).hexdigest(), 16)


def save_params(params: Dict[str, Any], save_path: str):
	"""
	Save the parameters in a file.
	:param params:
	:return:
	"""
	pickle.dump(params, open(save_path, "wb"))


def train_with_params(params: Dict[str, Any], data_folder="tr_results", verbose=False):
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)

	dataloaders = get_dataloaders(
		dataset_id=params["dataset_id"],
		batch_size=256,
		# as_timeseries=params["as_timeseries"],
		# n_steps=params["n_steps"],
		to_spikes_use_periods=params["to_spikes_use_periods"],
		nb_workers=psutil.cpu_count(logical=False),
	)
	network = SNN(
		inputs_size=28 * 28,
		output_size=10,
		n_hidden_neurons=params["n_hidden_neurons"],
		# int_time_steps=params["n_steps"],
		spike_func=params["spike_func"],
		hidden_layer_type=params["hidden_layer_type"],
		use_recurrent_connection=params["use_recurrent_connection"],
		checkpoint_folder=checkpoint_folder,
		learn_beta=params.get("learn_beta", False),
	)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	# x_viz, _ = next(iter(dataloaders["train"]))
	# out_viz, _ = snn(x_viz.to(snn.device))
	# print(make_dot(out_viz).render("figures/snn_torchviz", format="png"))
	# snn.to_onnx()
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


def get_all_params_combinations(params_space: Dict[str, Any] = None) -> List[Dict[str, Any]]:
	if params_space is None:
		params_space = get_training_params_space()
	# get all the combinaison of the parameters
	all_params = list(params_space.keys())
	all_params_values = list(params_space.values())
	all_params_combinaison = list(map(lambda x: list(x), list(itertools.product(*all_params_values))))

	# create a list of dict of all the combinaison
	all_params_combinaison_dict = list(map(lambda x: dict(zip(all_params, x)), all_params_combinaison))
	return all_params_combinaison_dict


def train_all_params(training_params: Dict[str, Any] = None, data_folder: str = "tr_data", verbose=False):
	"""
	Train the network with all the parameters.
	:param verbose:
	:param data_folder:
	:param training_params:
	:return:
	"""
	os.makedirs(data_folder, exist_ok=True)
	results_path = os.path.join(data_folder, "results.csv")
	if training_params is None:
		training_params = get_training_params_space()

	all_params_combinaison_dict = get_all_params_combinations(training_params)
	columns = ['checkpoints', *list(training_params.keys()), 'train_accuracy', 'val_accuracy', 'test_accuracy']

	# load dataframe if exists
	try:
		df = pd.read_csv(results_path)
	except FileNotFoundError:
		df = pd.DataFrame(columns=columns)

	p_bar = tqdm.tqdm(all_params_combinaison_dict, desc="Training all the parameters")
	for params in p_bar:
		if str(hash_params(params)) in df["checkpoints"].values:
			continue
		# p_bar.set_description(f"Training {params}")
		try:
			result = train_with_params(params, data_folder=data_folder, verbose=verbose)
			df = pd.concat([df, pd.DataFrame(
				dict(
					checkpoints=[result["checkpoints_name"]],
					**{k: [v] for k, v in params.items()},
					train_accuracy=[result["accuracies"]["train"]],
					val_accuracy=[result["accuracies"]["val"]],
					test_accuracy=[result["accuracies"]["test"]],
				))],  ignore_index=True,
			)
			df.to_csv(results_path)
			p_bar.set_postfix(
				params=params,
				train_accuracy=result["accuracies"]['train'],
				val_accuracy=result["accuracies"]['val'],
				test_accuracy=result["accuracies"]['test']
			)
		except Exception as e:
			logging.error(e)
			continue
	p_bar.close()
	return df



