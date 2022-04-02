import itertools
import logging
import pprint
from typing import Any, Dict

import psutil
from pythonbasictools.device import log_pytorch_device_setup
from pythonbasictools.logging import logs_file_setup

from src.datasets.datasets import DatasetId, get_dataloaders
from src.modules.snn import LoadCheckpointMode, SNN
from src.modules.spike_funcs import HeavisidePhiApprox, HeavisideSigmoidApprox, SpikeFuncType
from src.modules.spiking_layers import ALIFLayer, LIFLayer, LayerType
from src.modules.training import get_all_params_combinations, hash_params, train_all_params, train_with_params

if __name__ == '__main__':
	logs_file_setup(__file__)
	log_pytorch_device_setup()

	# delta_t = 1e-3
	# n_steps = 100
	#
	# ts = True
	# dataset_id = DatasetId.FASHION_MNIST
	# hidden_layer_type = ALIFLayer
	# meta_name = f"{dataset_id.name}{'-ts' if ts else ''}-{hidden_layer_type.__name__}"
	# logging.info(f"Dataset: {meta_name}")
	# dataloaders = get_dataloaders(
	# 	dataset_id,
	# 	batch_size=256,
	# 	as_timeseries=ts,
	# 	n_steps=n_steps,
	# 	to_spikes_use_periods=True,
	# 	# nb_workers=psutil.cpu_count(logical=False),
	# )
	#
	# snn = SNN(
	# 	inputs_size=28 * 28,
	# 	output_size=10,
	# 	n_hidden_neurons=[100, ],
	# 	int_time_steps=n_steps,
	# 	dt=delta_t,
	# 	spike_func=HeavisideSigmoidApprox,
	# 	hidden_layer_type=hidden_layer_type,
	# 	use_recurrent_connection=False,
	# 	checkpoint_folder=f"checkpoints-{meta_name}",
	# )
	# # x_viz, _ = next(iter(dataloaders["train"]))
	# # out_viz, _ = snn(x_viz.to(snn.device))
	# # print(make_dot(out_viz).render("figures/snn_torchviz", format="png"))
	# # snn.to_onnx()
	# loss_hist = snn.fit(
	# 	dataloaders["train"],
	# 	dataloaders["val"],
	# 	lr=1e-3,
	# 	nb_epochs=15,
	# 	load_checkpoint_mode=LoadCheckpointMode.LAST_EPOCH,
	# 	force_overwrite=True,
	# 	# optimizer=torch.optim.SGD(snn.parameters(), lr=1e-3, nesterov=True, momentum=0.9)
	# )
	# snn.load_checkpoint(LoadCheckpointMode.LAST_EPOCH)
	#
	# train_acc = snn.compute_classification_accuracy(dataloaders["train"])
	# test_acc = snn.compute_classification_accuracy(dataloaders["test"])
	# logging.info(f"Training accuracy: {train_acc:.3f}")
	# logging.info(f"Test accuracy: {test_acc:.3f}")
	train_all_params()
	# _, accs = train_with_params(
	# 	{
	# 		"dataset_id": DatasetId.MNIST,
	# 		"to_spikes_use_periods": True,
	# 		"n_steps": 10,
	# 		"n_hidden_neurons": 100,
	# 		"spike_func": SpikeFuncType.FastSigmoid,
	# 		"hidden_layer_type": LayerType.LIF,
	# 		"use_recurrent_connection": False,
	# 	},
	# 	verbose=True,
	# )
	# pprint.pprint(accs, indent=4)
