from typing import Any, Dict

from pythonbasictools.device import log_pytorch_device_setup
from pythonbasictools.logging import logs_file_setup

from src.datasets.datasets import DatasetId
from src.modules.spike_funcs import SpikeFuncType
from src.modules.spiking_layers import LayerType
from src.modules.training import train_all_params


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return:
	"""
	return {
		"dataset_id": [DatasetId.FASHION_MNIST],
		"to_spikes_use_periods": [False],
		"n_hidden_neurons": [100, ],
		"spike_func": [SpikeFuncType.FastSigmoid, SpikeFuncType.Phi],
		"hidden_layer_type": [LayerType.ALIF, ],
		"use_recurrent_connection": [False, ],
	}


if __name__ == '__main__':
	logs_file_setup(__file__)
	log_pytorch_device_setup()
	train_all_params(training_params=get_training_params_space(), data_folder="spk_tr_data", verbose=True)
