import enum
import json
import logging
import os
import time
from copy import deepcopy
from typing import Dict, Iterable, Union

import numpy as np
import torch
from pythonbasictools.progress_bar import printProgressBar
from torch import nn
from torch.utils.data import DataLoader

from src.modules.spike_funcs import HeavisideSigmoidApprox
from src.modules.spiking_layers import LIFLayer, ReadoutLayer
from src.modules.utils import mapping_update_recursively


class LoadCheckpointMode(enum.Enum):
	BEST_EPOCH = enum.auto()
	LAST_EPOCH = enum.auto()


class SNN(torch.nn.Module):
	SAVE_EXT = '.pth'
	SUFFIX_SEP = '-'
	CHECKPOINTS_META_SUFFIX = 'checkpoints'
	CHECKPOINT_SAVE_PATH_KEY = "save_path"
	CHECKPOINT_BEST_KEY = "best"
	CHECKPOINT_EPOCHS_KEY = "epochs"
	CHECKPOINT_EPOCH_KEY = "epoch"
	CHECKPOINT_LOSS_KEY = 'loss'
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
	CHECKPOINT_STATE_DICT_KEY = "model_state_dict"
	CHECKPOINT_FILE_STRUCT: Dict[str, Union[str, Dict[int, str]]] = {
		CHECKPOINT_BEST_KEY: CHECKPOINT_SAVE_PATH_KEY,
		CHECKPOINT_EPOCHS_KEY: {0: CHECKPOINT_SAVE_PATH_KEY},
	}
	load_mode_to_suffix = {mode: mode.name for mode in list(LoadCheckpointMode)}

	def __init__(
			self,
			inputs_size: int,
			output_size: int,
			n_hidden_neurons: Iterable[int] = None,
			use_recurrent_connection: Union[bool, Iterable[bool]] = True,
			dt=1e-3,
			int_time_steps=1_000,
			tau_syn=10e-3,
			tau_mem=5e-3,
			spike_func=HeavisideSigmoidApprox.apply,
			device=None,
			checkpoint_folder: str = "checkpoints",
			model_name: str = "snn",
	):
		super(SNN, self).__init__()
		self.input_size = inputs_size
		self.output_size = output_size

		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.dt = dt
		self.int_time_steps = int_time_steps
		self.tau_syn = tau_syn
		self.tau_mem = tau_mem
		self.spike_func = spike_func

		self.checkpoint_folder = checkpoint_folder
		self.model_name = model_name

		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else []
		self.use_recurrent_connection = use_recurrent_connection
		self.layers = nn.ModuleDict()
		self._add_layers_()
		self.initialize_weights_()

	@property
	def checkpoints_meta_path(self) -> str:
		return f"{self.checkpoint_folder}/{self.model_name}{SNN.SUFFIX_SEP}{SNN.CHECKPOINTS_META_SUFFIX}.json"

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def _add_input_layer_(self):
		if not self.n_hidden_neurons:
			return
		self.layers["input"] = LIFLayer(
			input_size=self.input_size,
			output_size=self.n_hidden_neurons[0],
			use_recurrent_connection=self.use_recurrent_connection,
			dt=self.dt,
			tau_syn=self.tau_syn,
			tau_mem=self.tau_mem,
			spike_func=self.spike_func,
			device=self.device,
		)

	def _add_hidden_layers_(self):
		if not self.n_hidden_neurons:
			return
		for i, hn in enumerate(self.n_hidden_neurons[:-1]):
			self.layers[f"hidden_{i}"] = LIFLayer(
				input_size=hn,
				output_size=self.n_hidden_neurons[i + 1],
				use_recurrent_connection=self.use_recurrent_connection,
				dt=self.dt,
				tau_syn=self.tau_syn,
				tau_mem=self.tau_mem,
				spike_func=self.spike_func,
				device=self.device,
			)

	def _add_readout_layer(self):
		if self.n_hidden_neurons:
			in_size = self.n_hidden_neurons[-1]
		else:
			in_size = self.input_size
		self.layers["readout"] = ReadoutLayer(
			input_size=in_size,
			output_size=self.output_size,
			dt=self.dt,
			tau_syn=self.tau_syn,
			tau_mem=self.tau_mem,
			spike_func=self.spike_func,
			device=self.device,
		)

	def _add_layers_(self):
		self._add_input_layer_()
		self._add_hidden_layers_()
		self._add_readout_layer()

	def initialize_weights_(self):
		for param in self.parameters():
			torch.nn.init.xavier_normal_(param)

	def _format_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
		if inputs.ndim == 2:
			inputs = torch.unsqueeze(inputs, 1)
		assert inputs.ndim == 3, "shape of inputs must be (batch size, time steps, nb features)"

		t_diff = self.int_time_steps - inputs.shape[1]
		assert t_diff >= 0, "inputs time steps must me less or equal to int_time_steps"
		if t_diff > 0:
			zero_inputs = torch.zeros((inputs.shape[0], t_diff, inputs.shape[-1]), device=self.device)
			inputs = torch.cat([inputs, zero_inputs], dim=1)
		return inputs

	def forward(self, inputs):
		inputs = self._format_inputs(inputs)
		hidden_states = {
			layer_name: [None for t in range(self.int_time_steps)]
			for layer_name, _ in self.layers.items()
		}
		outputs_trace = torch.zeros((inputs.shape[0], self.int_time_steps, self.output_size), device=self.device)

		for t in range(self.int_time_steps):
			forward_tensor = inputs[:, t]
			for layer_idx, (layer_name, layer) in enumerate(self.layers.items()):
				hh = hidden_states[layer_name][t - 1] if t > 0 else None
				forward_tensor, hidden_states[layer_name][t] = layer(forward_tensor, hh)
			outputs_trace[:, t] = forward_tensor

		return outputs_trace, hidden_states

	def fit(
			self,
			dataloader: DataLoader,
			lr=1e-3,
			nb_epochs=10,
			criterion=None,
			optimizer=None,
			load_checkpoint_mode: LoadCheckpointMode = None
	):
		self.train()
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		if criterion is None:
			criterion = nn.NLLLoss()
		if optimizer is None:
			optimizer = torch.optim.Adam(self.parameters(), lr=lr)

		loss_history = []
		start_epoch = 0
		if load_checkpoint_mode is not None:
			try:
				checkpoint = self.load_checkpoint(load_checkpoint_mode)
				self.load_state_dict(checkpoint[SNN.CHECKPOINT_STATE_DICT_KEY])
				optimizer.load_state_dict(checkpoint[SNN.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
				start_epoch = int(checkpoint[SNN.CHECKPOINT_EPOCH_KEY]) + 1
				loss_history = self.get_checkpoints_loss_history()
			except FileNotFoundError:
				logging.warning("No such checkpoint. Fit from beginning.")

		if start_epoch >= nb_epochs:
			return loss_history

		log_softmax_fn = nn.LogSoftmax(dim=1)
		best_loss = min(loss_history) if loss_history else np.inf
		start_time = time.time()
		printProgressBar(start_epoch, nb_epochs)
		for epoch in range(start_epoch, nb_epochs):
			epoch_loss = self._exec_epoch(
				dataloader,
				log_softmax_fn,
				criterion,
				optimizer,
			)
			loss_history.append(epoch_loss)
			is_best = epoch_loss < best_loss
			self.save_checkpoint(optimizer, epoch, float(epoch_loss), is_best)
			if is_best:
				best_loss = epoch_loss
			elapsed_time = time.time() - start_time
			printProgressBar(epoch + 1, nb_epochs, current_elapse_seconds=elapsed_time, suffix=f"{epoch_loss = :.5e}")
		return loss_history

	def _exec_epoch(
			self,
			dataloader,
			log_softmax_fn,
			criterion,
			optimizer,
	):
		batch_losses = []
		for x_batch, y_batch in dataloader:
			batch_loss = self._exec_batch(
				x_batch,
				y_batch,
				log_softmax_fn,
				criterion,
				optimizer,
			)
			batch_losses.append(batch_loss)
		return np.mean(batch_losses)

	def _exec_batch(
			self,
			x_batch,
			y_batch,
			log_softmax_fn,
			criterion,
			optimizer,
	):
		out, h_sates = self(x_batch.to(self.device))
		m, _ = torch.max(out, 1)
		log_p_y = log_softmax_fn(m)

		# Here we set up our regularizer loss
		# The strength parameters here are merely a guess and there should be ample room for improvement by
		# tuning these parameters.
		spikes = [h.Z for l_name, h_list in h_sates.items() for h in h_list if "Z" in h._asdict()]
		reg_loss = 1e-5 * sum([torch.sum(s) for s in spikes])  # L1 loss on total number of spikes
		reg_loss += 1e-5 * sum(
			[torch.mean(torch.sum(torch.sum(s, dim=0), dim=0) ** 2) for s in spikes]
		)  # L2 loss on spikes per neuron

		# Here we combine supervised loss and the regularizer
		batch_loss = criterion(log_p_y, y_batch.long().to(self.device)) + reg_loss

		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()
		return batch_loss.item()

	def _create_checkpoint_path(self, epoch: int = -1):
		return f"./{self.checkpoint_folder}/{self.model_name}{SNN.SUFFIX_SEP}{SNN.CHECKPOINT_EPOCH_KEY}{epoch}{SNN.SAVE_EXT}"

	def _create_new_checkpoint_meta(self, epoch: int, best: bool = False) -> dict:
		save_path = self._create_checkpoint_path(epoch)
		new_info = {SNN.CHECKPOINT_EPOCHS_KEY: {epoch: save_path}}
		if best:
			new_info[SNN.CHECKPOINT_BEST_KEY] = save_path
		return new_info

	def save_checkpoint(
			self,
			optimizer,
			epoch: int,
			epoch_loss: float = np.NaN,
			best: bool = False,
	):
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		save_path = self._create_checkpoint_path(epoch)
		torch.save({
			SNN.CHECKPOINT_EPOCH_KEY: epoch,
			SNN.CHECKPOINT_STATE_DICT_KEY: self.state_dict(),
			SNN.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
			SNN.CHECKPOINT_LOSS_KEY: epoch_loss,
		}, save_path)
		self.save_checkpoints_meta(self._create_new_checkpoint_meta(epoch, best))

	@staticmethod
	def get_save_path_from_checkpoints(
			checkpoint: Dict[str, Union[str, Dict[int, str]]],
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
	) -> str:
		if load_checkpoint_mode.BEST_EPOCH:
			return checkpoint[SNN.CHECKPOINT_BEST_KEY]
		elif load_checkpoint_mode.LAST_EPOCH:
			epochs_dict = checkpoint[SNN.CHECKPOINT_EPOCHS_KEY]
			last_epoch: int = max(epochs_dict)
			return checkpoint[SNN.CHECKPOINT_EPOCHS_KEY][last_epoch]
		else:
			raise ValueError()

	def get_checkpoints_loss_history(self):
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			meta: dict = json.load(jsonFile)
		checkpoints = [torch.load(path) for path in meta[SNN.CHECKPOINT_EPOCHS_KEY].values()]
		epoch_to_loss = {
			int(c[SNN.CHECKPOINT_EPOCH_KEY]): c[SNN.CHECKPOINT_LOSS_KEY]
			for c in checkpoints
		}
		history = [epoch_to_loss[k] for k in sorted(epoch_to_loss.keys())]
		return history

	def load_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
	) -> dict:
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		path = self.get_save_path_from_checkpoints(info, load_checkpoint_mode)
		return torch.load(path)

	def save_checkpoints_meta(self, new_info: dict):
		info = dict()
		if os.path.exists(self.checkpoints_meta_path):
			with open(self.checkpoints_meta_path, "r+") as jsonFile:
				info = json.load(jsonFile)
		mapping_update_recursively(info, new_info)
		with open(self.checkpoints_meta_path, "w+") as jsonFile:
			json.dump(info, jsonFile, indent=4)

	def compute_classification_accuracy(self, data: DataLoader) -> float:
		""" Computes classification accuracy on supplied data in batches. """
		self.eval()
		accs = []
		for x_local, y_local in data:
			out, _ = self(x_local.to(self.device))
			m, _ = torch.max(out.detach().cpu(), 1)  # max over time
			_, am = torch.max(m, 1)  # argmax over output units
			tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
			accs.append(tmp)
		return np.mean(accs)
