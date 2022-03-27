import time

import numpy as np
import torch
from pythonbasictools.progress_bar import printProgressBar
from torch import nn
from torch.utils.data import DataLoader

from src.modules.spike_funcs import HeavisideSigmoidApprox
from src.modules.spiking_layers import LIFLayer, ReadoutLayer


class SNN(torch.nn.Module):
	def __init__(
			self,
			inputs_size: int,
			output_size: int,
			n_hidden_neurons=None,
			use_recurrent_connection=True,
			dt=1e-3,
			int_time_steps=1_000,
			tau_syn=10e-3,
			tau_mem=5e-3,
			spike_func=HeavisideSigmoidApprox.apply,
			device=None,
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

		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else []
		self.use_recurrent_connection = use_recurrent_connection
		self.layers = nn.ModuleDict()
		self._add_layers_()
		self.initialize_weights_()



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

	def forward(self, inputs):
		hidden_states = {
			layer_name: [None for t in range(self.int_time_steps)]
			for layer_name, _ in self.layers.items()
		}
		input_shape = inputs.shape  # TODO: check and ad time dim to be adaptative of single et time series input
		outputs_trace = torch.zeros((input_shape[0], self.int_time_steps, self.output_size), device=self.device)
		outputs = torch.flatten(inputs, start_dim=1)
		for layer_idx, (layer_name, layer) in enumerate(self.layers.items()):
			outputs, hidden_states[layer_name][0] = layer(outputs)
		outputs_trace[:, 0] = outputs

		for t in range(1, self.int_time_steps):
			outputs = torch.flatten(torch.zeros(input_shape), start_dim=1).to(self.device)
			for layer_idx, (layer_name, layer) in enumerate(self.layers.items()):
				outputs, hidden_states[layer_name][t] = layer(outputs, hidden_states[layer_name][t-1])
			outputs_trace[:, t] = outputs

		return outputs_trace, hidden_states

	def fit(
			self,
			data: DataLoader,
			lr=1e-3,
			nb_epochs=10,
			batch_size=256,
			criterion=None,
			optimizer=None,
	):
		if criterion is None:
			criterion = nn.NLLLoss()
		if optimizer is None:
			# optimizer = torch.optim.SGD(self.get_weights(), lr=lr)
			optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

		log_softmax_fn = nn.LogSoftmax(dim=1)
		start_time = time.time()
		loss_history = []
		for epoch in range(nb_epochs):
			epoch_loss = []
			for x_batch, y_batch in data:
				out, h_sates = self(x_batch.to(self.device))
				m, _ = torch.max(out, 1)
				log_p_y = log_softmax_fn(m)

				# Here we set up our regularizer loss
				# The strength parameters here are merely a guess and there should be ample room for improvement by
				# tuning these parameters.
				spikes = [h.Z for l_name, h_list in h_sates.items() for h in h_list if "Z" in h._asdict()]
				reg_loss = 1e-5 * sum([torch.sum(s) for s in spikes])  # L1 loss on total number of spikes
				reg_loss += 1e-5 * sum([torch.mean(torch.sum(torch.sum(s, dim=0), dim=0) ** 2) for s in spikes])  # L2 loss on spikes per neuron

				# Here we combine supervised loss and the regularizer
				batch_loss = criterion(log_p_y, y_batch.long().to(self.device)) + reg_loss

				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
				epoch_loss.append(batch_loss.item())
			mean_loss = np.mean(epoch_loss)
			elapsed_time = time.time() - start_time
			printProgressBar(epoch+1, nb_epochs, current_elapse_seconds=elapsed_time, suffix=f"{mean_loss = :.5e}")
			loss_history.append(mean_loss)

		return loss_history

	def compute_classification_accuracy(self, data: DataLoader, batch_size=256):
		""" Computes classification accuracy on supplied data in batches. """
		accs = []
		for x_local, y_local in data:
			out, spikes, _ = self(x_local)
			m, _ = torch.max(out, 1)  # max over time
			_, am = torch.max(m, 1)  # argmax over output units
			tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
			accs.append(tmp)
		return np.mean(accs)
