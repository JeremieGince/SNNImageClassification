import time

import numpy as np
import torch
from pythonbasictools.progress_bar import printProgressBar
from torch import nn
from torch.utils.data import DataLoader
from src.models.spike_funcs import HeavisideSigmoidApprox
import enum


class ReadoutMth(enum.Enum):
	RNN = 0


class ForwardMth(enum.Enum):
	LAYER_THEN_TIME = 0
	TIME_THEN_LAYER = 1


class SNN(torch.nn.Module):
	def __init__(
			self,
			inputs_size: int,
			output_size: int,
			n_hidden_neurons=None,
			use_recurrent_connection=True,
			dt=1e-3,
			# int_time_steps=100,  # TODO: move to dataset
			tau_syn=10e-3,
			tau_mem=5e-3,
			spike_func=HeavisideSigmoidApprox.apply,
			device=None,
			forward_mth=ForwardMth.LAYER_THEN_TIME,
			readout_mth=ReadoutMth.RNN,
	):
		super(SNN, self).__init__()
		self.input_size = inputs_size
		self.output_size = output_size

		self.device = device
		if self.device is None:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			else:
				self.device = torch.device("cpu")

		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else []
		self.forward_weights = []
		if self.n_hidden_neurons:
			self.forward_weights.append(
				torch.empty((inputs_size, self.n_hidden_neurons[0]), device=self.device, requires_grad=True)
			)
			for i, hn in enumerate(self.n_hidden_neurons[:-1]):
				self.forward_weights.append(
					torch.empty((hn, self.n_hidden_neurons[i + 1]), device=self.device, requires_grad=True)
				)
			self.readout_weights = torch.empty(
				(self.n_hidden_neurons[-1], output_size),
				device=self.device, requires_grad=True
			)
		else:
			self.readout_weights = torch.empty((inputs_size, output_size), device=self.device, requires_grad=True)

		self.use_recurrent_connection = use_recurrent_connection
		self.recurrent_weights = []
		if use_recurrent_connection:
			for i, hn in enumerate(self.n_hidden_neurons):
				self.recurrent_weights.append(torch.empty((hn, hn), device=self.device, requires_grad=True))

		for layer in self.get_weights():
			torch.nn.init.xavier_normal_(layer)

		# self.int_time_steps = int_time_steps
		self.dt = dt
		self.alpha = np.exp(-dt/tau_syn)
		self.beta = np.exp(-dt/tau_mem)
		self.spike_func = spike_func

		self.forward_func = self.get_forward_func(forward_mth)
		self.readout_func = self.get_readout_func(readout_mth)

	def get_readout_func(self, readout_mth: ReadoutMth):
		readout_mth_to_func = {
			ReadoutMth.RNN: self.forward_readout_rnn,
		}
		return readout_mth_to_func.get(readout_mth, ReadoutMth.RNN)

	def get_forward_func(self, forward_mth: ForwardMth):
		forward_mth_to_func = {
			ForwardMth.LAYER_THEN_TIME: self.forward_layer_time,
			ForwardMth.TIME_THEN_LAYER: self.forward_time_layer,
		}
		return forward_mth_to_func.get(forward_mth, ForwardMth.LAYER_THEN_TIME)

	def get_weights(self):
		return [*self.forward_weights, *self.recurrent_weights, self.readout_weights]

	def forward(self, inputs):
		spikes_records, membrane_potential_records = self.forward_func(inputs)
		output_records = self.readout_func(inputs, spikes_records, membrane_potential_records)
		return output_records, spikes_records, membrane_potential_records

	def forward_layer_time(self, inputs):
		membrane_potential_records = []
		spikes_records = [inputs, ]
		batch_size, nb_time_steps, nb_features = inputs.shape

		# Compute hidden layers activity
		for ell, f_weights in enumerate(self.forward_weights):
			h_ell = torch.einsum("btf, fo -> bto", (spikes_records[-1], f_weights))

			forward_current = torch.zeros((batch_size, f_weights.shape[-1]), device=self.device, dtype=torch.float)
			forward_potential = torch.zeros((batch_size, f_weights.shape[-1]), device=self.device, dtype=torch.float)
			spikes = torch.zeros((batch_size, f_weights.shape[-1]), device=self.device, dtype=torch.float)

			local_membrane_potential_records = []
			local_spikes_records = []

			for t in range(h_ell.shape[1]):
				if self.use_recurrent_connection:
					current_recurrent = torch.einsum("bo, of -> bf", (spikes, self.recurrent_weights[ell]))
				else:
					current_recurrent = 0.0

				spikes = self.spike_func(forward_potential - 1.0)
				is_active = 1.0 - spikes.detach()

				forward_current = self.alpha * forward_current + h_ell[:, t] + current_recurrent
				forward_potential = (self.beta * forward_potential + forward_current) * is_active

				local_membrane_potential_records.append(forward_potential)
				local_spikes_records.append(spikes)

			membrane_potential_records.append(torch.stack(local_membrane_potential_records, dim=1))
			spikes_records.append(torch.stack(local_spikes_records, dim=1))

		return spikes_records[1:], membrane_potential_records

	def forward_readout_rnn(self, inputs, spikes_records=None, membrane_potential_records=None):
		batch_size, nb_time_steps, nb_features = inputs.shape

		h_out = torch.einsum("btf, fo -> bto", (spikes_records[-1], self.readout_weights))
		forward_current = torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float)
		forward_potential = torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float)
		output_records = [forward_potential, ]

		for t in range(h_out.shape[1]):
			forward_current = self.alpha * forward_current + h_out[:, t]
			forward_potential = (self.beta * forward_potential + forward_current)
			output_records.append(forward_potential)

		return torch.stack(output_records, dim=1)

	def forward_time_layer(self, inputs):
		membrane_potential_records = []
		spikes_records = []
		batch_size, input_size = inputs.shape

		forward_synaptic_currents = [
			torch.zeros((batch_size, f_weights.shape[0]), device=self.device, dtype=torch.float)
			for i, f_weights in enumerate(self.forward_weights)
		]
		forward_membrane_potentials = [
			torch.zeros((batch_size, f_weights.shape[0]), device=self.device, dtype=torch.float)
			for i, f_weights in enumerate(self.forward_weights)
		]
		spikes = [
			torch.zeros((batch_size, f_weights.shape[0]), device=self.device, dtype=torch.float)
			for i, f_weights in enumerate(self.forward_weights)
		]
		spikes[0] = inputs

		for t in range(self.int_time_steps):
			for ell, f_weights in enumerate(self.forward_weights):
				h_ell = torch.dot(spikes[ell], f_weights)
				forward_synaptic_currents[ell] = self.alpha * forward_membrane_potentials[ell] + h_ell
				forward_membrane_potentials[ell] = self.beta * forward_membrane_potentials[ell] + forward_synaptic_currents[ell]
				out = self.spike_func(forward_membrane_potentials[ell])
				spikes[ell] = out.detach()

	# def current2firing_time(self, x, tau=20.0, thr=0.2, tmax=1.0, epsilon=1e-7):
	# 	""" Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.
	#
	# 	Args:
	# 	x -- The "current" values
	#
	# 	Keyword args:
	# 	tau -- The membrane time constant of the LIF neuron to be charged
	# 	thr -- The firing threshold value
	# 	tmax -- The maximum time returned
	# 	epsilon -- A generic (small) epsilon > 0
	#
	# 	Returns:
	# 	Time to first spike for each "current" x
	# 	"""
	# 	idx = x < thr
	# 	x = np.clip(x, thr + epsilon, 1e9)
	# 	T = tau * np.log(x / (x - thr))
	# 	T[idx] = tmax
	# 	return T
	#
	# def sparse_data_generator(self, X, y, batch_size, nb_steps, nb_units, shuffle=True):
	# 	""" This generator takes datasets in analog format and generates spiking network input as sparse tensors.
	#
	# 	Args:
	# 		X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
	# 		y: The labels
	# 	"""
	#
	# 	labels_ = np.array(y, dtype=np.int)
	# 	number_of_batches = len(X) // batch_size
	# 	sample_index = np.arange(len(X))
	#
	# 	# compute discrete firing times
	# 	tau_eff = 20e-3 / self.dt
	# 	firing_times = np.array(self.current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=int)
	# 	unit_numbers = np.arange(nb_units)
	#
	# 	if shuffle:
	# 		np.random.shuffle(sample_index)
	#
	# 	total_batch_count = 0
	# 	counter = 0
	# 	while counter < number_of_batches:
	# 		batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
	#
	# 		coo = [[] for i in range(3)]
	# 		for bc, idx in enumerate(batch_index):
	# 			c = firing_times[idx] < nb_steps
	# 			times, units = firing_times[idx][c], unit_numbers[c]
	#
	# 			batch = [bc for _ in range(len(times))]
	# 			coo[0].extend(batch)
	# 			coo[1].extend(times)
	# 			coo[2].extend(units)
	#
	# 		i = torch.LongTensor(coo).to(self.device)
	# 		v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)
	#
	# 		X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(self.device)
	# 		y_batch = torch.tensor(labels_[batch_index], device=self.device)
	#
	# 		yield X_batch.to(device=self.device), y_batch.to(device=self.device)
	#
	# 		counter += 1

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
			optimizer = torch.optim.Adamax(self.get_weights(), lr=lr, betas=(0.9, 0.999))

		log_softmax_fn = nn.LogSoftmax(dim=1)
		start_time = time.time()
		loss_history = []
		for epoch in range(nb_epochs):
			epoch_loss = []
			for x_batch, y_batch in data:
				out, spikes, potential = self(x_batch.to_dense())
				m, _ = torch.max(out, 1)
				log_p_y = log_softmax_fn(m)

				# Here we set up our regularizer loss
				# The strength parameters here are merely a guess and there should be ample room for improvement by
				# tuning these parameters.
				reg_loss = 1e-5 * sum([torch.sum(s) for s in spikes])  # L1 loss on total number of spikes
				reg_loss += 1e-5 * sum([torch.mean(torch.sum(torch.sum(s, dim=0), dim=0) ** 2) for s in spikes])  # L2 loss on spikes per neuron

				# Here we combine supervised loss and the regularizer
				batch_loss = criterion(log_p_y, y_batch.long()) + reg_loss

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
			out, spikes, _ = self(x_local.to_dense())
			m, _ = torch.max(out, 1)  # max over time
			_, am = torch.max(m, 1)  # argmax over output units
			tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
			accs.append(tmp)
		return np.mean(accs)

