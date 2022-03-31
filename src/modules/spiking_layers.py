from typing import NamedTuple, Tuple, Type

import numpy as np
import torch
from torch import nn

from src.modules.spike_funcs import HeavisideSigmoidApprox, SpikeFunction


class RNNLayer(torch.nn.Module):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(RNNLayer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.use_recurrent_connection = use_recurrent_connection
		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.dt = dt
		self.kwargs = kwargs
		self._set_default_kwargs()

	def _set_default_kwargs(self):
		raise NotImplementedError()

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		raise NotImplementedError

	def _init_forward_state(
			self,
			state: Tuple[torch.Tensor, ...] = None,
			batch_size: int = 1
	) -> Tuple[torch.Tensor, ...]:
		if state is None:
			state = self.create_empty_state(batch_size)
		elif any([e is None for e in state]):
			empty_state = self.create_empty_state(batch_size)
			state = list(state)
			for i, e in enumerate(state):
				if e is None:
					state[i] = empty_state[i]
			state = tuple(state)
		return state

	def forward(self, inputs: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
		raise NotImplementedError

	def initialize_weights_(self):
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.normal_(param)


class LIFState(NamedTuple):
	"""
	V: membrane potential
	Z: spikes
	"""
	I: torch.Tensor
	V: torch.Tensor
	Z: torch.Tensor


class LIFLayer(RNNLayer):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(LIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			dt=dt,
			device=device,
			**kwargs
		)

		self.forward_weights = nn.Parameter(
			torch.empty((self.input_size, self.output_size), device=self.device, dtype=torch.float32),
			requires_grad=True
		)

		if use_recurrent_connection:
			self.recurrent_weights = nn.Parameter(
				torch.empty((self.output_size, self.output_size), device=self.device, dtype=torch.float32),
				requires_grad=True
			)
		else:
			self.recurrent_weights = None

		self.alpha = torch.tensor(np.exp(-dt / self.kwargs["tau_m"]), dtype=torch.float32, device=self.device)
		self.threshold = torch.tensor(self.kwargs["threshold"], dtype=torch.float32, device=self.device)
		self.gamma = torch.tensor(self.kwargs["gamma"], dtype=torch.float32, device=self.device)
		self.spike_func = spike_func
		self.initialize_weights_()

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_m", 20.0)
		self.kwargs.setdefault("threshold", 1.0)
		self.kwargs.setdefault("gamma", 1.0)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[[membrane potential of shape (batch_size, self.output_size)]
			[spikes of shape (batch_size, self.output_size)]]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		# state = LIFState(
		# 	V=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		# 	Z=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		# 	I=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		# )
		state = tuple([torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(2)])
		return state

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		# state = self._init_forward_state(state, batch_size)
		# next_state = self.create_empty_state(batch_size)
		V, Z = self._init_forward_state(state, batch_size)
		# next_V, next_Z = self.create_empty_state(batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, self.recurrent_weights)
		else:
			rec_current = 0.0
		next_V = self.alpha * V + input_current + rec_current - Z * self.threshold
		next_Z = self.spike_func.apply(next_V, self.threshold, self.gamma)
		return next_V, (next_V, next_Z)


class ALIFLayer(LIFLayer):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			spike_func: Type[SpikeFunction] = HeavisideSigmoidApprox,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(ALIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			spike_func=spike_func,
			dt=dt,
			device=device,
			**kwargs
		)
		self.beta = torch.tensor(np.exp(-dt / self.kwargs["tau_th"]), dtype=torch.float32, device=self.device)
		self.rho = torch.tensor(np.exp(-dt / self.kwargs["tau_a"]), dtype=torch.float32, device=self.device)

	def _set_default_kwargs(self):
		super(ALIFLayer, self)._set_default_kwargs()
		self.kwargs.setdefault("tau_a", 20.0)
		self.kwargs.setdefault("tau_th", 20.0)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[[membrane potential of shape (batch_size, self.output_size)]
			[current threshold of shape (batch_size, self.output_size)]
			[spikes of shape (batch_size, self.output_size)]]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		) for _ in range(3)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		V, a, Z = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(Z, self.recurrent_weights)
		else:
			rec_current = 0.0
		# v_j^{t+1} = \alpha * v_j^t + \sum_i W_{ji}*z_i^t + \sum_i W_{ji}^{in}x_i^{t+1} - z_j^t * v_{th}
		next_V = self.beta * V + input_current + rec_current - Z * self.threshold
		next_a = self.rho * a + Z  # a^{t+1} = \rho * a_j^t + z_j^t
		A = self.threshold + self.beta * next_a  # A_j^t = v_{th} + \beta * a_j^t
		next_Z = self.spike_func.apply(next_V, A, self.gamma)  # z_j^t = H(v_j^t - A_j^t)
		return next_Z, (next_V, next_a, next_Z)


class ReadoutState(NamedTuple):
	"""
	I: current
	V: membrane potential
	"""
	I: torch.Tensor
	V: torch.Tensor


class ReadoutLayer(RNNLayer):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(ReadoutLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=False,
			dt=dt,
			device=device,
			**kwargs
		)
		self.forward_weights = nn.Parameter(
			torch.empty((self.input_size, self.output_size), device=self.device, dtype=torch.float32),
			requires_grad=True,
		)
		self.bias_weights = nn.Parameter(
			torch.empty((self.output_size,), device=self.device),
			requires_grad=True,
		)
		self.kappa = torch.tensor(np.exp(-self.dt / self.kwargs["tau_out"]), dtype=torch.float32, device=self.device)
		self.initialize_weights_()

	def _set_default_kwargs(self):
		self.kwargs.setdefault("tau_out", 20.0)

	def initialize_weights_(self):
		super(ReadoutLayer, self).initialize_weights_()
		torch.nn.init.constant_(self.bias_weights, 0.0)

	def create_empty_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
		"""
		Create an empty state in the following form:
			[membrane potential of shape (batch_size, self.output_size)]
		:param batch_size: The size of the current batch.
		:return: The current state.
		"""
		state = [torch.zeros(
			(batch_size, self.output_size),
			device=self.device,
			dtype=torch.float32,
			requires_grad=True,
		)]
		return tuple(state)

	def forward(self, inputs: torch.Tensor, state: Tuple[torch.Tensor, ...] = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		# state = self._init_forward_state(state, batch_size)
		V, = self._init_forward_state(state, batch_size)
		next_V = self.kappa * V + torch.matmul(inputs, self.forward_weights) + self.bias_weights
		return next_V, (next_V, )
