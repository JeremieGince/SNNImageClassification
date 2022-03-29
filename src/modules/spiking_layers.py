import time
from typing import Callable, Dict, NamedTuple

import numpy as np
import torch
from pythonbasictools.progress_bar import printProgressBar
from torch import nn
from torch.utils.data import DataLoader
from src.modules.spike_funcs import HeavisideSigmoidApprox
import enum

from src.modules.utils import SpikingInputSpec, SpikingInputType


class DynamicType(enum.Enum):
	Emre = enum.auto()
	Bellec = enum.auto()


class RNNLayer(torch.nn.Module):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			dynamicType: DynamicType = DynamicType.Emre,
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
		self.dynamicType = dynamicType
		self.kwargs = kwargs
		self._set_default_kwargs()

	def _set_default_kwargs(self):
		if self.dynamicType == DynamicType.Emre:
			self.kwargs.setdefault("tau_syn", 10e-3)
			self.kwargs.setdefault("tau_mem", 5e-3)
			self.kwargs.setdefault("threshold", 1.0)
		elif self.dynamicType == DynamicType.Bellec:
			self.kwargs.setdefault("tau_mem", 20)
			self.kwargs.setdefault("tau_out", 20)
			self.kwargs.setdefault("threshold", 1.0)

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def get_forward_func(self) -> Callable:
		dynamic_to_func = {
			DynamicType.Emre: self.forward_Erme,
			DynamicType.Bellec: self.forward_Bellec,
		}
		return dynamic_to_func[self.dynamicType]

	def create_empty_state(self, batch_size: int = 1):
		raise NotImplementedError

	def _init_forward_state(self, state: NamedTuple = None, batch_size: int = 1):
		if state is None:
			state = self.create_empty_state(batch_size)
		return state

	def forward_Erme(self, inputs: torch.Tensor, state: NamedTuple = None):
		raise NotImplementedError

	def forward_Bellec(self, inputs: torch.Tensor, state: NamedTuple = None):
		raise NotImplementedError

	def forward(self, inputs: torch.Tensor, state=None):
		return self.get_forward_func()(inputs, state)

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
			spike_func: torch.autograd.Function = HeavisideSigmoidApprox,
			dynamicType: DynamicType = DynamicType.Emre,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(LIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			dynamicType=dynamicType,
			dt=dt,
			device=device,
			**kwargs
		)

		self.forward_weights = nn.Parameter(
				torch.empty((self.input_size, self.output_size), device=self.device),
				requires_grad=True
		)

		if use_recurrent_connection:
			self.recurrent_weights = nn.Parameter(
				torch.empty((self.output_size, self.output_size), device=self.device),
				requires_grad=True
			)
		else:
			self.recurrent_weights = None

		self.alpha = np.exp(-dt / self.kwargs.get("tau_syn", 10e-3))
		self.beta = np.exp(-dt / self.kwargs.get("tau_mem", 20))
		self.threshold = self.kwargs.get("threshold", 1.0)
		self.spike_func = spike_func
		self.initialize_weights_()

	def create_empty_state(self, batch_size: int = 1) -> LIFState:
		state = LIFState(
			V=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
			Z=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
			I=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		)
		return state

	def forward_Erme(self, inputs: torch.Tensor, state: LIFState = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		state = self._init_forward_state(state, batch_size)

		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(state.Z, self.recurrent_weights)
		else:
			rec_current = 0.0

		is_active = 1.0 - state.Z.detach()
		next_I = self.alpha * state.I + input_current + rec_current
		next_V = (self.beta * state.V + next_I) * is_active
		next_Z = self.spike_func.apply(next_V - self.threshold)
		next_state = LIFState(I=next_I, V=next_V, Z=next_Z)
		return next_state.Z, next_state

	def forward_Bellec(self, inputs: torch.Tensor, state: LIFState = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		state = self._init_forward_state(state, batch_size)
		input_current = torch.matmul(inputs, self.forward_weights)
		if self.use_recurrent_connection:
			rec_current = torch.matmul(state.Z, self.recurrent_weights)
		else:
			rec_current = 0.0
		next_V = self.beta * state.V + input_current + rec_current - state.Z * self.threshold
		next_Z = self.spike_func.apply(next_V - self.threshold)
		next_state = LIFState(I=None, V=next_V, Z=next_Z)
		return next_state.Z, next_state


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
			dynamicType: DynamicType = DynamicType.Emre,
			dt=1e-3,
			device=None,
			**kwargs
	):
		super(ReadoutLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=False,
			dynamicType=dynamicType,
			dt=dt,
			device=device,
			**kwargs
		)
		self.forward_weights = nn.Parameter(
			torch.empty((self.input_size, self.output_size), device=self.device),
			requires_grad=True
		)
		self.bias_weights = nn.Parameter(
			torch.empty((self.output_size, ), device=self.device),
			requires_grad=True
		)
		self.alpha = np.exp(-dt / self.kwargs.get("tau_syn", 10e-3))
		self.beta = np.exp(-dt / self.kwargs.get("tau_mem", 5e-3))
		self.kappa = np.exp(-self.dt / self.kwargs.get("tau_out", 20.0))
		self.initialize_weights_()

	def initialize_weights_(self):
		super(ReadoutLayer, self).initialize_weights_()
		torch.nn.init.constant_(self.bias_weights, 0.0)

	def create_empty_state(self, batch_size: int = 1) -> ReadoutState:
		state = ReadoutState(
			V=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
			I=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		)
		return state

	def forward_Erme(self, inputs: torch.Tensor, state: ReadoutState = None):
		assert len(inputs.shape) == 2
		batch_size, nb_features = inputs.shape
		state = self._init_forward_state(state, batch_size)

		next_I = self.alpha * state.I + torch.matmul(inputs, self.forward_weights) + self.bias_weights
		next_V = self.beta * state.V + next_I
		next_state = ReadoutState(I=next_I, V=next_V)
		return next_state.V, next_state

	def forward_Bellec(self, inputs: torch.Tensor, state: ReadoutState = None):
		assert inputs.ndim == 2
		batch_size, nb_features = inputs.shape
		state = self._init_forward_state(state, batch_size)
		next_V = self.kappa * state.V + torch.matmul(inputs, self.forward_weights) + self.bias_weights
		next_state = ReadoutState(I=None, V=next_V)
		return next_state.V, next_state







