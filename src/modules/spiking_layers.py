import time
from typing import Dict, NamedTuple

import numpy as np
import torch
from pythonbasictools.progress_bar import printProgressBar
from torch import nn
from torch.utils.data import DataLoader
from src.modules.spike_funcs import HeavisideSigmoidApprox
import enum

from src.modules.utils import SpikingInputSpec, SpikingInputType


class RNNLayer(torch.nn.Module):
	def __init__(
			self,
			input_size: int,
			output_size: int,
			use_recurrent_connection=True,
			dt=1e-3,
			device=None,
	):
		super(RNNLayer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.use_recurrent_connection = use_recurrent_connection
		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.dt = dt

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def create_empty_state(self, batch_size: int = 1):
		raise NotImplementedError

	def _init_forward_state(self, state: NamedTuple = None, batch_size: int = 1):
		if state is None:
			state = self.create_empty_state(batch_size)
		return state

	def forward(self, inputs: torch.Tensor, state: NamedTuple = None):
		raise NotImplementedError

	def initialize_weights_(self):
		for param in self.parameters():
			torch.nn.init.xavier_normal_(param)


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
			dt=1e-3,
			tau_syn=10e-3,
			tau_mem=5e-3,
			spike_func=HeavisideSigmoidApprox.apply,
			device=None,
	):
		super(LIFLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=use_recurrent_connection,
			dt=dt,
			device=device,
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

		self.alpha = np.exp(-dt / tau_syn)
		self.beta = np.exp(-dt / tau_mem)
		self.spike_func = spike_func

	def check_inputs_valibility(self, inputs: Dict[str, torch.Tensor]):
		assert isinstance(inputs, Dict), "inputs must be a type of Dict[str, torch.Tensor]."
		assert all([isinstance(v, torch.Tensor) for _, v in inputs.items()]),\
			"inputs must be a type of Dict[str, torch.Tensor]."
		assert self.input_spec.iType.name in inputs, "inputs must contain a key which must be a SpikingInputType."

	def create_empty_state(self, batch_size: int = 1) -> LIFState:
		state = LIFState(
			V=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
			Z=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
			I=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		)
		return state

	def forward(self, inputs: torch.Tensor, state: LIFState = None):
		assert len(inputs.shape) == 2
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
		next_Z = self.spike_func(next_V - 1.0)
		next_state = LIFState(I=next_I, V=next_V, Z=next_Z)
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
			dt=1e-3,
			tau_syn=10e-3,
			tau_mem=5e-3,
			spike_func=HeavisideSigmoidApprox.apply,
			device=None,
	):
		super(ReadoutLayer, self).__init__(
			input_size=input_size,
			output_size=output_size,
			use_recurrent_connection=False,
			dt=dt,
			device=device,
		)
		self.forward_weights = nn.Parameter(
			torch.empty((self.input_size, self.output_size), device=self.device),
			requires_grad=True
		)
		self.alpha = np.exp(-dt / tau_syn)
		self.beta = np.exp(-dt / tau_mem)
		self.spike_func = spike_func

	def create_empty_state(self, batch_size: int = 1) -> ReadoutState:
		state = ReadoutState(
			V=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
			I=torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float),
		)
		return state

	def forward(self, inputs: torch.Tensor, state: ReadoutState = None):
		assert len(inputs.shape) == 2
		batch_size, nb_features = inputs.shape
		state = self._init_forward_state(state, batch_size)

		next_I = self.alpha * state.I + torch.matmul(inputs, self.forward_weights)
		next_V = self.beta * state.V + next_I
		next_state = ReadoutState(I=next_I, V=next_V)
		return next_state.V, next_state








