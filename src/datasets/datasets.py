import enum
import os

import numpy as np
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor


class DatasetId(enum.Enum):
	MNIST = enum.auto()
	FASHION_MNIST = enum.auto()


class ToSpikes:
	def __init__(
			self,
			n_steps: int,
			t_max: float = None,
			tau=20.0 * 1e-3,
			thr=0.2,
			use_periods=False,
			epsilon=1e-7,
	):
		"""
		:param n_steps: The number of time step
		:param t_max:
		:param tau: The membrane time constant of the LIF neuron to be charged
		:param thr: The firing threshold value
		:param epsilon: A generic (small) epsilon > 0
		"""
		self.n_steps = n_steps
		self.t_max = n_steps if t_max is None else t_max
		self.tau = tau
		self.thr = thr
		self.epsilon = epsilon
		self.spikes_indices = None
		self.use_periods = use_periods
		self.spikes_gen_func = self.firing_periods_to_spikes if use_periods else self.firing_times_to_spikes

	def pixels_to_firing_periods(self, x: np.ndarray) -> np.ndarray:
		"""
		Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.
		:param x: the normalized between 0.0 and 1.0 input pixels
		:return: Time between spikes for each pixel of x
		"""
		# t_per_step = self.t_max / self.n_steps
		idx = x < self.thr
		x = np.clip(x, self.thr + self.epsilon, 1.0e9)
		T = self.tau * np.log(x / (x - self.thr))
		T[idx] = self.t_max
		# periods = T // t_per_step
		return T.astype(int)

	def firing_periods_to_spikes_loop(self, firing_periods: np.ndarray) -> np.ndarray:
		spikes = np.zeros((self.n_steps, *firing_periods.shape), dtype=float)
		starts = np.clip(firing_periods, 0, self.n_steps - 1).astype(int)
		for i, period in enumerate(firing_periods):
			spikes[np.arange(starts[i], self.n_steps, step=period, dtype=int), i] = 1.0
		return spikes

	def firing_periods_to_spikes_clip(self, firing_periods: np.ndarray) -> np.ndarray:
		if self.spikes_indices is None:
			self.spikes_indices = np.indices((self.n_steps, *firing_periods.shape))
		starts = np.clip(firing_periods, 0, self.n_steps - 1, dtype=int)
		# starts[starts > (self.n_steps - 1)] = self.n_steps - 1
		spikes_range = self.spikes_indices[0] - starts[self.spikes_indices[1]]
		spikes = ((spikes_range % firing_periods[self.spikes_indices[1]]) == 0) * (spikes_range >= 0)
		return spikes.astype(float)

	def firing_periods_to_spikes(self, firing_periods: np.ndarray) -> np.ndarray:
		if self.spikes_indices is None:
			self.spikes_indices = np.indices((self.n_steps, *firing_periods.shape))
		firing_periods[firing_periods > (self.n_steps - 1)] = self.n_steps - 1
		firing_periods[firing_periods < 1] = 1
		spikes_range = self.spikes_indices[0] - firing_periods[self.spikes_indices[1]]
		spikes = ((spikes_range % firing_periods[self.spikes_indices[1]]) == 0) * (spikes_range >= 0)
		return spikes.astype(float)

	def firing_times_to_spikes(self, firing_times: np.ndarray) -> np.ndarray:
		spikes = np.zeros((self.n_steps, *firing_times.shape))
		firing_times_mask = firing_times < self.n_steps
		pix_indexes_masked = np.arange(len(firing_times))[firing_times_mask]
		spikes[firing_times[firing_times_mask], pix_indexes_masked] = 1.
		return spikes

	def _format_inputs(self, x) -> np.ndarray:
		if isinstance(x, torch.Tensor):
			return x.numpy()
		return x

	def __call__(self, x) -> torch.Tensor:
		x = self._format_inputs(x)
		firing_periods: np.ndarray = self.pixels_to_firing_periods(x)
		spikes = self.spikes_gen_func(firing_periods)
		return torch.tensor(spikes)


def get_dataloaders(
		dataset_id: DatasetId,
		batch_size: int = 64,
		train_val_split_ratio: float = 0.85,
		as_timeseries: bool = True,
		n_steps: int = 100,
		to_spikes_use_periods: bool = False,
		nb_workers: int = 0,
):
	"""

	:param dataset_id:
	:param batch_size:
	:param train_val_split_ratio: The ratio of train data (i.e. train_length/data_length).
	:param as_timeseries:
	:param n_steps:
	:param to_spikes_use_periods:
	:param nb_workers:
	:return:
	"""
	list_of_transform = [
		ToTensor(),
		torch.flatten,
	]
	if as_timeseries:
		list_of_transform.append(ToSpikes(n_steps=n_steps, use_periods=to_spikes_use_periods))
	transform = Compose(list_of_transform)

	if dataset_id == DatasetId.MNIST:
		root = os.path.expanduser("./data/datasets/torch/mnist")
		train_dataset = MNIST(root, train=True, download=True, transform=transform)
		test_dataset = MNIST(root, train=False, download=True, transform=transform)
	elif dataset_id == DatasetId.FASHION_MNIST:
		root = os.path.expanduser("./data/datasets/torch/fashion-mnist")
		train_dataset = FashionMNIST(root, train=True, transform=transform, download=True)
		test_dataset = FashionMNIST(root, train=False, transform=transform, download=True)
	else:
		raise ValueError()

	train_length = int(len(train_dataset) * train_val_split_ratio)
	val_length = len(train_dataset) - train_length
	train_set, val_set = torch.utils.data.random_split(train_dataset, [train_length, val_length])

	train_dataloader = DataLoader(
		train_set, batch_size=batch_size, shuffle=True, num_workers=nb_workers
	)
	val_dataloader = DataLoader(
		val_set, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	)
	test_dataloader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	)
	return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)



