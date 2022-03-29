import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset


class TimeSeriesMNISTDataset(Dataset):
	def __init__(self, arr_timeseries, labels, transform=None, target_transform=None):
		self.timeseries_labels = labels
		self.timeseries_arrays = arr_timeseries
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.timeseries_labels)

	def __getitem__(self, idx):
		timeseries = self.timeseries_arrays[idx]
		label = self.timeseries_labels[idx]
		if self.transform:
			timeseries = self.transform(timeseries)
		if self.target_transform:
			label = self.target_transform(label)
		return timeseries, label


def current_to_timeseries(
		X,
		n_steps: int,
		t_max: float,
		tau=20.0,
		thr=0.2,  # todo trouver une méthode pour calculer le threshold ex moyenne des pixels
		epsilon=1e-7
):
	"""
	Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

	:param X: Current value
	:param tau: The membrane time constant of the LIF neuron to be charged
	:param thr: The firing threshold value
	:param n_steps: The number of time step
	:param t_max:
	:param epsilon: A generic (small) epsilon > 0
	:return: Time to first spike for each "current" x
	"""
	t_per_step = t_max/n_steps
	spikes = np.zeros((X.shape[0], n_steps), dtype=int)
	X = np.clip(X, thr + epsilon, 1e9)
	T = tau * np.log(X / (X - thr))
	indices = T // t_per_step
	for index, i in enumerate(indices):
		all_indices = np.arange(i, n_steps, step=i)
		spikes[index, all_indices] = 1
	return spikes


def timeseries_dataloader_generator(
		X,
		y,
		batch_size: int,
		n_steps: int,
		time_per_step: float,
		thr: float,
		tau_mem: float = 20,
		shuffle=True
):
	"""
	This generator takes datasets in analog format and generates spiking network input as sparse tensors.

	:param thr:
	:param X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
	:param y: Data labels
	:param batch_size:
	:param n_steps:
	:param time_per_step:
	:param tau_mem:
	:param shuffle:
	:return:
	"""

	labels_ = np.array(y)
	firing_times = current_to_timeseries(X, n_steps, time_per_step, tau_mem, thr)  #, dtype=int)
	dataset = TimeSeriesMNISTDataset(firing_times, labels_, transform=transforms.ToTensor())
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


########################################################################################################################


def current2firing_time(
		X,
		tau=20.0,
		thr=0.2,  # todo trouver une méthode pour calculer le threshold ex moyenne des pixels
		tmax=1.0,
		epsilon=1e-7
):
	"""
	Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

	:param X: Current value
	:param tau: The membrane time constant of the LIF neuron to be charged
	:param thr: The firing threshold value
	:param tmax: The maximum time returned
	:param epsilon: A generic (small) epsilon > 0
	:return: Time to first spike for each "current" x
	"""
	idx = X < thr
	X = np.clip(X, thr + epsilon, 1e9)
	T = tau * np.log(X / (X - thr))
	T[idx] = tmax
	# idx = torch.less(X, thr)
	# X = torch.clamp(X, thr + epsilon, 1e9)
	# T = tau * torch.log(X / (X - thr))
	return T


def sparse_dataloader_gen(
		X,
		y,
		batch_size: int,
		nb_steps: int,
		# nb_units: int,
		# time_per_step: float,
		# device,
		tau_mem: float = 20,
		shuffle=True
):
	"""
	This generator takes datasets in analog format and generates spiking network input as sparse tensors.

	:param X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
	:param y: Data labels
	:param batch_size:
	:param nb_steps:
	:param nb_units:
	:param time_per_step:
	:param device:
	:param tau_mem:
	:param shuffle:
	:return:
	"""

	labels_ = np.array(y)
	# number_of_batches = len(X) // batch_size
	# sample_index = np.arange(len(X))
	# compute discrete firing times
	# tau_eff = tau_mem / time_per_step
	firing_times = current2firing_time(X, tau=tau_mem, tmax=nb_steps)  #, dtype=int)
	dataset = TimeSeriesMNISTDataset(firing_times, labels_, transform=transforms.ToTensor())
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
	# unit_numbers = np.arange(nb_units)
	# if shuffle:
	# 	np.random.shuffle(sample_index)
	# total_batch_count = 0
	# counter = 0
	# while counter < number_of_batches:
	# 	batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
	#
	# 	coo = [[] for _ in range(3)]
	# 	for bc, idx in enumerate(batch_index):
	# 		c = firing_times[idx] < nb_steps
	# 		times, units = firing_times[idx][c], unit_numbers[c]
	#
	# 		batch = [bc for _ in range(len(times))]
	# 		coo[0].extend(batch)
	# 		coo[1].extend(times)
	# 		coo[2].extend(units)
	#
	# 	i = torch.LongTensor(coo).to(device)
	# 	v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
	#
	# 	X_batch = torch.sparse_coo_tensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
	# 	y_batch = torch.tensor(labels_[batch_index], device=device)
	#
	# 	yield X_batch.to(device=device), y_batch.to(device=device)
	#
	# 	counter += 1


def dataset_to_timeseries(dataset: Dataset, batch_size: int, nb_steps: int, tau_mem: float = 20e-3, shuffle=True):
	dataset.transform = transforms.Compose(
		[
			dataset.transform,
			np.array
		]
	)
	list_value_label = [dataset[i] for i in range(len(dataset))]
	values, labels = list(zip(*list_value_label))
	return sparse_dataloader_gen(np.array(values, dtype=float), np.array(labels), batch_size, nb_steps, tau_mem, shuffle)


if __name__ == '__main__':
	transform = transforms.Compose(
		[
			np.array,
			transforms.Lambda(lambda a: a/255)
		]
	)
	mnist_train = MNIST('./mnist', train=True, download=True, transform=transform)
	# mnist_test = MNIST('src/datasets/mnist', train=False, download=True, transform=transform)
	dataloader = dataset_to_timeseries(mnist_train, batch_size=60, nb_steps=100, tau_mem=20.0, shuffle=True)
	train_features, train_labels = next(iter(dataloader))
	print(type(train_features))
	# print(f"Feature batch shape: {train_features.size()}")
	# print(f"Labels batch shape: {train_labels.size()}")
	# data_train = torch.utils.data.DataLoader(
	# 	MNIST(
	# 		'src/datasets/mnist', train=True, download=True,
	# 		transform=transforms.Compose([
	# 			transforms.ToTensor()
	# 		])),
	# 	batch_size=64,
	# 	shuffle=True
	# )
	#
	# test_loader = torch.utils.data.DataLoader(
	# 	MNIST('src/datasets/mnist', train=False, download=True),
	# 	batch_size=batch_size, **kwargs)
