import logging
import os

import matplotlib.pyplot as plt
import psutil
import torch
import torchvision
from pythonbasictools.device import log_pytorch_device_setup
from pythonbasictools.logging import logs_file_setup
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from src.datasets.datasets import ToSpikes, dataset_to_timeseries
from src.modules.snn import ForwardMth
from src.modules.snn import LoadCheckpointMode, SNN
from torchviz import make_dot

from src.modules.spike_funcs import HeavisidePhiApprox


def norm_255(x):
	return x / 255.0


def get_dataloaders(
		dataset_name,
		batch_size=64,
		as_timeseries: bool = False,
		dt=0.1,
		T=100,
		nb_workers: int = 0,
):
	list_of_transform = [
		ToTensor(),
		Lambda(norm_255),
		Lambda(torch.flatten)
	]
	if as_timeseries:
		list_of_transform.append(ToSpikes(n_steps=T, t_max=dt))
	transform = Compose(list_of_transform)

	if dataset_name.lower() == "mnist":
		root = os.path.expanduser("./data/datasets/torch/mnist")
		train_dataset = MNIST(root, train=True, download=True, transform=transform)
		test_dataset = MNIST(root, train=False, download=True, transform=transform)

	elif dataset_name.lower() == "fashion_mnist":
		root = os.path.expanduser("./data/datasets/torch/fashion-mnist")
		train_dataset = torchvision.datasets.FashionMNIST(
			root, train=True, transform=transform, target_transform=None, download=True
		)
		test_dataset = torchvision.datasets.FashionMNIST(
			root, train=False, transform=transform, target_transform=None, download=True
		)
	else:
		raise ValueError()

	# if as_timeseries:
	# 	train_dataloader = dataset_to_timeseries(
	# 		train_dataset, batch_size=batch_size, nb_steps=T, tau_mem=20.0, dt=dt, shuffle=True
	# 	)
	# 	test_dataloader = dataset_to_timeseries(
	# 		test_dataset, batch_size=batch_size, nb_steps=T, tau_mem=20.0, dt=dt, shuffle=False
	# 	)
	# else:
	# 	train_dataloader = DataLoader(
	# 		train_dataset, batch_size=batch_size, shuffle=True, num_workers=nb_workers
	# 	)
	# 	test_dataloader = DataLoader(
	# 		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	# 	)
	train_dataloader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True, num_workers=nb_workers
	)
	test_dataloader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=nb_workers
	)
	return dict(train=train_dataloader, test=test_dataloader)


if __name__ == '__main__':
	logs_file_setup(__file__)
	log_pytorch_device_setup()

	torch.autograd.set_detect_anomaly(True)

	delta_t = 1e-3
	n_steps = 100

	ts = True
	d_name = f"mnist"
	logging.info(f"Dataset: {d_name}{'-ts' if ts else ''}")
	dataloaders = get_dataloaders(
		d_name,
		batch_size=256,
		as_timeseries=ts,
		dt=delta_t,
		T=n_steps,
		# nb_workers=psutil.cpu_count(logical=False),
	)

	snn = SNN(
		inputs_size=28 * 28,
		output_size=10,
		n_hidden_neurons=[100, ],
		int_time_steps=n_steps,
		dt=delta_t,
		spike_func=HeavisidePhiApprox,
		checkpoint_folder=f"checkpoints-{d_name}{'-ts' if ts else ''}-profile",
	)
	# x_viz, _ = next(iter(dataloaders["train"]))
	# out_viz, _ = snn(x_viz.to(snn.device))
	# print(make_dot(out_viz).render("figures/snn_torchviz", format="png"))
	# snn.to_onnx()
	loss_hist = snn.fit(
		dataloaders["train"],
		lr=1e-3,
		nb_epochs=100,
		load_checkpoint_mode=LoadCheckpointMode.LAST_EPOCH,
		force_overwrite=True,
		# optimizer=torch.optim.SGD(snn.parameters(), lr=1e-3, nesterov=True, momentum=0.9)
	)
	snn.load_checkpoint(LoadCheckpointMode.BEST_EPOCH)

	train_acc = snn.compute_classification_accuracy(dataloaders["train"])
	test_acc = snn.compute_classification_accuracy(dataloaders["test"])
	logging.info(f"Training accuracy: {train_acc:.3f}")
	logging.info(f"Test accuracy: {test_acc:.3f}")

	plt.figure(figsize=(5, 3), dpi=300)
	plt.plot(loss_hist)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	# sns.despine()
	plt.show()


