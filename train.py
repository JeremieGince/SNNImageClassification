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

from src.datasets.datasets import dataset_to_timeseries
from src.modules.snn import ForwardMth
from src.modules.snn import LoadCheckpointMode, SNN
from torchviz import make_dot

from src.modules.spiking_layers import DynamicType


def norm_255(x):
	return x / 255.0


def get_dataloaders(
		dataset_name,
		batch_size=64,
		as_timeseries: bool = False,
		nb_workers: int = 0,
):
	transforms = Compose([
		ToTensor(),
		Lambda(norm_255),
		Lambda(torch.flatten)
	])
	if dataset_name.lower() == "mnist":
		root = os.path.expanduser("./data/datasets/torch/mnist")
		train_dataset = MNIST(root, train=True, download=True, transform=transforms)
		test_dataset = MNIST(root, train=False, download=True, transform=transforms)

	elif dataset_name.lower() == "fashion_mnist":
		root = os.path.expanduser("./data/datasets/torch/fashion-mnist")
		train_dataset = torchvision.datasets.FashionMNIST(
			root, train=True, transform=transforms, target_transform=None, download=True
		)
		test_dataset = torchvision.datasets.FashionMNIST(
			root, train=False, transform=transforms, target_transform=None, download=True
		)
	else:
		raise ValueError()

	if as_timeseries:
		train_dataloader = dataset_to_timeseries(
			train_dataset, batch_size=batch_size, nb_steps=100, tau_mem=20.0, shuffle=True
		)
		test_dataloader = dataset_to_timeseries(
			test_dataset, batch_size=batch_size, nb_steps=100, tau_mem=20.0, shuffle=False
		)
	else:
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

	ts = False
	d_name = f"mnist"
	logging.info(f"Dataset: {d_name}{'-ts' if ts else ''}")
	dataloaders = get_dataloaders(d_name, batch_size=256, as_timeseries=ts)

	snn = SNN(
		inputs_size=28 * 28,
		output_size=10,
		n_hidden_neurons=[100, ],
		int_time_steps=100,
		dt=1e-3,
		checkpoint_folder=f"checkpoints-{d_name}{'-ts' if ts else ''}-lt-bellec",
		forward_mth=ForwardMth.TIME_THEN_LAYER,
		dynamicType=DynamicType.Bellec,
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


