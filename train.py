import os

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from pythonbasictools.device import log_pytorch_device_setup
from pythonbasictools.logging import logs_file_setup
from src.modules.snn_v2 import SNN


if __name__ == '__main__':
	logs_file_setup(__file__)
	log_pytorch_device_setup()

	transforms = Compose([
		ToTensor(),
		Lambda(lambda x: x/255.0)
	])

	# Here we load the Dataset
	root = os.path.expanduser("./data/datasets/torch/fashion-mnist")
	train_dataset = torchvision.datasets.FashionMNIST(
		root, train=True, transform=transforms, target_transform=None, download=True
	)
	test_dataset = torchvision.datasets.FashionMNIST(
		root, train=False, transform=transforms, target_transform=None, download=True
	)

	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

	snn = SNN(inputs_size=28*28, output_size=10, n_hidden_neurons=[100, ], int_time_steps=100)
	loss_hist = snn.fit(train_dataloader, lr=1e-2, nb_epochs=30)

	print("Training accuracy: %.3f" % (snn.compute_classification_accuracy(train_dataloader)))
	print("Test accuracy: %.3f" % (snn.compute_classification_accuracy(test_dataloader)))

	plt.figure(figsize=(3.3, 2), dpi=150)
	plt.plot(loss_hist)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	sns.despine()
	plt.show()

