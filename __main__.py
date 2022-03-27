import os

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from src.modules.snn import SNN


if __name__ == '__main__':
	# Here we load the Dataset
	root = os.path.expanduser("./data/datasets/torch/fashion-mnist")
	train_dataset = torchvision.datasets.FashionMNIST(
		root, train=True, transform=None, target_transform=None, download=True
	)
	test_dataset = torchvision.datasets.FashionMNIST(
		root, train=False, transform=None, target_transform=None, download=True
	)
	# Standardize data
	# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
	x_train = np.array(train_dataset.data, dtype=float)
	x_train = x_train.reshape(x_train.shape[0], -1) / 255
	# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
	x_test = np.array(test_dataset.data, dtype=float)
	x_test = x_test.reshape(x_test.shape[0], -1) / 255

	# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
	# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
	y_train = np.array(train_dataset.targets, dtype=int)
	y_test = np.array(test_dataset.targets, dtype=int)

	snn = SNN(inputs_size=28*28, output_size=10, n_hidden_neurons=[100, ], int_time_steps=100)
	loss_hist = snn.fit(x_train, y_train, lr=1e-2, nb_epochs=30)

	print("Training accuracy: %.3f" % (snn.compute_classification_accuracy(x_train, y_train)))
	print("Test accuracy: %.3f" % (snn.compute_classification_accuracy(x_test, y_test)))

	plt.figure(figsize=(3.3, 2), dpi=150)
	plt.plot(loss_hist)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	sns.despine()
	plt.show()

