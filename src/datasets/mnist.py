import numpy as np
import torch


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
	""" This generator takes datasets in analog format and generates spiking network input as sparse tensors.

	Args:
		X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
		y: The labels
	"""

	labels_ = np.array(y, dtype=np.int)
	number_of_batches = len(X) // batch_size
	sample_index = np.arange(len(X))

	# compute discrete firing times
	tau_eff = 20e-3 / time_step
	firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=int)
	unit_numbers = np.arange(nb_units)

	if shuffle:
		np.random.shuffle(sample_index)

	total_batch_count = 0
	counter = 0
	while counter < number_of_batches:
		batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

		coo = [[] for i in range(3)]
		for bc, idx in enumerate(batch_index):
			c = firing_times[idx] < nb_steps
			times, units = firing_times[idx][c], unit_numbers[c]

			batch = [bc for _ in range(len(times))]
			coo[0].extend(batch)
			coo[1].extend(times)
			coo[2].extend(units)

		i = torch.LongTensor(coo).to(device)
		v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

		X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
		y_batch = torch.tensor(labels_[batch_index], device=device)

		yield X_batch.to(device=device), y_batch.to(device=device)

		counter += 1