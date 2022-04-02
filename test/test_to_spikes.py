import numpy as np
import torch
from torchvision.transforms import Compose, Lambda, ToTensor

from src.datasets.datasets import ToSpikes


class TestToSpikes:
	def test_pixels_to_firing_periods_zeros(self):
		transform = ToSpikes(100, 100, tau=20.0, thr=0.2, epsilon=1e-7)
		x_zero = np.array([0.0])
		firing_periods_zero = transform.pixels_to_firing_periods(x_zero)
		assert np.all(firing_periods_zero == transform.n_steps)

	def test_pixels_to_firing_periods(self):
		transform = ToSpikes(100, 100, tau=20.0, thr=0.2, epsilon=1e-7)
		pix = np.array([0.82352941, 0.82745098, 0.83529412, 0.8745098, 0.8627451, 0.95294118, 0.79215686, 0., 0., 0.])
		firing_periods = np.array([5, 5, 5, 5, 5, 4, 5, 100, 100, 100, ])
		firing_periods_hat = transform.pixels_to_firing_periods(pix)
		assert np.allclose(firing_periods, firing_periods_hat)

	def test_pixels_to_firing_periods_2(self):
		transform = ToSpikes(10, 10, tau=20.0, thr=0.2, epsilon=1e-7)
		pix = np.array(
			[0.8627451, 0.90980392, 0.96470588, 0., 0.01176471, 0.79215686,
			 0.89411765, 0.87843137, 0.86666667, 0.82745098]
		)
		firing_periods = np.array([5, 4, 4, 10, 10, 5, 5, 5, 5, 5, ])
		firing_periods_hat = transform.pixels_to_firing_periods(pix)
		assert np.allclose(firing_periods, firing_periods_hat)

		firing_periods = torch.sparse_coo_tensor(
			indices=torch.tensor([[4, 4, 5, 5, 5, 5, 5, 5], [1, 2, 0, 5, 6, 7, 8, 9]]),
			values=torch.tensor([1., 1., 1., 1., 1., 1., 1., 1.]), size=(10, 10),
		).to_dense().numpy()
		print(f"{firing_periods = }")

	def test_call(self):
		transform = ToSpikes(10, 10, tau=20.0, thr=0.2, epsilon=1e-7)
		pix = np.array(
			[0.8627451, 0.90980392, 0.96470588, 0., 0.01176471,
			 0.79215686, 0.89411765, 0.87843137, 0.86666667, 0.82745098,
			 0.82745098, 0.83921569]
		)
		spikes = torch.sparse_coo_tensor(
			indices=torch.tensor([[4,  4,  5,  5,  5,  5,  5,  5,  5,  5], [1,  2,  0,  5,  6,  7,  8,  9, 10, 11]]),
			values=torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), size=(10, 12),
		).to_dense().numpy()
		spikes_hat = transform(pix)
		assert np.allclose(spikes, spikes_hat), f"{spikes_hat = }"

	def test_firing_times_to_spikes(self):
		transform = ToSpikes(10, 10, tau=20.0, thr=0.2, epsilon=1e-7)
		firing_times = np.array([5, 4, 4, 10, 10, 5, 5, 5, 5, 5, 5, 5])
		spikes = torch.sparse_coo_tensor(
			indices=torch.tensor([[4, 4, 5, 5, 5, 5, 5, 5, 5, 5], [1, 2, 0, 5, 6, 7, 8, 9, 10, 11]]),
			values=torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), size=(10, 12),
		).to_dense().numpy()
		spikes_hat = transform.firing_times_to_spikes(firing_times)
		assert np.allclose(spikes, spikes_hat), f"{spikes_hat = }"

	def test_firing_periods_to_spikes(self):
		transform = ToSpikes(5, 5)
		firing_periods = np.array([1, 2, transform.n_steps + 1])
		spikes = np.array([
			[0, 0, 0],
			[1, 0, 0],
			[1, 1, 0],
			[1, 0, 0],
			[1, 1, 1],
		])
		spikes_hat = transform.firing_periods_to_spikes(firing_periods)
		assert np.allclose(spikes, spikes_hat), f"{spikes_hat = }"

	def test_call_on_real(self):
		x_dict = np.load("test_x_to_spikes.npy", allow_pickle=True).item()
		transform = Compose([
			ToTensor(),
			Lambda(lambda x: x/255.),
			Lambda(torch.flatten),
			ToSpikes(100, 100, tau=20.0, thr=0.2, epsilon=1e-7)
		])
		assert np.allclose(x_dict["spikes"], transform(x_dict['x']).numpy())



