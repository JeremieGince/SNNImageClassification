import numpy as np
import torch

from src.modules.utils import batchwise_temporal_filter


class TestTemporalFilter:
	def test_batchwise_temporal_filter(self):
		# test the batchwise_temporal_filter function
		# test with a batch of size 1
		decay = 0.9
		input_data = torch.tensor([
			[
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9]
			]
		])
		output_data = torch.tensor([
			[7 + decay*(4 + decay*1), 8 + decay*(5 + decay*2), 9 + decay*(6 + decay*3)]
		])
		output_hat = batchwise_temporal_filter(input_data, decay)
		assert np.allclose(output_hat.numpy(), output_data.numpy())

	def test_batchwise_temporal_filter_identity(self):
		decay = 1.0
		input_data = torch.tensor([
			[
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9]
			]
		])
		output_data = torch.sum(input_data, dim=1)
		output_hat = batchwise_temporal_filter(input_data, decay)
		assert np.allclose(output_hat.numpy(), output_data.numpy())


