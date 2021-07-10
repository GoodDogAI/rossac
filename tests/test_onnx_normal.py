import unittest
import torch
import numpy as np

from torch.distributions import Normal
from onnx_normal_dist import ONNXNormal


class TestONNXNormalDistribution(unittest.TestCase):
    def test_sampling(self):
        orig = Normal(loc=torch.Tensor([1.0]), scale=torch.Tensor([2.0]))
        onnx = ONNXNormal(loc=torch.Tensor([1.0]), scale=torch.Tensor([2.0]))

        orig_sample = orig.sample(sample_shape=(100_000,))
        onnx_sample = onnx.sample(sample_shape=(100_000,))

        np.testing.assert_almost_equal(np.mean(orig_sample.numpy()), 1.0, decimal=2)
        np.testing.assert_almost_equal(np.mean(onnx_sample.numpy()), 1.0, decimal=2)


