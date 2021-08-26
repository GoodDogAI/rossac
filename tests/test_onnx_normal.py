import unittest
import torch
import numpy as np

from torch.distributions import Normal
from onnx_normal_dist import ONNXNormal

from hypothesis import given, strategies as st


class TestONNXNormalDistribution(unittest.TestCase):
    @given(st.floats(min_value=-1e9, max_value=1e9), st.floats(min_value=0.1, max_value=5))
    def test_sampling(self, mean, stddev):
        orig = Normal(loc=torch.Tensor([mean]), scale=torch.Tensor([stddev]))
        onnx = ONNXNormal(loc=torch.Tensor([mean]), scale=torch.Tensor([stddev]))

        orig_sample = orig.rsample(sample_shape=(100_000,))
        onnx_sample = onnx.rsample(sample_shape=(100_000,))

        self.assertLessEqual(abs(np.mean(orig_sample.numpy()) - mean), abs(mean) * 0.1 + 0.1)
        self.assertLessEqual(abs(np.mean(onnx_sample.numpy()) - mean), abs(mean) * 0.1 + 0.1)

        self.assertLessEqual(abs(np.std(orig_sample.numpy()) - stddev), abs(mean) * 0.1 + 0.1)
        self.assertLessEqual(abs(np.std(onnx_sample.numpy()) - stddev), abs(mean) * 0.1 + 0.1)

    @given(st.floats(min_value=0.1), st.floats(min_value=0.1), st.floats(allow_nan=False, allow_infinity=False))
    def test_logprob(self, mean, stddev, value):
        orig = Normal(loc=torch.Tensor([mean]), scale=torch.Tensor([stddev]))
        onnx = ONNXNormal(loc=torch.Tensor([mean]), scale=torch.Tensor([stddev]))

        np.testing.assert_almost_equal(orig.log_prob(torch.Tensor([value])),
                                       onnx.log_prob(torch.Tensor([value])))