import tensorflow as tf
import numpy as np
from ops import squeeze2d, squeeze2d_old


class TestUtils(tf.test.TestCase):
    # create a notebook for testing out transpose in > 2 rank tensors
    def test_squeeze2d(self):
        arr = np.reshape(np.arange(0, 1*8*8*1, 1), [1, 8, 8, 1])
        x = tf.constant(arr)
        with self.session():
            squeeze_result = squeeze2d(x).eval()
            squeeze_result_old = squeeze2d_old(x).eval()
        import pdb; pdb.set_trace()
        self.fail()
