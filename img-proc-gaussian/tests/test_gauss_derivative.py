__author__ = 'kostadin'

import unittest

import numpy as np
import calc.gaussian_mask as gm
import image_op.image_io as io



class MyTestCase(unittest.TestCase):
    def test_gauss_derivative(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        img_dx, img_dy = gm.gauss_derivatives(image, (9, 9))
          # np.testing.assert_array_almost_equal(img_dx, img_dy, 5)
        io.save_array_as_gray_image(img_dx, "../Generated/bauckhage_dx.jpg", normalize=True)
        io.save_array_as_gray_image(img_dx, "../Generated/bauckhage_dy.jpg", normalize=True)
        io.save_array_as_gray_image(np.sqrt(np.power(img_dx, 2) + np.power(img_dy, 2)),
                                    "../Generated/bauckhage_magnitude.jpg", normalize=True)

if __name__ == '__main__':
    unittest.main()
