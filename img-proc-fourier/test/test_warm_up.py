'''
Created on Oct 29, 2014

@author: revaz
'''
import unittest
from fourier.Helper import Helper

class Test(unittest.TestCase):

    def test_euclidean_distance(self):
        
        self.assertAlmostEqual(Helper.euclidean_distance([1, 1], [2, 2]), 1.414, 3)
        self.assertAlmostEqual(Helper.euclidean_distance([1,1,1], [2,2,2]), 1.732, 3)
        self.assertRaises(RuntimeError, Helper.euclidean_distance, [1,1,1], [22,2])
        
        
    def test_draw_circle(self):
        original_image = Helper.read_image('bauckhage.jpg', as_array=True)
        new_image = Helper.draw_circle(original_image, 30, 50, True) 
        
        error_msg = 'Wrong area of the image has been modified'
        center = original_image.shape[0] / 2
        
        # check if the point which is not in the range of radii from the center is unchanged      
        self.assertEquals(original_image[center, center - 51], new_image[center, center - 51], error_msg)
        self.assertEquals(original_image[center, center - 29], new_image[center, center - 29], error_msg)
        self.assertEquals(original_image[center, center + 29], new_image[center, center + 29], error_msg)
        self.assertEquals(original_image[center, center + 51], new_image[center, center + 51], error_msg)
      
        # check if the point which is in the range of radii from the center is set to 0
        self.assertEquals(0, new_image[center, center - 49], error_msg)
        self.assertEquals(0, new_image[center, center - 31], error_msg)
        self.assertEquals(0, new_image[center, center + 31], error_msg)
        self.assertEquals(0, new_image[center, center + 49], error_msg)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()