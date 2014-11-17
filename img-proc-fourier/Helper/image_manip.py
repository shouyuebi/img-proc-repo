import numpy as np

from Assert import Assert
import fourier_calc as fourier_calc
from distance import euclidean_2points


def draw_circle(picture_array, r_min, r_max, inside):
    """
    This method draw a black circle in a picture_array. Client must provide the min radius and the max radius of a circle, picture_array.
    @param picture_array: on which will the circle will be drawn,
    @param r_min: min radius of a circle
    @param r_max: max radius of a circle
    @param inside: true if the inside of the circle will be black, otherwise the circle will retain the value from the pixels of the picture_array and the rest of the picture_array will be black 
    """
    for i in range(picture_array.shape[0]):
        for j in range(picture_array.shape[1]):
            norm = euclidean_2points([i, j], [picture_array.shape[0] / 2, picture_array.shape[1] / 2])
            is_in_radius = (norm >= r_min and norm <= r_max)
            if inside == False:
                if is_in_radius == False:
                    picture_array[i, j] = 1                   
            else:
                if is_in_radius == True:
                    picture_array[i, j] = 0                       
    return picture_array

def combine_magnitude_and_phase(magnitude_image, phase_image):
    """
    This method creates new image by combining the magnitude form **magnitude_image** and phase from **phase_image**. Client must provide
    two images with same resolution.
    @param magnitude_image: first image
    @param phase_image: second image
    @return: new image
    """
    Assert.isTrue(magnitude_image.size == phase_image.size, "Images have different resolution!")
    ftImageA = np.fft.fft2(magnitude_image)
    ftImageB = np.fft.fft2(phase_image)
    
    magnitude = fourier_calc.magnitude(ftImageA)
    phase = fourier_calc.phase(ftImageB)
    imageC = fourier_calc.create_complex_number(magnitude, phase)

    return np.fft.ifft2(imageC)