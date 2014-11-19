from image_op.image_io import read_image
from image_op.mask import Mask
from image_op.euclidean_ring_mask import EucRingMask
from calc.distance import manhattan_2d_array
from plot import plot_multiple_arrays

DEFAULT_IMAGES = ['Resources/bauckhage.jpg',
                  'Resources/clock.jpg',
                  'Resources/cat.png',
                  'Resources/asterixGrey.jpg']


def task_1_manhattan(image_path, radius_min, radius_max):
    input_image = read_image(image_path, as_array=True)

    manhattan_dist_array = manhattan_2d_array(input_image.shape, (input_image.shape[0] / 2, input_image.shape[1] / 2))
    manhattan_ring_condition = (manhattan_dist_array >= radius_min) != (manhattan_dist_array >= radius_max)
    mask = Mask(input_image.shape,
                lambda x: 0, lambda x: x,
                manhattan_ring_condition)

    output_image = mask.apply_mask(input_image)

    plot_multiple_arrays([[input_image, output_image]], "Task 1 Manhattan", ["Input Image", "Output Image"])


def task_1_shifted(image_path, radius_min, radius_max, center):
    input_image = read_image(image_path, as_array=True)

    manhattan_dist_array = manhattan_2d_array(input_image.shape, center)
    manhattan_ring_condition = (manhattan_dist_array >= radius_min) != (manhattan_dist_array >= radius_max)
    manhattan_ring_mask = Mask(input_image.shape,
                               lambda x: 0, lambda x: x,
                               manhattan_ring_condition)

    euc_ring_mask = EucRingMask(input_image.shape,
                                lambda x: 0, lambda x: x,
                                radius_min, radius_max, center)

    output_image_manhattan = manhattan_ring_mask.apply_mask(input_image)
    output_image_euclidean = euc_ring_mask.apply_mask(input_image)

    plot_multiple_arrays([[input_image, output_image_manhattan, output_image_euclidean]],
                         "Task 1 Manhattan",
                         ["Input Image", "Output Image Manhattan", "Output Image Euclidean"])


def task_1_reverse(image_path, radius_min, radius_max):
    input_image = read_image(image_path, as_array=True)

    center = (input_image.shape[0] / 2, input_image.shape[1] / 2)

    manhattan_dist_array = manhattan_2d_array(input_image.shape, center)
    manhattan_ring_condition = (manhattan_dist_array >= radius_min) != (manhattan_dist_array >= radius_max)
    manhattan_ring_mask = Mask(input_image.shape,
                               lambda x: x, lambda x: 0,
                               manhattan_ring_condition)

    euc_ring_mask = EucRingMask(input_image.shape,
                                lambda x: x, lambda x: 0,
                                radius_min, radius_max, center)

    output_image_manhattan = manhattan_ring_mask.apply_mask(input_image)
    output_image_euclidean = euc_ring_mask.apply_mask(input_image)

    plot_multiple_arrays([[input_image, output_image_manhattan, output_image_euclidean]],
                         "Task 1 Manhattan",
                         ["Input Image", "Output Image Manhattan", "Output Image Euclidean"])


if __name__ == '__main__':
    # task_1_manhattan(DEFAULT_IMAGES[0], 25, 55)
    # task_1_shifted(DEFAULT_IMAGES[1], 25, 55, (133, 96))
    task_1_reverse(DEFAULT_IMAGES[0], 25, 55)