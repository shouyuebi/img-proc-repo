import sys

from image_op.image_io import read_image
from plot import plot_multiple_arrays


def task_1():
    print 'task 1'
    test_image = read_image("Resources/bauckhage.jpg", as_array=True)
    plot_multiple_arrays([[test_image]], "Task 1", ["Test Image"])

def main(argv):
    task_1()

if __name__ == '__main__':
    main(sys.argv)