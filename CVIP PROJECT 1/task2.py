"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
and the functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    # raise NotImplementedError

	# Function to calculate mean of patch and template
    def calculate_mean(matrix=None):
        if matrix is None:
            print("Error!")
        total_elements = len(matrix)*len(matrix[0])
        return sum([n for m in matrix for n in m])/total_elements

	# Calculate mean of template and patch
    template_mean = calculate_mean(matrix=template)
    patch_mean = calculate_mean(matrix=patch)

    t_diff_matrix = []
    p_diff_matrix = []

	# Subtracting mean from each element from template and patch
    for i in template:
        for j in i:
            t_diff_matrix.append(j - template_mean)
    for i in patch:
        for j in i:
            p_diff_matrix.append(j - patch_mean)

    num = 0
    t_sum = 0
    p_sum = 0

	# Implementing Normalized Cross Convolution
	# Calculating the numerator value
	# Multiplying element of template with element of patch(after mean substraction)
	# Squaring the element value and summing all the elements for both template and patch
    for i,val in enumerate(t_diff_matrix):
        num = num + val*p_diff_matrix[i]
        t_sum = t_sum + val*val
        p_sum = p_sum + p_diff_matrix[i]*p_diff_matrix[i]

	# Calculating the denominator value
	# Taking square root of the product of sum of template elements and patch elements
    denum = np.sqrt(t_sum*p_sum)

    return num/denum


def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    # TODO: implement this function.
    # raise NotImplementedError
    # raise NotImplementedError

	# Calculating rows and columns of the template

    rows = len(img) - len(template)
    cols = len(img[0]) - len(template[0])
    norm_list = []

	# Create a patch of image
	# Performing Normalized Cross Convolution
    for i in range(rows):
        for j in range(cols):
            c_img = utils.crop(img,i,i+len(template),j,j+len(template[0]))
            norm_list.append(norm_xcorr2d(c_img,template))

	# Finding maximum of all the NCC values
    maxx = max(norm_list)

	# Getting the index of the maximum NCC value
    max_pos = norm_list.index(maxx)

	# Calculating value of x, y
	#X_index/total_cols gives us the x-coordinate of the matched template
	#Y_index/total_cols gives us the y-coordinate of the matched template

    return int(max_pos/cols),int(max_pos%cols),maxx


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)

    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
