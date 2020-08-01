"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""


import utils
import numpy as np
import json

def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    # TODO: implement this function.

    img2 = np.copy(img)

    # crop() referenced from previous project
    def crop(img, rmin, rmax, cmin, cmax):
        """Crops a given image."""
        if len(img) < rmax:
            print('WARNING')
        patch = img[rmin: rmax]
        patch = [row[cmin: cmax] for row in patch]
        return patch

    # for kernel 3x3, padding is 3/2 = 1
    padded_image = utils.zero_pad(img, 1, 1)
    # print(padded_image)
    # print(img.shape[0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cropped_img = crop(padded_image, i, i + 3, j, j + 3)
            median_val = np.median(cropped_img)
            img2[i][j] = median_val
    return img2

def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """    
    # TODO: implement this function.

    mse_val = np.sum(np.square(np.subtract(img1, img2))).mean()
    # print(mse_val)
    return mse_val
    

if __name__ == "__main__":
    img = utils.read_image('lenna-noise.png')
    gt = utils.read_image('lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result,'results/task2_result.jpg')


