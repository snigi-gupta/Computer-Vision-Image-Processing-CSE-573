"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""


import utils
import numpy as np
import json
import time


def kmeans(img,k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # TODO: implement this function.

    # counter to keep track of iterations
    count = 0

    # to store all optimal values for all combination points
    store = []

    # to keep track of known/seen points(centers)
    seen_points = []

    # get unique points
    unique_points = np.unique(np.array([y for x in img for y in x]))
    number_of_unique_points = len(unique_points)

    unique_combinations = [(unique_points[x], unique_points[y]) for x in range(number_of_unique_points) for y in
                           range(x + 1, number_of_unique_points)]
    # print(len(unique_combinations))

    # iterating for all unique possible combinations
    for t in unique_combinations:
        count += 1

        # get the points
        pt1 = t[0]
        pt2 = t[1]
        sum_d = 0
        flag = False

        # iterating k means for max 10 iterations
        for u in range(10):

            # calculate distance of pixel from point(center)
            d1 = abs(pt1 - img)
            d2 = abs(pt2 - img)

            # get image matrix for the cluster it belongs to using np.where
            img1 = np.where(d1 <= d2, img, 0)  # belongs to cluster 0
            img2 = np.where(d1 > d2, img, 0)  # belongs to cluster 1

            # get sum of all values in the matrix
            sum_img1 = np.sum(img1)
            sum_img2 = np.sum(img2)

            # get number of points belonging to the cluster
            num_points_img1 = len(img1[img1 != 0])
            num_points_img2 = len(img1[img2 != 0])

            # to remove divide-by-zero error
            if num_points_img1 == 0:
                num_points_img1 = 1
            if num_points_img2 == 0:
                num_points_img2 = 1

            old_pt1 = pt1
            old_pt2 = pt2

            # add points to seen list for further check
            seen_points.append([old_pt1, old_pt2])
            pt1 = sum_img1 / num_points_img1
            pt2 = sum_img2 / num_points_img2

            if old_pt1 == pt1 and old_pt2 == pt2:
                # breaking since points have converged
                # print("breaking since points have converged")
                flag = True
                break
            if [pt1, pt2] in seen_points:
                # breaking since these centers are already found
                # print("breaking since these centers are already found")
                break

        # get the error/sumdistance
        if flag:
            sum_d = np.sum(np.where(abs(pt1 - img) <= abs(pt2 - img), abs(pt1 - img), abs(pt2 - img)))
            # storing the optimal values
            store.append([pt1, pt2, sum_d])

        # printing count to know till where the code is running
        if count % 1000 == 0:
            print(" Unique points done: {}".format(count))
            # print(store)

    # get the optimal value
    min_val = min(store, key=lambda x: x[2])
    label_val = np.where(abs(min_val[0] - img) <= abs(min_val[1] - img), 0, 1)

    return [int(min_val[0]), int(min_val[1])], label_val, int(min_val[2])


def visualize(centers,labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    # TODO: implement this function.

    labels = np.where(labels == 0, centers[0], centers[1])

    return labels.astype(np.uint8)

     
if __name__ == "__main__":
    img = utils.read_image('lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img,k)
    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers":centers, "distance":sumdistance, "time":running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
