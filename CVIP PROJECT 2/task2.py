"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    # Change image to grayscale
    left_img_sample = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img_sample = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    sift_obj = cv2.xfeatures2d.SIFT_create()

    # find the key points and descriptors using SIFT
    keypoint_left, descriptor_left = sift_obj.detectAndCompute(left_img_sample, None)
    keypoint_right, descriptor_right = sift_obj.detectAndCompute(right_img_sample, None)

    # cv2.imwrite('results/keypoints.jpg', cv2.drawKeypoints(left_img, keypoint_left, None))

    # convert the keypoints to numpy arrays
    keypoint_left = np.float32([kp.pt for kp in keypoint_left])
    keypoint_right = np.float32([kp.pt for kp in keypoint_right])

    """
    1. match features between the two images
    2. compute the raw matches and initialize the list of actual matches
    """
    the_matcher = cv2.DescriptorMatcher_create("BruteForce")
    the_raw_matches = the_matcher.knnMatch(descriptor_right, descriptor_left, 2)
    the_matches = []

    """
    1. loop over the raw matches
    2. ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test), ratio = 0.75
    """
    for m in the_raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            the_matches.append((m[0].trainIdx, m[0].queryIdx))

    """
    1. computing a homography requires at least 4 matches
    2. construct the two sets of points
    3. compute the homography between the two sets of points, threshold = 4.0
    4. get the matches along with the homography matrix and status of each matched point
    """
    if len(the_matches) > 4:
        points_right = np.float32([keypoint_right[i] for (_, i) in the_matches])
        points_left = np.float32([keypoint_left[i] for (i, _) in the_matches])

        (H, status) = cv2.findHomography(points_right, points_left, cv2.RANSAC, 4.0)
        M = (the_matches, H, status)
    else:
        print("No homography could be computed.\n")
        return None

    """
    if the match is None, then there aren't enough matched keypoints to create a panorama, return None
    """
    if M is None:
        print("Not enough matched keypoints.\n")
        return None

    """
    1. apply a perspective warp to stitch images
    """
    (the_matches, H, status) = M
    result = cv2.warpPerspective(right_img, H, (right_img.shape[1] + left_img.shape[1], right_img.shape[0]))
    # cv2.imwrite("results/wrap_result.jpg", result)

    result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    # cv2.imwrite("results/result.jpg", result)

    # below code only for visualization.
    """
    1. draw matches
    """
    (h_left, w_left) = left_img.shape[:2]
    (h_right, w_right) = right_img.shape[:2]
    matchedimage = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype="uint8")
    matchedimage[0:h_right, 0:w_right] = right_img
    matchedimage[0:h_left, w_right:] = left_img

    """
    1. loop over the matches
    2. only process the match who's keypoint was a successful match.
    """
    for ((trainIdx, queryIdx), s) in zip(the_matches, status):
        if s == 1:
            # draw the match
            points_right = (int(keypoint_right[queryIdx][0]), int(keypoint_right[queryIdx][1]))
            points_left = (int(keypoint_left[trainIdx][0]) + w_right, int(keypoint_left[trainIdx][1]))
            cv2.line(matchedimage, points_left, points_right, (0, 255, 0), 1)
    # cv2.imwrite("results/matched.jpg", matchedimage)

    # raise NotImplementedError
    return result

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)


