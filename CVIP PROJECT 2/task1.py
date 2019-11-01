"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def get_slope_of_line(two_points):

    point1 = two_points[0]['value']
    point2 = two_points[1]['value']

    # point 1 coordinates
    x0 = point1[0]
    y0 = point1[1]

    # point 2 coordinates
    x1 = point2[0]
    y1 = point2[1]

    m_slope = (y1-y0)/(x1-x0 + 10**(-10))

    y_intercept = -m_slope*x0 + y0

    return m_slope, y_intercept

def get_distance(one_point,m_slope,y_intercept):

    x, y = one_point['value']

    distance = abs(m_slope*x - y + y_intercept)/(1+m_slope**2)**0.5

    return distance

def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.

    answer = []
    # list to track chosen points
    chosen = []

    # set minimum error to infinity (some large value)
    minimum_error = 10 ** 15

    # iterator to track number of iterations
    iterator = 0

    # keep track of every permutation possible
    permutations = ((len(input_points) * (len(input_points) - 1)) / 2) * 2

    # start loop
    while len(chosen) < permutations and iterator < k:
        inl = []
        otl = []
        sum_of_distances = 0
        inlier_count = 0
        random.shuffle(input_points)

        # select first two points and keep unselected points in a separate list
        select = input_points[:2]
        unselect = input_points[2:]

        # get names of points selected
        get_points = (select[0]['name'], select[1]['name'])
        get_points_reversed = (select[1]['name'], select[0]['name'])

        if get_points in chosen:
            continue
        else:
            iterator += 1
            chosen.append(get_points)
            chosen.append(get_points_reversed)
            m_slope, y_intercept = get_slope_of_line(select)

        print("Points: {}: {} and {}: {}".format(select[0]['name'], select[0]['value'], select[1]['name'], select[1]['value']))
        print("Slope: {}, Y-intercept: {}".format(m_slope, y_intercept))
        print("-----------------------------------------------------------")
        inl.append(select[0]['name'])
        inl.append(select[1]['name'])

        for other_point in unselect:
            point_distance = get_distance(other_point, m_slope, y_intercept)
            print("Point {}, Distance = {}".format(other_point, point_distance))
            if point_distance <= t:
                sum_of_distances += point_distance
                inlier_count += 1
                inl.append(other_point['name'])
            else:
                otl.append(other_point['name'])

        if inlier_count >= d:
            average_error = sum_of_distances / inlier_count
            print("Error= {}".format(average_error))
            print("-----------------------------------------------------------")

            if average_error < minimum_error:
                minimum_error = average_error
                answer = [inl, otl]

    print("ANSWER:\nInliers: {}, Outliers: {}".format(answer[0], answer[1]))
    return answer
    # raise NotImplementedError

if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8  
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()


