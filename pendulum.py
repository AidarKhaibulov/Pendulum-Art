import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def rot_matr(ang):
    mtr = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    return mtr


def diagonalMatrix(a, b, c):
    mtr = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
    return mtr


def draw_NURBS_curve(img, points, color):
    x0,y0=points[0][0],points[0][1]
    x1,y1=points[1][0],points[1][1]
    x2,y2=points[2][0],points[2][1]
    x3,y3=points[3][0],points[3][1]
    t = np.linspace(0, 1, density)
    x_coordinates = ((x0 * (1 - t) ** 3 + x1 * (1 - t) ** 2 * t + x2 * (1 - t) * t ** 2 + x3 * t ** 3))
    y_coordinates = ((y0 * (1 - t) ** 3 + y1 * (1 - t) ** 2 * t + y2 * (1 - t) * t ** 2 + y3 * t ** 3))
    z_coordinates = ((1 * (1 - t) ** 3 + 1 * (1 - t) ** 2 * t + 1 * (1 - t) * t ** 2 + 1 * t ** 3))
    res_x = []
    res_y = []
    for i in range(len(x_coordinates)):
        res_x.append(int(x_coordinates[i] / z_coordinates[i]))
        res_y.append(int(y_coordinates[i] / z_coordinates[i]))
    for i in range(density):
        if 0 <= res_x[i] < width and 0 <= res_y[i] < height:
            img[res_x[i], res_y[i], :3] = color
    return [res_y, res_x]


width, height = 1000, 1000

density = 500

img = np.zeros((width, height, 3), dtype=np.uint8)

center_matrix = diagonalMatrix(1, 1, 1)
shifting_matrix = diagonalMatrix(1, 1, 1)

center_matrix[2][0] = -0.25 * width
center_matrix[2][1] = -0.5 * width

shifting_matrix[2][0] = 0.25 * width
shifting_matrix[2][1] = 0.5 * width

circle_center_matrix = diagonalMatrix(1, 1, 1)
circle_shifting_matrix = diagonalMatrix(1, 1, 1)

degrees = 158
frames = []
frames1 = []
fig = plt.figure()

for degree in range(0,degrees,20):
    #refreshing scene
    for i in range(height):
        for j in range(width):
            img[i][j] = [19, 99, 134]
    #angle of shifting
    alp = math.radians(-degree)
    print(degree)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:2, :2] = rot_matr(alp)

    thread_length = int(0.2 * width)
    right_thread = np.zeros((2, thread_length), dtype=np.uint16)
    left_thread = np.zeros((2, thread_length), dtype=np.uint16)

    # drawing threads
    for i in range(thread_length):
        right_thread[0][i] = 0.25 * width
        right_thread[1][i] = 0.5 * width + i
        left_thread[0][i] = 0.25 * width + i
        left_thread[1][i] = 0.5 * width

    for i in range(thread_length):
        if (degree >= 0.55*degrees):
            # drawing right thread rotated at 90 degrees
            rot_matrix = np.zeros((3, 3))
            alpha = math.radians(-90)
            rot_matrix[:2, :2] = rot_matr(alpha)
            temp = [right_thread[0][i], right_thread[1][i], 1]
            temp = temp @ center_matrix
            temp = temp @ rot_matrix
            temp[2] = 1
            temp = temp @ shifting_matrix
            right_thread[0][i] = temp[0]
            right_thread[1][i] = temp[1]
        else:
            # shifting right thread
            temp = [right_thread[0][i], right_thread[1][i], 1]
            temp = temp @ center_matrix
            temp = temp @ rotation_matrix
            temp[2] = 1
            temp = temp @ shifting_matrix
            right_thread[0][i] = temp[0]
            right_thread[1][i] = temp[1]
        # shifting left thread
        if (degree >= degrees/2.4):
            alpha = math.radians(-degree + 66)
            rotationSecond = np.zeros((3, 3))
            rotationSecond[:2, :2] = rot_matr(alpha)
            temp = [left_thread[0][i], left_thread[1][i], 1]
            temp = temp @ center_matrix
            temp = temp @ rotationSecond
            temp[2] = 1
            temp = temp @ shifting_matrix
            left_thread[0][i] = temp[0]
            left_thread[1][i] = temp[1]

    r = 100
    RBP1 = [right_thread[0][thread_length - 1], right_thread[1][thread_length - 1], 1]
    RBP2 = [right_thread[0][thread_length - 1] - r, right_thread[1][thread_length - 1], 1]
    RBP3 = [right_thread[0][thread_length - 1] - r, right_thread[1][thread_length - 1] + r, 1]
    RBP4 = [right_thread[0][thread_length - 1], right_thread[1][thread_length - 1] + r, 1]
    RBP5 = [right_thread[0][thread_length - 1], right_thread[1][thread_length - 1], 1]
    RBP6 = [right_thread[0][thread_length - 1] + r, right_thread[1][thread_length - 1], 1]
    RBP7 = [right_thread[0][thread_length - 1] + r, right_thread[1][thread_length - 1] + r, 1]
    RBP8 = [right_thread[0][thread_length - 1], right_thread[1][thread_length - 1] + r, 1]

    LBP1 = [left_thread[0][thread_length - 1], left_thread[1][thread_length - 1], 1]
    LBP2 = [left_thread[0][thread_length - 1], left_thread[1][thread_length - 1] + r, 1]
    LBP3 = [left_thread[0][thread_length - 1] + r, left_thread[1][thread_length - 1] + r, 1]
    LBP4 = [left_thread[0][thread_length - 1] + r, left_thread[1][thread_length - 1], 1]
    LBP5 = [left_thread[0][thread_length - 1], left_thread[1][thread_length - 1], 1]
    LBP6 = [left_thread[0][thread_length - 1], left_thread[1][thread_length - 1] - r, 1]
    LBP7 = [left_thread[0][thread_length - 1] + r, left_thread[1][thread_length - 1] - r, 1]
    LBP8 = [left_thread[0][thread_length - 1] + r, left_thread[1][thread_length - 1], 1]

    right_circle_points = np.array([RBP1, RBP2, RBP3, RBP4, RBP5, RBP6, RBP7, RBP8])
    left_circle_points = np.array([LBP1, LBP2, LBP3, LBP4, LBP5, LBP6, LBP7, LBP8])

    circle_center_matrix[2][0] = -(right_thread[0][thread_length - 1] - 1)
    circle_center_matrix[2][1] = -(right_thread[1][thread_length - 1] - 1)

    circle_shifting_matrix[2][0] = (right_thread[0][thread_length - 1] - 1)
    circle_shifting_matrix[2][1] = (right_thread[1][thread_length - 1] - 1)

    # shifting right ball logic
    if (degree < 0.55*degrees):
        right_circle_points = right_circle_points @ circle_center_matrix
        right_circle_points = right_circle_points @ rotation_matrix
        for i in range(8):
            right_circle_points[i][2] = 1
        right_circle_points = right_circle_points @ circle_shifting_matrix
    else:
        # drawing right ball rotated at 90 degrees
        rot_matrix = np.zeros((3, 3))
        alpha = math.radians(-90)
        rot_matrix[:2, :2] = rot_matr(alpha)
        right_circle_points = right_circle_points @ circle_center_matrix
        right_circle_points = right_circle_points @ rot_matrix
        for i in range(8):
            right_circle_points[i][2] = 1
        right_circle_points = right_circle_points @ circle_shifting_matrix

    circle_center_matrix[2][0] = -(left_thread[0][thread_length - 1] - 1)
    circle_center_matrix[2][1] = -(left_thread[1][thread_length - 1] - 1)

    circle_shifting_matrix[2][0] = (left_thread[0][thread_length - 1] - 1)
    circle_shifting_matrix[2][1] = (left_thread[1][thread_length - 1] - 1)

    # shifting left ball
    if (degree >= degrees/2.4):
        koef = 3.4
        if degree >= 0.63*degrees:
            if degree > 0.81*degrees:
                k = (degree - 100)
                left_circle_points[0][0] += koef * k
                left_circle_points[0][1] += r - koef * (k - 28)
                left_circle_points[1][0] += koef * k
                left_circle_points[1][1] -= r - koef * (k - 28)
                left_circle_points[4][0] += koef * k
                left_circle_points[4][1] -= r - koef * (k - 28)
                left_circle_points[5][0] += koef * k
                left_circle_points[5][1] += r - koef * (k - 28)
            else:
                k =3.4* (degree - 100)
                left_circle_points[0][0] +=k
                left_circle_points[0][1] += k
                left_circle_points[1][0] +=k
                left_circle_points[1][1] -= k
                left_circle_points[4][0] +=k
                left_circle_points[4][1] -= k
                left_circle_points[5][0] +=k
                left_circle_points[5][1] +=k

        alpha = math.radians(-degree + degrees/2.4)
        rotationSecondCircle = np.zeros((3, 3))
        rotationSecondCircle[:2, :2] = rot_matr(alpha)
        left_circle_points = left_circle_points @ circle_center_matrix
        left_circle_points = left_circle_points @ rotationSecondCircle
        for i in range(8):
            left_circle_points[i][2] = 1
        left_circle_points = left_circle_points @ circle_shifting_matrix

    right_circle_color = [0, 0, 0]
    left_circle_color = [255, 255, 255]

    #swaping balls colors
    if degree >= degrees/2.4:
        right_circle_color = [255, 255, 255]
        left_circle_color = [0, 0, 0]

    draw_NURBS_curve(img, right_circle_points[0:4], right_circle_color)
    draw_NURBS_curve(img, right_circle_points[4:8], right_circle_color)

    draw_NURBS_curve(img, left_circle_points[0:4], left_circle_color)
    draw_NURBS_curve(img, left_circle_points[4:8], left_circle_color)

    # swap threads colors
    for i in range(0, thread_length):
        img[int(right_thread[0][i])][int(right_thread[1][i])] = right_circle_color
        img[int(left_thread[0][i])][int(left_thread[1][i])] = left_circle_color

    im = plt.imshow(img)
    plt.show()
    frames.append([im])

    img1 = np.zeros((width, height, 3), dtype=np.uint8)
    for p in range(height):
        for q in range(width//2):
            img1[p,q],img1[p,width-1-q]=img[p,width-1-q],img[p,q]
    im1 = plt.imshow(img1)
    #plt.show()
    frames1.append([im1])


frames+=frames1
ani = animation.ArtistAnimation(fig, frames, interval=4, blit=True, repeat_delay=100)
writer = PillowWriter(fps=24)
ani.save("circle.gif", writer=writer)
plt.show()
