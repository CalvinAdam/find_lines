import cv2
import numpy as np


def find_lines(img, img_name):
    (_, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Making a horizontal and vertical kernel
    kernel_length = np.array(img).shape[1] // 120

    # Vertical kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    # Horizontal kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # This filters out everything that isn't a straight line
    vertical_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
    horizontal_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)

    vertical_lines = cv2.HoughLinesP(vertical_lines_img, rho, theta, threshold, np.array([]),
                                     min_line_length, max_line_gap)
    horizontal_lines = cv2.HoughLinesP(horizontal_lines_img, rho, theta, threshold, np.array([]),
                                       min_line_length, max_line_gap)
    if vertical_lines is not None and horizontal_lines is not None:
        lines = np.concatenate((vertical_lines, horizontal_lines))
    elif vertical_lines is not None:
        lines = vertical_lines
    elif horizontal_lines is not None:
        lines = horizontal_lines
    else:
        return

    for valid_line in lines:  # Adds the lines to a separate image
        for x1, y1, x2, y2 in valid_line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 128, 0), 4)

    # Combines the original image with the line image and writes it to a file
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imwrite(f"documents_lines/{img_name}-with-lines.jpg", lines_edges)


if __name__ == '__main__':
    find_lines(cv2.imread('documents/00000035.TIF', cv2.IMREAD_GRAYSCALE), "00000035.TIF")
