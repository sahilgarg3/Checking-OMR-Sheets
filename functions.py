import cv2 as cv
import numpy as np


def concat(scale, ImageList):
    rows = len(ImageList)
    if type(ImageList[0]) == list:

        lengths = [len(i) for i in ImageList]
        cols = max(lengths)
        blank = np.zeros((ImageList[0][0].shape[0], ImageList[0][0].shape[1]), np.uint8)
        for i in range(rows):
            while len(ImageList[i]) != cols:
                ImageList[i].append(blank)

        for x in range(0, rows):
            for y in range(0, cols):
                if len(ImageList[x][y].shape) == 2:
                    ImageList[x][y] = cv.cvtColor(ImageList[x][y], cv.COLOR_GRAY2BGR)
                if ImageList[0][0].shape[:2] == ImageList[x][y].shape[:2]:
                    ImageList[x][y] = cv.resize(ImageList[x][y], (0, 0), None, scale, scale)
                else:
                    ImageList[x][y] = cv.resize(ImageList[x][y], (ImageList[0][0].shape[1], ImageList[0][0].shape[0]),
                                                None, scale, scale)

        blankImage = np.zeros((ImageList[0][0].shape[0], ImageList[0][0].shape[1]), np.uint8)
        hor = [blankImage] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(ImageList[x])
        ver = np.vstack(hor)

    else:
        for x in range(0, rows):
            if ImageList[0].shape[:2] == ImageList[x].shape[:2]:
                ImageList[x] = cv.resize(ImageList[x], (0, 0), None, scale, scale)
            else:
                ImageList[x] = cv.resize(ImageList[x], (ImageList[0].shape[1], ImageList[0].shape[0]),
                                         None, scale, scale)

            if len(ImageList[x].shape) == 2:
                ImageList[x] = cv.cvtColor(ImageList[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(ImageList)
        ver = hor

    return ver


def preprocessing(img, threshold1=40, threshold2=90):
    imgBlur = cv.GaussianBlur(img, (3, 3), 0)
    imgCanny = cv.Canny(imgBlur, threshold1, threshold2)
    kernel = np.ones((5, 5), np.int32)
    imgDia = cv.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv.erode(imgDia, kernel, iterations=1)
    return imgCanny, imgErode


# contours, area, peri, approx, len(approx), center, bbox
def get_contour(image1, image2, minArea=2000, filters=0, draw=False):
    """ image1 = Original Image /////
    image2 = Canny or Erode Image"""
    contour, hierarchy = cv.findContours(image2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    imgContour = image1.copy()
    final_cont = []
    for cnt in contour:
        area = cv.contourArea(cnt)
        if area > minArea:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            bbox = cv.boundingRect(approx)
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
            if filters > 0:
                if len(approx) == filters:
                    final_cont.append([cnt, area, peri, approx, len(approx), center, bbox])
            else:
                final_cont.append([cnt, area, peri, approx, len(approx), center])
    final_cont = sorted(final_cont, key=lambda x: x[1], reverse=True)
    for i in range(0, len(final_cont)):
        peri = cv.arcLength(final_cont[i][0], True)
        approx = cv.approxPolyDP(final_cont[i][0], 0.02 * peri, True)
        final_cont[i][3] = approx
    if draw:
        for i in range(0, len(final_cont)):
            cv.drawContours(imgContour, final_cont[i][0], -1, (0, 255, 0), 5)
        return final_cont, imgContour
    return final_cont, imgContour


# contours,center, radius, bbox
def get_contour_circle(image1, image2, minArea=5000, draw=False):
    """ image1 = Original Image /////
        image2 = Canny or Erode Image"""
    contour, hierarchy = cv.findContours(image2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    imgContour = image1.copy()
    final_cont = []
    for cnt in contour:
        area = cv.contourArea(cnt)
        if area > minArea:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            bbox = cv.boundingRect(approx)
            final_cont.append([cnt, center, radius, bbox])
    if draw:
        for i in range(0, len(final_cont)):
            cv.drawContours(imgContour, final_cont[i][0], -1, (0, 255, 0), 5)
        return final_cont, imgContour
    return final_cont, imgContour


def get_warp(image, point, width, height, final_width, final_height):
    pts1 = np.float32(point)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_warp = cv.warpPerspective(image, matrix, (final_width, final_height))
    return img_warp


def get_warp_inverse(image, width, height, point, final_width, final_height):
    pts2 = np.float32(point)
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img_warp_inverse = cv.warpPerspective(image, matrix, (final_width, final_height))
    return img_warp_inverse


def reorder(points):
    new_points = np.zeros_like(points)
    points = points.reshape((4, 2))

    sum1 = points.sum(1)
    new_points[0] = points[np.argmin(sum1)]
    new_points[3] = points[np.argmax(sum1)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def splitting(img, questions, choices):

    rows = np.array_split(img, questions)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, choices)
        for box in cols:
            boxes.append(box)
    return boxes


def show_answers(img, myIndex, myAnswers, grading, questions, choices):
    secW = img.shape[1] / questions
    secH = img.shape[0] / choices

    for x in range(0, questions):
        Ans = myIndex[x]
        cx = int((Ans*secW) + secW//2)
        cy = int((x * secH) + secH//2)
        if grading[x] == 1:
            mycolor = (0, 255, 0)
        else:
            mycolor = (0, 0, 255)
            correct_ans = myAnswers[x]
            cxs = int((correct_ans * secW) + secW // 2)
            cv.circle(img, (cxs, cy), 25, (0, 255, 0), -1)
        cv.circle(img, (cx, cy), 40, mycolor, -1)
    return img