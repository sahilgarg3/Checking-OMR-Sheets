import cv2 as cv
import numpy as np
import functions


##########################################################################################
#                        USER INITIALIZATIONS
##########################################################################################
question, options = 5, 5            # QUESTIONS AND CHOICES
myAnswers = [1, 2, 0, 4, 4]         # Actual Answers
choiceArea = 1000                   # Area of the choosable options
img_width, img_height = 550, 600
global contours
##########################################################################################
#########################################################################################
#   PREPROCESSING THE IMAGE
#########################################################################################
image = cv.imread('OMR 5.png')
image = cv.resize(image, (img_width, img_height))
img_cont_2 = image.copy()
Canny, Erode = functions.preprocessing(image)
#############################################
try:
    ##########################################################################################
    #                                   CONTOURS
    ##########################################################################################
    contours, img_cont = functions.get_contour(image, Canny, filters=4, draw=True, minArea=9000)
    #  contours contains (contours, area, peri, approx, len(approx), center, bbox)
    #                 With respect to the given parameters
    ##########################################################################################
    #            Verification of the contours by drawing on img_cont_2
    #               Checking whether marking area is given or not
    ##########################################################################################
    cv.drawContours(img_cont_2, contours[0][0], -1, (0, 0, 255), 5)
    if len(contours) == 2:
        cv.drawContours(img_cont_2, contours[1][0], -1, (255, 0, 0), 5)
    ##########################################################################################
    ##########################################################################################
    #                           Targeted Contour
    #                 Arranging in order of points for Warp Pers.
    ##########################################################################################
    BiggestContour = contours[0][3]
    BiggestContour = functions.reorder(BiggestContour)
    if len(contours) == 2:
        globalContour = contours[1][3]
        globalContour = functions.reorder(globalContour)
    # print('Biggest Contour',BiggestContour)
    # print('Global Contour',globalContour)
    ##########################################################################################
    ##########################################################################################
    #                               Answering Area
    ##########################################################################################
    pt1 = np.float32(BiggestContour)
    warpImg1 = functions.get_warp(image, pt1, img_width, img_height, img_width, img_height)
    ##########################################################################################
    ##########################################################################################
    warpCanny, warpErode = functions.preprocessing(warpImg1)
    ##########################################################################################
    contours_circle, img_cont_circle = functions.get_contour_circle(warpImg1, warpCanny, draw=True, minArea=choiceArea)
except:
    contours_circle, img_cont_circle = functions.get_contour_circle(image, Canny, draw=True, minArea=choiceArea)
    # contours_circle contains (contours,center, radius, bbox)
finally:
    x, y = [], []
    points_c = []
    for obj in contours_circle:
        x.append(obj[3][0])
        y.append(obj[3][1])
        final_point = obj[3][0] + obj[3][2], obj[3][1] + obj[3][3]
        points_c.append(final_point)
    first = np.min(x), np.min(y)  # Initial points of the first option
    points_c = np.array(points_c)
    sum1 = points_c.sum(1)
    last = points_c[np.argmax(sum1)]  # Initial points of the last option
    # cv.circle(img_cont_circle, first, 8, (0, 0, 0), -1)       # Verifying the location
    # cv.circle(img_cont_circle, (last[0], last[1]), 8, (0, 0, 0), -1)       # Verifying the location
    img_cont_circle = img_cont_circle[first[1] - 10:last[1] + 10, first[0] - 10:last[0] + 10]
    if len(contours) > 0:
        warpImg = warpImg1.copy()
    else:
        warpImg = image.copy()
    warpImg = warpImg[first[1] - 10:last[1] + 10, first[0] - 10:last[0] + 10]
    warpImg = cv.resize(warpImg, (img_width, img_height))
    img_cont_circle = cv.resize(img_cont_circle, (img_width, img_height))
    ##########################################################################################
    #                               Threshold Image
    ##########################################################################################
    warpGray = cv.cvtColor(img_cont_circle, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(warpGray, 150, 255, cv.THRESH_BINARY_INV)[1]
    ##########################################################################################
    ##########################################################################################
    #                            Splitting each Option
    ##########################################################################################
    boxes = functions.splitting(threshold, question, options)
    ##########################################################################################
    ##########################################################################################
    #                           Count non-zeroes pixels in the box
    #                            Finding NON ZEROS PIXEL VALUES
    ##########################################################################################
    # # print(cv.countNonZero(boxes[0]), cv.countNonZero(boxes[1]))
    myPixelValues = np.zeros((question, options), np.int32)
    countC, countR = 0, 0
    for x in boxes:
        totalPixels = cv.countNonZero(x)
        myPixelValues[countR][countC] = totalPixels
        countC += 1
        if countC == options:
            countR += 1
            countC = 0
    # print(myPixelValues)              # NON Zeros values in the matrix
    ##########################################################################################
    ##########################################################################################
    #                   FINDING INDEX VALUES OF THE MARKINGS
    ##########################################################################################
    myIndex = []
    for x in range(0, question):
        arr = myPixelValues[x]
        index = np.argmax(arr)
        myIndex.append(index)
    # print(myIndex)
    ##########################################################################################
    ##########################################################################################
    #                     GETTING THE RESULT OF THE TEST
    ##########################################################################################
    grading = []
    for x in range(0, question):
        if myAnswers[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    # print(grading)
    score = (sum(grading) / question) * 100
    # print(f"{score} %")              # Grades in Percentage
    ##########################################################################################
    ##########################################################################################
    #                       DISPLAYING ANSWERS
    ##########################################################################################
    imgResult = warpImg.copy()
    imgResult = functions.show_answers(imgResult, myIndex, myAnswers, grading, question, options)
    ##########################################################################################
    ##########################################################################################
    #                       RESULT ON THE BLANK IMAGE
    ##########################################################################################
    imgRawDraw = np.zeros_like(warpImg)
    imgRawDraw = functions.show_answers(imgRawDraw, myIndex, myAnswers, grading, question, options)
    ##########################################################################################
    ##########################################################################################
    #                           INVERSE WARPING
    ##########################################################################################
    try:
        points2 = [[first[0] - 10, first[1] - 10], [last[0] + 10, first[1] - 10], [first[0] - 10, last[1] + 10],
                   [last[0] + 10, last[1] + 10]]
        imgInverse = functions.get_warp_inverse(imgRawDraw, img_width, img_height, points2, img_width,
                                                    img_height)
        point1 = np.float32(BiggestContour)
        imgInverseWarp = functions.get_warp_inverse(imgInverse, img_width, img_height, point1, img_width, img_height)

    except:
        points2 = [[first[0] - 10, first[1] - 10], [last[0] + 10, first[1] - 10], [first[0] - 10, last[1] + 10],
                   [last[0] + 10, last[1] + 10]]
        imgInverseWarp = functions.get_warp_inverse(imgRawDraw, img_width, img_height, points2, img_width,
                                                    img_height)
    ##########################################################################################
    #                   Displaying Answers on Original Image
    ##########################################################################################
    imgFinal = image.copy()
    imgFinal = cv.addWeighted(imgFinal, 1, imgInverseWarp, 1, 0)
    cv.putText(imgFinal, str(score) + '%', (20, 50), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
    ##########################################################################################
    ##########################################################################################
    #                           Display of Images
    ##########################################################################################
    ImgConcat = functions.concat(0.3, [[image, Canny, warpImg, warpGray]
                                       , [img_cont_circle, threshold, imgResult, imgRawDraw],
                                       [imgInverseWarp, imgFinal]]
                                 )
    cv.imshow('Result', ImgConcat)
    cv.imshow('Image Final', imgFinal)
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    cv.waitKey(0)
