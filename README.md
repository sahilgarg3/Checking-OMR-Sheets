# Checking-OMR-Sheets

Marking of the OMR Sheets according to the number of correct options selected by individual using OpenCV
## [OMR Marking](OMR_Marking.py)

## Requirements
- OpenCV
- NumPy
- [Function.py](function.py) which is the python file consisting multiple functions to perform specific task which could be required in many such project again and again.
---
## User Inputs
1. Number of questions
2. Number of options/choices per question
3. Actual or Correct Answers
4. Maximum area of the circles to be considered for option
5. Width and Height of the image for display purpose
---
Inputs of **get_countour** function are as follows:
- Image
- Canny Image
- Filters (Minimum number of boundaries)
- Draw (Whether to draw the contours or not)
- MinArea (Minimum area of the contour to be considered)

Outputs of **get_countour** function are as follows:
- List
  - Contours
  - Area of the contours
  - Perimeter of the countours 
  - Number of boundaries
  - Center of the contour
  - Bounding Box 
- Image with Contours drawn or not depending upon the parameter
Note: Contours are in decreasing order w.r.t the area covered by the contour.
---
Inputs of **reorder** function is points.

This function is to reorder the given points in proper order to get points for Warp Perspective.

---
Inputs of **get_warp** function are as follows:
- Image
- Width and Height
- Points
- Final Width and Height
This function gives the warped image of the input image within given data points
---
Inputs of **get_contour_circle** function are as follows:
- Original Image
- Canny Image
- Draw
- minArea
Outputs of  **get_contour_circle** function are as follows:
- List
  - Contours
  - Center of the contours
  - Radius
  - Bounding Box
- Countoured Image
---
Inputs of **splitting** function are as follows:
- Image
- Number of questions
- Number of choices/options
Output of **splitting** function is matrix of images of options.
---
Inputs of **show_answers** function are as follows:
- Image
- Correct Answers
- Selected Answers
- Grades
- Number of Questions
- Number of choices
Output of the fuction is the image of the OMR-Sheet with results/answers on it.
---
Function **get_warp_inverse** is almost similar to get_warp function but instead of extracting the warp image, this function insert the warp image on the original image.

---
Inputs of **concat** function are as follows:
- Scale (Scale of the original image)
- List of Images/Videos
This funciton gives the concat images/videos given as input to it in the list, irrespective of the nature, scale, dimensions of the image/video
