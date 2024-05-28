######################################
## FOLLOWING TUTORIAL FROM
## https://stackoverflow.com/questions/66757199/color-percentage-in-image-for-python-using-opencv
######################################

# Imports
import cv2
import numpy as np

# Read image
imagePath = "/media/nharrison/Extreme SSD/clear_water_log/clear_img/"
img = cv2.imread(imagePath+"2024-05-28T16-35-59.png")

# Here, you define your target color as
# a tuple of three values: RGB
green = [130, 158, 0]

# You define an interval that covers the values
# in the tuple and are below and above them by 20
diff = 20

# Be aware that opencv loads image in BGR format,
# that's why the color values have been adjusted here:
boundaries = [([green[2], green[1]-diff, green[0]-diff],
           [green[2]+diff, green[1]+diff, green[0]+diff])]

# # Scale your BIG image into a small one:
# scalePercent = 0.3

# # Calculate the new dimensions
# width = int(img.shape[1] * scalePercent)
# height = int(img.shape[0] * scalePercent)
# newSize = (width, height)

# # Resize the image:
# img = cv2.resize(img, newSize, None, None, None, cv2.INTER_AREA)

# # check out the image resized:
# cv2.imshow("img resized", img)
# cv2.waitKey(0)

## using hsv color space (hue, saturation, and value) is more accurate but more time intensive
# The HSV mask values, defined for the green color:
lowerValues = np.array([29, 89, 70])
upperValues = np.array([179, 255, 255])

# Convert the image to HSV:
hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create the HSV mask
hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)

# AND mask & input image:
hsvOutput = cv2.bitwise_and(img, img, mask=hsvMask)

# You can use the mask to count the number of white pixels.
# Remember that the white pixels in the mask are those that
# fall in your defined range, that is, every white pixel corresponds
# to a green pixel. Divide by the image size and you got the
# percentage of green pixels in the original image:
ratio_green = cv2.countNonZero(hsvMask)/(img.size/3)
colorPercent = (ratio_green * 100)# / scalePercent

cv2.putText(img,
            str(np.round(colorPercent, 2)) + '% Green',
            (100,500), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            255)

# numpy's hstack is used to stack two images horizontally,
# so you see the various images generated in one figure:
# cv2.imshow("images", np.hstack([img, hsvOutput]))
cv2.imshow("images", img)
cv2.waitKey(0)
