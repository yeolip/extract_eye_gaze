import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
import math
import dlib
import cv2
import operator
import time
from PupilDetector2 import GradientIntersect
# import low_pass_filter as set_param_low_pass_filter, low_pass_filter, test_use_previous_after_to_one_result
# from low_pass_filter import set_param_low_pass_filter, low_pass_filter, test_use_previous_after_to_one_result

from low_pass_filter import *

#3d face model point index
C_R_HEAR = 0
C_L_HEAR = 1
C_NOSE   = 2
C_R_MOUTH= 3
C_L_MOUTH= 4
C_R_EYE  = 5
C_L_EYE  = 6

#2d face mappint point sequence - 68 point
RIGHT_EYE = list(range(36, 42))  # 6
LEFT_EYE = list(range(42, 48))  # 6
NOSE = list(range(27, 36))  # 9
MOUTH_OUTLINE = list(range(48, 60))
MOUTH_INNER = list(range(60, 68))  # 나는 60번이 INNER로 보임

#2d face mappint point sequence - 21 point
RIGHT_EYE_MINI = list(range(3, 9))  # 6
LEFT_EYE_MINI = list(range(9, 15))  # 6
MOUTH_OUTLINE_MINI = list(range(15, 19))
MOUTH_INNER_MINI = list(range(19, 21))  # 나는 60번이 INNER로 보임


#eye open and repeat threshold
# EYE_CLOSE_THRESH = 0.26
EYE_CLOSE_THRESH = 1.5
EYE_CLOSE_REPEAT = 15
EYE_DROWSINESS_THRESH = 0.20
EYE_DROWSINESS_REPEAT = 5 * 2


#status
RET_NOT_DETECT = 0
RET_DETECT = 1
RET_OPEN   = 2
RET_CLOSE  = 3

degreeToRadian = math.pi/180
radianToDegree = 180/math.pi

kGradientThreshold = 5.0
kWeightBlurSize = 3;
maxEyeSize = 8;

# SOLVER FOR PNP
cameraMatrix = np.eye(3)  # A checker en fct de l'optique choisie
distCoeffs = np.zeros((5, 1))
eyeConst = 1.5

# IMAGE POI FOR 7 POINT
FacePOI = np.zeros((7, 2), dtype=np.float32)
ThreeDFacePOI = np.zeros((7, 3), dtype=np.float32)
# RIGHTHEAR
ThreeDFacePOI[C_R_HEAR, 0] = -6
ThreeDFacePOI[C_R_HEAR, 1] = 0
ThreeDFacePOI[C_R_HEAR, 2] = -8
# LEFTHEAR
ThreeDFacePOI[C_L_HEAR, 0] = 6
ThreeDFacePOI[C_L_HEAR, 1] = 0
ThreeDFacePOI[C_L_HEAR, 2] = -8
# NOSE
ThreeDFacePOI[C_NOSE, 0] = 0
ThreeDFacePOI[C_NOSE, 1] = -4
ThreeDFacePOI[C_NOSE, 2] = 2.5
# RIGHTMOUTH
ThreeDFacePOI[C_R_MOUTH, 0] = -5
ThreeDFacePOI[C_R_MOUTH, 1] = -8
ThreeDFacePOI[C_R_MOUTH, 2] = 0
# LEFTMOUTH
ThreeDFacePOI[C_L_MOUTH, 0] = 5
ThreeDFacePOI[C_L_MOUTH, 1] = -8
ThreeDFacePOI[C_L_MOUTH, 2] = 0
# RIGHTEYE
ThreeDFacePOI[C_R_EYE, 0] = -3
ThreeDFacePOI[C_R_EYE, 1] = 0
ThreeDFacePOI[C_R_EYE, 2] = -1
# LEFTEYE
ThreeDFacePOI[C_L_EYE, 0] = 3
ThreeDFacePOI[C_L_EYE, 1] = 0
ThreeDFacePOI[C_L_EYE, 2] = -1

ThreeDFacePOI2 = np.zeros((7, 3), dtype=np.float32)
# RIGHTHEAR
ThreeDFacePOI2[C_R_HEAR, 0] = -6
ThreeDFacePOI2[C_R_HEAR, 1] = 0
ThreeDFacePOI2[C_R_HEAR, 2] = -8
# LEFTHEAR
ThreeDFacePOI2[C_L_HEAR, 0] = 6
ThreeDFacePOI2[C_L_HEAR, 1] = 0
ThreeDFacePOI2[C_L_HEAR, 2] = -8
# NOSE
ThreeDFacePOI2[C_NOSE, 0] = 0
ThreeDFacePOI2[C_NOSE, 1] = 4
ThreeDFacePOI2[C_NOSE, 2] = 2.5
# RIGHTMOUTH
ThreeDFacePOI2[C_R_MOUTH, 0] = -5
ThreeDFacePOI2[C_R_MOUTH, 1] = 8
ThreeDFacePOI2[C_R_MOUTH, 2] = 0
# LEFTMOUTH
ThreeDFacePOI2[C_L_MOUTH, 0] = 5
ThreeDFacePOI2[C_L_MOUTH, 1] = 8
ThreeDFacePOI2[C_L_MOUTH, 2] = 0
# RIGHTEYE
ThreeDFacePOI2[C_R_EYE, 0] = -3.13
ThreeDFacePOI2[C_R_EYE, 1] = 0
ThreeDFacePOI2[C_R_EYE, 2] = -1
# LEFTEYE
ThreeDFacePOI2[C_L_EYE, 0] = 3.13
ThreeDFacePOI2[C_L_EYE, 1] = 0
ThreeDFacePOI2[C_L_EYE, 2] = -1

PERPROMANCE_TEST = 1

deg2Rad = math.pi/180
rad2Deg = 180/math.pi

# declare external func
gi = GradientIntersect()


def timelap_check(title, start):
    if(PERPROMANCE_TEST == 1):
        print('\tTimeLap - {:s} {:.6f}'.format(title, time.time() - start))

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([ x, y, z])

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def calc_dist(p1, p2):
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return distance

def shape_to_np(shape, dtype="int", offset=(0,0)):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x + offset[0], shape.part(i).y + offset[1])
    # return the list of (x, y)-coordinates
    return coords

def analyseFace(img, detector, predictor, quality=0, offset=(0, 0)):
    dets = detector(img, quality)
    result = []
    result_other = []
    train_type = 0
    for k, d in enumerate(dets):
        instantFacePOI = np.zeros((7, 2), dtype=np.float32)
        eyeCorners = np.zeros((2, 2, 2), dtype=np.float32)

        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        train_type = shape.num_parts
        print("train_type",train_type)

        if(shape.num_parts == 21):  #custom training
            instantFacePOI[C_R_HEAR][0] = shape.part(0).x + offset[0]
            instantFacePOI[C_R_HEAR][1] = shape.part(0).y + offset[1]
            instantFacePOI[C_L_HEAR][0] = shape.part(1).x + offset[0]
            instantFacePOI[C_L_HEAR][1] = shape.part(1).y + offset[1]
            instantFacePOI[C_NOSE][0] = shape.part(2).x + offset[0]
            instantFacePOI[C_NOSE][1] = shape.part(2).y + offset[1]
            instantFacePOI[C_R_MOUTH][0] = shape.part(15).x + offset[0]
            instantFacePOI[C_R_MOUTH][1] = shape.part(15).y + offset[1]
            instantFacePOI[C_L_MOUTH][0] = shape.part(17).x + offset[0]
            instantFacePOI[C_L_MOUTH][1] = shape.part(17).y + offset[1]

            leftEyeX = 0
            leftEyeY = 0
            for i in range(3, 9):
                if (i == 3 or i == 6):
                    continue
                leftEyeX += shape.part(i).x
                leftEyeY += shape.part(i).y
            leftEyeX = int(leftEyeX / 4.0)
            leftEyeY = int(leftEyeY / 4.0)
            eyeCorners[0][0] = [shape.part(3).x + offset[0], shape.part(3).y + offset[1]]
            eyeCorners[0][1] = [shape.part(6).x + offset[0], shape.part(6).y + offset[1]]
            instantFacePOI[C_R_EYE][0] = leftEyeX + offset[0]
            instantFacePOI[C_R_EYE][1] = leftEyeY + offset[1]
            rightEyeX = 0
            rightEyeY = 0
            for i in range(9, 15):
                if (i == 9 or i == 12):
                    continue
                rightEyeX += shape.part(i).x
                rightEyeY += shape.part(i).y
            rightEyeX = int(rightEyeX / 4.0)
            rightEyeY = int(rightEyeY / 4.0)
            eyeCorners[1][0] = [shape.part(9).x + offset[0], shape.part(9).y + offset[1]]
            eyeCorners[1][1] = [shape.part(12).x + offset[0], shape.part(12).y + offset[1]]
            instantFacePOI[C_L_EYE][0] = rightEyeX + offset[0]
            instantFacePOI[C_L_EYE][1] = rightEyeY + offset[1]
            data = [instantFacePOI,
                    (int(d.left() + offset[0]), int(d.top() + offset[1]), int(d.right() + offset[0]),int(d.bottom() + offset[1])),\
                    eyeCorners]
            result.append(data)

            p_lefteye = []
            p_righteye = []
            p_mouse_in = []
            p_mouse_out = []

            p_lefteye.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in LEFT_EYE_MINI])
            p_righteye.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in RIGHT_EYE_MINI])
            p_mouse_out.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in MOUTH_OUTLINE_MINI])
            p_mouse_in.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in MOUTH_INNER_MINI])

            result_other.append([p_lefteye, p_righteye, p_mouse_in, p_mouse_out])

        else:
            # oreille droite
            instantFacePOI[C_R_HEAR][0] = shape.part(0).x + offset[0]
            instantFacePOI[C_R_HEAR][1] = shape.part(0).y + offset[1]
            # oreille gauche
            instantFacePOI[C_L_HEAR][0] = shape.part(16).x + offset[0]
            instantFacePOI[C_L_HEAR][1] = shape.part(16).y + offset[1]
            # nez
            instantFacePOI[C_NOSE][0] = shape.part(30).x + offset[0]
            instantFacePOI[C_NOSE][1] = shape.part(30).y + offset[1]
            # bouche gauche
            instantFacePOI[C_R_MOUTH][0] = shape.part(48).x + offset[0]
            instantFacePOI[C_R_MOUTH][1] = shape.part(48).y + offset[1]
            # bouche droite
            instantFacePOI[C_L_MOUTH][0] = shape.part(54).x + offset[0]
            instantFacePOI[C_L_MOUTH][1] = shape.part(54).y + offset[1]

            leftEyeX = 0
            leftEyeY = 0
            # for i in range(36, 42):
            #     leftEyeX += shape.part(i).x
            #     leftEyeY += shape.part(i).y
            # leftEyeX = int(leftEyeX / 6.0)
            # leftEyeY = int(leftEyeY / 6.0)
            for i in range(37, 42):
                if(i == 39):
                    continue
                leftEyeX += shape.part(i).x
                leftEyeY += shape.part(i).y
            leftEyeX = int(leftEyeX / 4.0)
            leftEyeY = int(leftEyeY / 4.0)
            eyeCorners[0][0] = [shape.part(36).x + offset[0], shape.part(36).y + offset[1]]
            eyeCorners[0][1] = [shape.part(39).x + offset[0], shape.part(39).y + offset[1]]

            instantFacePOI[C_R_EYE][0] = leftEyeX + offset[0]
            instantFacePOI[C_R_EYE][1] = leftEyeY + offset[1]

            rightEyeX = 0
            rightEyeY = 0
            # for i in range(42, 48):
            #     rightEyeX += shape.part(i).x
            #     rightEyeY += shape.part(i).y
            # rightEyeX = int(rightEyeX / 6.0)
            # rightEyeY = int(rightEyeY / 6.0)
            for i in range(43, 48):
                if(i == 45):
                    continue
                rightEyeX += shape.part(i).x
                rightEyeY += shape.part(i).y
            rightEyeX = int(rightEyeX / 4.0)
            rightEyeY = int(rightEyeY / 4.0)
            eyeCorners[1][0] = [shape.part(42).x + offset[0], shape.part(42).y + offset[1]]
            eyeCorners[1][1] = [shape.part(45).x + offset[0], shape.part(45).y + offset[1]]
            instantFacePOI[C_L_EYE][0] = rightEyeX + offset[0]
            instantFacePOI[C_L_EYE][1] = rightEyeY + offset[1]
            data = [instantFacePOI, (
            int(d.left() + offset[0]), int(d.top() + offset[1]), int(d.right() + offset[0]), int(d.bottom() + offset[1])),
                    eyeCorners]
            result.append(data)

            p_lefteye = []
            p_righteye = []
            p_mouse_in = []
            p_mouse_out = []

            p_lefteye.extend([[shape.part(t).x +offset[0], shape.part(t).y+ offset[1]] for t in LEFT_EYE])
            p_righteye.extend([[shape.part(t).x +offset[0], shape.part(t).y+ offset[1]] for t in RIGHT_EYE])
            p_mouse_out.extend([[shape.part(t).x +offset[0], shape.part(t).y+ offset[1]] for t in MOUTH_OUTLINE])
            p_mouse_in.extend([[shape.part(t).x +offset[0], shape.part(t).y+ offset[1]] for t in MOUTH_INNER])

            result_other.append([p_lefteye, p_righteye, p_mouse_in, p_mouse_out])
            # print('result_other', result_other)

    return result, result_other, train_type

def computeGradient(img):
    # out2 = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  # create a receiver array
    if img.shape[0] < 2 or img.shape[1] < 2:  # TODO I'm not sure that secure out of range
        print("EYES too small")
        return out
    for y in range(0, out.shape[0]):
        out[y][0] = img[y][1] - img[y][0]
        for x in range(1, out.shape[1] - 1):
            out[y][x] = (img[y][x + 1] - img[y][x - 1]) / 2.0
        out[y][out.shape[1] - 1] = img[y][out.shape[1] - 1] - img[y][out.shape[1] - 2]
    # cv2.imshow("test",out)
    # cv2.waitKey(0)
    return out


def testPossibleCentersFormula(x, y, weight, gx, gy, out):
    for cy in range(0, out.shape[0]):
        for cx in range(0, out.shape[1]):
            if x == cx and y == cy:
                continue
            dx = x - cx
            dy = y - cy
            magnitude = math.sqrt(dx * dx + dy * dy)
            dx = dx / magnitude
            dy = dy / magnitude
            dotProduct = dx * gx + dy * gy
            dotProduct = max(0.0, dotProduct)
            out[cy][cx] += dotProduct * dotProduct * weight[cy][cx]


def findEyeCenter(eyeImage, offset):
    if (len(eyeImage.shape) <= 0 or eyeImage.shape[0] <= 0 or eyeImage.shape[1] <= 0):
        return tuple(map(operator.add, (0, 0), offset))
    if (int(eyeImage.size / (eyeImage.shape[0] * (eyeImage.shape[1]))) == 3):
        eyeImg = np.asarray(cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY))
    else:
        eyeImg = eyeImage.copy()  #.copy()
    eyeImg = eyeImg.astype(np.float32)
    scaleValue = 1.0;
    if (eyeImg.shape[0] > maxEyeSize or eyeImg.shape[1] > maxEyeSize):
        scaleValue = max(maxEyeSize / float(eyeImg.shape[0]), maxEyeSize / float(eyeImg.shape[1]))
        eyeImg = cv2.resize(eyeImg, None, fx=scaleValue, fy=scaleValue, interpolation=cv2.INTER_AREA)

    # img_int8 = img.astype(np.uint8)
    # eyeImg = eyeImg.astype(np.uint8)
    # eyeImg = cv2.equalizeHist(eyeImg)
    # eyeImg = cv2.GaussianBlur(eyeImg, (3,3), 0)

    gradientX = computeGradient(eyeImg)
    gradientY = np.transpose(computeGradient(np.transpose(eyeImg)))
    gradientMatrix = matrixMagnitude(gradientX, gradientY)

    gradientThreshold = computeDynamicThreshold(gradientMatrix, kGradientThreshold)
    # Normalisation
    for y in range(0, eyeImg.shape[0]):  # Iterate through rows
        for x in range(0, eyeImg.shape[1]):  # Iterate through columns
            if (gradientMatrix[y][x] > gradientThreshold):
                gradientX[y][x] = gradientX[y][x] / gradientMatrix[y][x]
                gradientY[y][x] = gradientY[y][x] / gradientMatrix[y][x]
            else:
                gradientX[y][x] = 0.0
                gradientY[y][x] = 0.0

    # Invert and blur befor algo
    weight = cv2.GaussianBlur(eyeImg, (kWeightBlurSize, kWeightBlurSize), 0)
    for y in range(0, weight.shape[0]):  # Iterate through rows
        for x in range(0, weight.shape[1]):  # Iterate through columns
            weight[y][x] = 255 - weight[y][x]

    outSum = np.zeros((eyeImg.shape[0], eyeImg.shape[1]), dtype=np.float32)  # create a receiver array
    for y in range(0, outSum.shape[0]):  # Iterate through rows
        for x in range(0, outSum.shape[1]):  # Iterate through columns
            if (gradientX[y][x] == 0.0 and gradientY[y][x] == 0.0):
                continue
            testPossibleCentersFormula(x, y, weight, gradientX[y][x], gradientY[y][x], outSum)

    # scale all the values down, basically averaging them
    numGradients = (weight.shape[0] * weight.shape[1]);
    out = np.divide(outSum, numGradients * 10)
    # find maxPoint
    (minval, maxval, mincoord, maxcoord) = cv2.minMaxLoc(out)
    maxcoord = (int(maxcoord[0] / scaleValue), int(maxcoord[1] / scaleValue))
    return tuple(map(operator.add, maxcoord, offset))


def matrixMagnitude(gradX, gradY):
    mags = np.zeros((gradX.shape[0], gradX.shape[1]), dtype=np.float32)  # create a receiver array
    for y in range(0, mags.shape[0]):
        for x in range(0, mags.shape[1]):
            gx = gradX[y][x]
            gy = gradY[y][x]
            magnitude = math.sqrt(gx * gx + gy * gy)
            mags[y][x] = magnitude
    return mags


def computeDynamicThreshold(gradientMatrix, DevFactor):
    (meanMagnGrad, meanMagnGrad) = cv2.meanStdDev(gradientMatrix)
    stdDev = meanMagnGrad[0] / math.sqrt(gradientMatrix.shape[0] * gradientMatrix.shape[1])
    return DevFactor * stdDev + meanMagnGrad[0]


def getEyePOI(eyes):
    result = []
    for eye in eyes:
        left = eye[0][0]
        right = eye[1][0]
        middle = (eye[0][1] + eye[1][1]) / 2.0
        width = eye[1][0] - eye[0][0]
        height = width / 4.0
        result.append((int(left), int(middle - height), int(right), int(middle + height)))
    return result


def scale(rectangle, scale):
    width = rectangle[2] - rectangle[0]
    height = rectangle[3] - rectangle[1]
    midddle = (width / 2 + rectangle[0], height / 2 + rectangle[1])
    left = midddle[0] - int(scale * width / 2)
    top = midddle[1] - int(scale * height / 2)
    right = midddle[0] + int(scale * width / 2)
    bottom = midddle[1] + int(scale * height / 2)
    return (left, top, right, bottom)


def getEyePos(corners, img):
    # here we don't need both but the biggest one
    eyes = getEyePOI(corners)
    # print('corners', corners, '\neyes', eyes)
    choosen = 0
    eyeToConsider = eyes[0]
    if ((eyes[0][0] - eyes[0][2]) > (eyes[1][0] - eyes[1][2])):
        eyeToConsider = eyes[1]
        choosen = 1

    scalesrect = scale(eyeToConsider, 1.2)
    croppedImage = img[
                   int(max(scalesrect[1], 0)):int(max(scalesrect[3], 0)),
                   int(max(scalesrect[0], 0)):int(max(scalesrect[2], 0))
                   ]
    return [findEyeCenter(croppedImage, [scalesrect[0], scalesrect[1]]), corners[choosen]]

def getEyePos2(corners, img, left_or_right=0):
    # here we don't need both but the biggest one
    eyes = getEyePOI(corners)
    # print('corners', corners, '\neyes', eyes)
    if(left_or_right == 0):
        choosen = 0
        eyeToConsider = eyes[0]
    elif(left_or_right == 1):
        choosen = 1
        eyeToConsider = eyes[1]
    else:
        assert(1)

    scalesrect = scale(eyeToConsider, 1.2)
    croppedImage = img[
                   int(max(scalesrect[1], 0)):int(max(scalesrect[3], 0)),
                   int(max(scalesrect[0], 0)):int(max(scalesrect[2], 0))
                   ]
    return [findEyeCenter(croppedImage, [scalesrect[0], scalesrect[1]]), corners[choosen]]


def getEyePos3(corners, img, left_or_right=0):
    scaleValue = 1.5
    # here we don't need both but the biggest one
    eyes = getEyePOI(corners)
    # print('corners', corners, '\neyes', eyes)
    if(left_or_right == 0):
        choosen = 0
        eyeToConsider = eyes[0]
    elif(left_or_right == 1):
        choosen = 1
        eyeToConsider = eyes[1]
    else:
        assert(1)

    scalesrect = scale(eyeToConsider, scaleValue)
    croppedImage = img[
                   int(max(scalesrect[1], 0)):int(max(scalesrect[3], 0)),
                   int(max(scalesrect[0], 0)):int(max(scalesrect[2], 0))
                   ]

    loc_l = gi.locate(croppedImage)
    maxcoord = (int(loc_l[1] ), int(loc_l[0] ))
    # return [t, corners[choosen] ]
    ret = tuple(map(operator.add, maxcoord, [scalesrect[0], scalesrect[1]]))
    print('&&&&&&&&&&&&&&&&&&', scalesrect[0], scalesrect[1], '***', maxcoord)
    return [ret, corners[choosen]]
    # return [findEyeCenter(croppedImage, [scalesrect[0], scalesrect[1]]), corners[choosen]]



#############
def sub_eyecenter_and_pupilcenter(tFacePOI, pupilcenter, cameraMatrix, tproj_matrix):
    print("//////////sub_eyecenter_and_pupilcenter")
    print('FacePOI[5]',tFacePOI[6], pupilcenter[0], pupilcenter[1])
    # print('//////', np.asmatrix(proj_matrix).I)

    #inverse matrix
    # tmatrix = np.eye(4)
    # tmatrix[0:3,0:3] = rt
    # tmatrix[0:3,3] = tvec.T
    # tmatrix = np.asmatrix(tmatrix)
    # tmatrix_inv = tmatrix.I
    # print('tmatrix', tmatrix)
    # print('tmatrix_inv', tmatrix_inv)

    # print(np.ones((tFacePOI.shape[0], tFacePOI.shape[1]+1)))
    imgpos_face = np.ones((tFacePOI.shape[0], tFacePOI.shape[1]+1))
    imgpos_face[:,0:2] = tFacePOI
    print('image_face',imgpos_face)
    a = np.asmatrix(tproj_matrix).I * np.asmatrix(cameraMatrix).I * np.float32(imgpos_face).T
    a = a / a[3]
    print(a)
    b = np.asmatrix(tproj_matrix).I * np.asmatrix(cameraMatrix).I * np.float32([[tFacePOI[6][0], tFacePOI[6][1], 1], [pupilcenter[0], pupilcenter[1], 1]]).T
    b = b / b[3]

    print(b)
    print(np.subtract(b[0:-1,1], b[0:-1,0]))

    return np.subtract(b[0:-1,1], b[0:-1,0]), b
def rotMatFromEye(eyeData):
    # print eyeData
    # eyeDiameter = eyeConst * Distance(eyeData[1][0], eyeData[1][1])
    eyeCenter = ((eyeData[1][0][0] + eyeData[1][1][0]) / 2.0, (eyeData[1][0][1] + eyeData[1][1][1]) / 2.0)
    eyePos = eyeData[0]
    # HERE WE CONSTRUCT A MATRIX OF A BASE WHERE THE UNIT IS THE DIAMETER OF THE EYE AND AXIS OF THIS
    mainEyeAxis = ((eyeData[1][0][0] - eyeData[1][1][0]), (eyeData[1][0][1] - eyeData[1][1][1]))
    secondEyeAxis = perpendicular(mainEyeAxis)

    reverseTransitionMatrix = (mainEyeAxis, secondEyeAxis)

    transitionMatrix = np.linalg.inv(reverseTransitionMatrix)
    print('transitionMatrix', transitionMatrix)
    eyeCenterInEyeRef = np.dot(transitionMatrix, eyeCenter)
    eyeCenterInEyeRef[1] = eyeCenterInEyeRef[1] + 0.2

    eyePosInEyeRef = np.dot(transitionMatrix, eyePos)

    eyeOffset = eyePosInEyeRef - eyeCenterInEyeRef

    eyeOffset = [clamp(eyeOffset[0], -0.99, 0.99), clamp(eyeOffset[1], -0.99, 0.99)]
    # Now we get the rotation values
    thetay = np.arcsin(eyeOffset[0]) * eyeConst
    thetax = np.arcsin(eyeOffset[1]) * eyeConst
    print('각도', thetax*radianToDegree, thetay*radianToDegree)
    print('변환    ',changeRotation_pitchyaw2unitvec('PYR',[thetax,thetay,0],'PYR'))
    # Aaand the rotation matrix
    rot = eulerAnglesToRotationMatrix([thetax, thetay, 0])
    #pitch yaw roll순임

    # print rot
    return rot

def rotMatFromEye2(eyeData):
    eyeConst = 1.0
    # print eyeData
    # eyeDiameter = eyeConst * Distance(eyeData[1][0], eyeData[1][1])
    eyeCenter = ((eyeData[1][0][0] + eyeData[1][1][0]) / 2.0, (eyeData[1][0][1] + eyeData[1][1][1]) / 2.0)
    eyePos = eyeData[0]
    # HERE WE CONSTRUCT A MATRIX OF A BASE WHERE THE UNIT IS THE DIAMETER OF THE EYE AND AXIS OF THIS
    mainEyeAxis = ((eyeData[1][0][0] - eyeData[1][1][0]), (eyeData[1][0][1] - eyeData[1][1][1]))
    secondEyeAxis = perpendicular(mainEyeAxis)

    reverseTransitionMatrix = (mainEyeAxis, secondEyeAxis)

    transitionMatrix = np.linalg.inv(reverseTransitionMatrix)
    print('transitionMatrix', transitionMatrix)
    eyeCenterInEyeRef = np.dot(transitionMatrix, eyeCenter)
    eyeCenterInEyeRef[1] = eyeCenterInEyeRef[1] + 0.2

    eyePosInEyeRef = np.dot(transitionMatrix, eyePos)

    eyeOffset = eyePosInEyeRef - eyeCenterInEyeRef

    eyeOffset = [clamp(eyeOffset[0], -0.99, 0.99), clamp(eyeOffset[1], -0.99, 0.99)]
    # Now we get the rotation values
    thetay = np.arcsin(eyeOffset[0]) * eyeConst
    thetax = np.arcsin(eyeOffset[1]) * eyeConst
    print('각도', thetax*radianToDegree, thetay*radianToDegree)
    print('변환    ',changeRotation_pitchyaw2unitvec('PYR',[thetax,thetay,0],'PYR'))
    # Aaand the rotation matrix
    rot = eulerAnglesToRotationMatrix([thetax, thetay, 0])
    #pitch yaw roll순임

    # print rot
    return rot

# converting from Cartesian Coordinates to Spherical Coordinates.
def changeRotation_pitchyaw2unitvec(typeIn, nR_eulerangle, typeOut ):
    up = np.array([0,0,1])
    print('')
    t_pitch_ang = 0
    t_yaw_ang = 0
    t_roll_ang = 0

    print(" Enter", typeIn, "return", typeOut)
    if (typeIn == "PYR"):  # Pitch / Yaw / Roll
        t_pitch_ang = nR_eulerangle[0]
        t_yaw_ang = nR_eulerangle[1]
        t_roll_ang = nR_eulerangle[2]
    elif (typeIn == "RPY"):  # Roll / Pitch / Yaw
        t_pitch_ang = nR_eulerangle[1]
        t_yaw_ang = nR_eulerangle[2]
        t_roll_ang = nR_eulerangle[0]
    else:
        print("Not support!!",1/0)

    beta_tilt = -t_pitch_ang
    alpha_yaw = t_yaw_ang
    gazeVector = [math.sin(alpha_yaw) * math.cos(beta_tilt), math.sin(beta_tilt),
                  math.cos(alpha_yaw) * math.cos(beta_tilt)]
    # gazeVector = lpupil_roll_pitch_yaw * deg2Rad

    if (typeOut == "PYR"):  # Pitch / Yaw / Roll
        t_x =  gazeVector[0]
        t_y = gazeVector[1]
        t_z = gazeVector[2]
    elif (typeOut == "RPY"):  # Roll / Pitch / Yaw
        t_x = gazeVector[2]
        t_y = gazeVector[0]
        t_z = gazeVector[1]
    else:
        print("Not support!!",1/0)

    print('gazeVector=',typeOut, np.array([t_x, t_y, t_z]))
    return np.array([t_x, t_y, t_z])

# Given the data from a faceExtract
def getCoordFromFace(FacePOI, eyeData, img, cameraMatrix, distCoeffs):
    print("\n//////////////getCoordFromFace")
    # SOLVER FOR PNPs
    retval, rvec, tvec = cv2.solvePnP(ThreeDFacePOI2, FacePOI, cameraMatrix, distCoeffs);
    # rvec[0] = rvec[0]+3.14/10 # roll임 - world coordinate가 y가 위로 +일경우
    # rvec[1] = rvec[1]+3.14/10 # yaw임 (얼굴이 왼쪽으로 +a , 얼굴이 오른쪽으로 -a)
    # rvec[2] = rvec[2]+3.14/10 # pitch임 (얼굴이 위쪽으로 +a , 얼굴이 아래쪽으로 -a)

    rt, jacobian = cv2.Rodrigues(rvec)
    rot2 = rotMatFromEye(eyeData)

    origin = [tvec[0][0], tvec[1][0], tvec[2][0]]
    headDir = np.dot(rot2, np.dot(rt, [0, 0, 1]))
    camPlaneOrthVector = [0, 0, 1]
    pointOnPlan = [0, 0, 0]

    tview_point = intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
    print('tview_point',tview_point)

    temp = np.dot(rt, [0, 0, 1])
    print("eyeData", eyeData[0])
    print("head rot ", rvec.T*radianToDegree)
    # print("head rot yaw", np.dot(rt, [0, 0, 1])*radianToDegree)
    # print("head rot tilt", np.dot(rt, [1, 0, 0])*radianToDegree)
    # print("test", np.dot(rt, [0, 0, 1])[0], np.dot(rt, [0, 0, 1])[1], np.dot(rt, [0, 0, 1])[2], math.atan2(temp[1], temp[0])* radianToDegree)
    # if(math.atan2(temp[1], temp[0]) * radianToDegree-90 < 0):
    #     print('yaw',360 + (math.atan2(temp[1], temp[0]) * radianToDegree-90))
    # else:
    #     print('yaw', math.atan2(temp[1], temp[0]) * radianToDegree-90)
    # print('pitch',-(math.atan2(temp[2], np.sqrt(temp[0]*temp[0]+ temp[1]*temp[1]))* radianToDegree))
    print('head trans [{:03.2f}, {:03.2f}, {:03.2f}]'.format(tvec[0][0], tvec[1][0], tvec[2][0]))

    axis = np.float32([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]]) + ThreeDFacePOI2[2]  #nose

    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
    # modelpts, jac2 = cv2.projectPoints(ThreeDFacePOI2, rvec, tvec, cameraMatrix, distCoeffs)
    rvec_matrix = cv2.Rodrigues(rvec)[0]

    # proj_matrix = np.hstack((np.dot(rot2, rt), tvec))
    proj_matrix = np.hstack((rvec_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    print('eulerAngles', eulerAngles[6])
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles[6]]

    # xpan = np.asmatrix([math.cos(pitch), math.sin(pitch), 0])
    # # xpan = xpan.reshape(1,3)
    # xw = np.dot(np.asmatrix(rt).I, [0, 0, 1])
    # print('roll', xw, ' data', xpan.T)
    # print(np.dot(xw, xpan.T))
    # roll2 = math.acos(np.dot(xw, xpan.T))
    # if (xw[0:1,2] < 0):
    #     roll2 = -roll2;
    # # print("xw[2]", xw[0:1,2])
    # print('roll2', roll2)

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    print('pitch_ {:.02f}, yaw_ {:.02f}, roll_ {:.02f}'.format(pitch,yaw,roll))



    # cv2.putText(img, '^pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(math.degrees(math.asin(math.sin(rvec[0]))), math.degrees(math.asin(math.sin(rvec[1]))), -math.degrees(math.asin(math.sin(rvec[2])))),
    #             (int(eyeData[0][0] - 250), int(eyeData[0][1] - 120)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=8)
    cv2.putText(img, '^pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(math.degrees(math.asin(math.sin(rvec[0]))), math.degrees(math.asin(math.sin(rvec[1]))), -math.degrees(math.asin(math.sin(rvec[2])))),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=8)

    # cv2.putText(img, ' pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(pitch, yaw, roll),
    #             (int(eyeData[0][0] - 250), int(eyeData[0][1] - 60)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=8)
    cv2.putText(img, ' pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(pitch, yaw, roll),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=8)


    nose_end_point2D, jacobian = cv2.projectPoints(np.array([ThreeDFacePOI2[2]]),rvec, tvec, cameraMatrix, distCoeffs)
    # print(nose_end_point2D)

    leye_end_point2D, jacobian = cv2.projectPoints(np.array([(3.0, 0.0, 25.0)]), rvec, tvec, cameraMatrix, distCoeffs)
    # print(reye_end_point2D)


    # pose estimation
    # self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    retGap, retGapCoord = sub_eyecenter_and_pupilcenter(FacePOI, eyeData[0], cameraMatrix, proj_matrix)
    print(np.array([(3.0, 0.0, 25.0)] + retGap.T))
    rvec_pupil = cv2.Rodrigues(np.dot(rot2,rt))[0]
    lpupil_end_point2D, jacobian = cv2.projectPoints(np.array([(3.0, 0.0, 25.0)]), rvec_pupil, tvec+retGap, cameraMatrix, distCoeffs)

    K = 1.31    #distance between eyeball center and pupil center
    K0 = 0.53    #cornea radius

    tretGap = np.array(retGap)
    # tretGap[2] = 0
    tretGapCoord = np.array(retGapCoord.T[1])[0][0:3]
    tretGapCoord[2] = 0
    print('retGapCoord', np.array(retGapCoord.T[1])[0][0:3])

    taa = np.array([ThreeDFacePOI2[6]]) - np.array([(0, 0, K)])
    tbb = np.array([ThreeDFacePOI2[6]]) + tretGap.T
    # tbb = np.array([ThreeDFacePOI2[6]]) + tretGapCoord

    tcc = np.array([ThreeDFacePOI2[6]])
    temp2 = cv2.Rodrigues(tcc - taa)[0]
    print('temp2',temp2, tcc - taa)
    print('taa',taa)
    print('tbb',tbb)
    print('tbb-taa',(tbb - taa))
    print('tcc-taa',(tcc - taa))
    aaa = np.array(tbb - taa)
    bbb = np.array(tcc - taa)
    calc_ang = np.dot(aaa[0],bbb[0])
    print(calc_ang)
    calc_ang2 = np.sqrt(aaa[0][0]*aaa[0][0]+aaa[0][1]*aaa[0][1]+aaa[0][2]*aaa[0][2]) * np.sqrt(bbb[0][0]*bbb[0][0]+bbb[0][1]*bbb[0][1]+bbb[0][2]*bbb[0][2])
    tang = np.arccos(calc_ang/ calc_ang2)
    print('tang',tang*radianToDegree)
    temp  = cv2.Rodrigues(tbb - taa)[0]
    print(temp)
    print("norm", cv2.norm(tbb - taa))
    xx = np.arccos((tbb - taa)/cv2.norm(tbb - taa))
    print('x ang', xx * radianToDegree)
    temp3d = np.array([taa, tbb, (tbb - taa) *10+ taa])
    print(temp3d)

    tangle = math.atan2(cv2.norm(np.cross(aaa, bbb)), np.dot(aaa, bbb.T))
    print(tangle*radianToDegree)

    leyeball_to_pupil_point2D2, jacobian = cv2.projectPoints(temp3d, rvec, tvec,
                                                     cameraMatrix, distCoeffs)
    print('leyeball_to_pupil_point2D2',leyeball_to_pupil_point2D2)
    leyeball_to_pupil_point2D = intersectionWithPlan(origin, np.dot(cv2.Rodrigues(xx)[0], np.dot(rt, [0, 0, 1])), camPlaneOrthVector, pointOnPlan)
    print('leyeball_to_pupil_point2D',leyeball_to_pupil_point2D)

    # tview_point = intersectionWithPlan(origin , headDir, camPlaneOrthVector, pointOnPlan)
    # print('tview_point',np.array(tview_point).ravel())

    return tview_point, nose_end_point2D, leye_end_point2D, imgpts, lpupil_end_point2D, leyeball_to_pupil_point2D, leyeball_to_pupil_point2D2


def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n



def intersectionWithPlan(linePoint, lineDir, planOrth, planPoint):
    d = np.dot(np.subtract(linePoint, planPoint), planOrth) / (np.dot(lineDir, planOrth))
    intersectionPoint = np.subtract(np.multiply(d, lineDir), linePoint)
    return intersectionPoint

def line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi

def draw_xyz_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    # print(corner)
    # print(tuple(imgpts[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
    return img

def getIntersection(line1, line2):
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])

    s2 = np.array(line2[0])
    e2 = np.array(line2[1])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    if abs(a1 - a2) < 1e-8:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (x, y)
#############



class eyeTracker(object):
    def __init__(self, predictor_path):
        self.EyeCloseCounter = 0
        twidth = 640
        theight = 480
        tmaxSize = max(twidth, theight)
        tK = np.array([[tmaxSize, 0, twidth / 2.0], [0, tmaxSize, theight / 2.0], [0, 0, 1]], np.float32)
        tD = np.zeros((5,1))
        self.initilaize_calib(tK, tD)
        self.initialize_p3dmodel(ThreeDFacePOI2)
        self.initilaize_training_path(predictor_path)

        #preprocess
        self.faces_eye = []
        self.faces_status = []
        self.faces_point = []

        #algo_data
        self.mEye_centers_r = []
        self.mEye_centers_l = []
        self.mRT = []
        self.mEularAngle = []
        self.mLandmark_2d = []
        self.mEyeballgaze_l=[]
        self.mEyeballgaze_r = []
        self.mViewpoint_2d_l = []
        self.mViewpoint_2d_r = []
        self.mVpoint_2d_l = []
        self.mVpoint_2d_r = []

        #algo_global_data
        self.gEye_centers_r = []
        self.gEye_centers_l = []
        pass

    def initilaize_training_path(self, predictor_path):
        # predictor_path = './dlib/shape_predictor_68_face_landmarks.dat'
        # predictor_path = './dlib/shape_predictor_5_face_landmarks.dat'
        # face_rec_model_path = './dlib/dlib_face_recognition_resnet_model_v1.dat'

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        # self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    def initilaize_calib(self, tCameraMatrix, tDistCoeffs):
        # cameraMatrix = np.eye(3)  # A checker en fct de l'optique choisie
        # distCoeffs = np.zeros((5, 1))
        self.cameraMatrix = tCameraMatrix
        self.distCoeffs = tDistCoeffs
        print(cameraMatrix, distCoeffs)
        pass

    def initialize_p3dmodel(self, paramPOI):
        # tparamPOI = np.zeros((7, 3), dtype=np.float32)
        # # RIGHTHEAR
        # tparamPOI[0] = [-6., 0., -8.]
        # # LEFTHEAR
        # tparamPOI[1] = [6., 0., -8.]
        # # NOSE
        # tparamPOI[2] = [0., 4., 2.5]
        # # RIGHTMOUTH
        # tparamPOI[3] = [-5., 8., 0.]
        # # LEFTMOUTH
        # tparamPOI[4] = [5., 8., 0.]
        # # RIGHTEYE
        # tparamPOI[5] = [-3.5, 0., -1.]
        # # LEFTEYE
        # tparamPOI[6] = [3.5, 0., -1.]

        self.ref_p3dmodel = paramPOI

    def initilaize_data(self):
        self.gEye_centers_r = []
        self.gEye_centers_l = []
        pass

    def preprocess(self, image, activeROI):
        x = activeROI[0]
        y = activeROI[1]
        w = activeROI[2]
        h = activeROI[3]
        cropImage = image[ int(max(y , 0)):int(max(y + h, 0)),
                           int(max(x , 0)):int(max(x + w, 0))]

        self.faces_eye, self.faces_status, self.faces_point = self.analyseFace_from_dlib(cropImage, self.detector, self.predictor, offset=(x,y))
        if(len(self.faces_eye)):
            print("# of detected : {:d} person".format(len(self.faces_eye)))
        return len(self.faces_eye)

    def algo_run(self, gray, tSelect=0):
        self.mEye_centers_r = []
        self.mEye_centers_l = []
        self.mRT = []
        self.mEularAngle = []
        self.mLandmark_2d = []
        self.mEyeballgaze_l=[]
        self.mEyeballgaze_r = []
        self.mViewpoint_2d_l = []
        self.mViewpoint_2d_r = []
        self.mVpoint_2d_l = []
        self.mVpoint_2d_r = []
        self.mViewpoint_2d_l_three = []
        self.mViewpoint_2d_r_three = []
        for index, POI in enumerate(self.faces_eye):
            # print('\nindex', index, '\nPOI', POI)
            # print(eye_corners.shape)

            time_s = time.time()
            eye_corners = POI[2]
            #Right eye
            eye_center_point_r = getEyePos3(eye_corners, gray, 0)
            self.mEye_centers_r.append(eye_center_point_r)
            #Left eye
            eye_center_point_l = getEyePos3(eye_corners, gray, 1)
            self.mEye_centers_l.append(eye_center_point_l)
            timelap_check('2-1.mEye_centers ', time_s)

            time_s = time.time()
            tR, tT, eulerAngle_degree = self.getWorldCoordFromFace(self.ref_p3dmodel, POI[0], self.cameraMatrix, self.distCoeffs)
            self.mRT.append([tR,tT])
            self.mEularAngle.append(eulerAngle_degree)
            # print('tR',tR,'tT',tT)
            timelap_check('2-2.RT ', time_s)

            time_s = time.time()
            tlandmark_2d = self.getLandmark(self.ref_p3dmodel[C_NOSE], tR, tT)
            self.mLandmark_2d.append(tlandmark_2d)
            # print('tlandmark_2d',  tlandmark_2d[0][0], np.round(tlandmark_2d[1:4,-1]))
            timelap_check('2-3.Landmark ', time_s)

            if(tSelect//10%10 == 1):
                time_s = time.time()
                #Left eyeball gaze
                eyeballgaze_l = self.getEyeballCenterGaze(self.ref_p3dmodel[C_L_EYE], tR, tT)
                self.mEyeballgaze_l.append(eyeballgaze_l)
                # print('eyeballgaze_l', eyeballgaze_l)

                #Right eyeball gaze
                eyeballgaze_r = self.getEyeballCenterGaze(self.ref_p3dmodel[C_R_EYE], tR, tT)
                self.mEyeballgaze_r.append(eyeballgaze_r)
                # print('eyeballgaze_r', eyeballgaze_r)
                timelap_check('2-4.eyeball gaze ', time_s)

            if (tSelect // 100 % 10 == 1):
                time_s = time.time()
                #Left eye gaze - method one
                tViewpoint_2d_l = self.getEyeGaze_method_one(self.mEye_centers_l[index], tR, tT)
                self.mViewpoint_2d_l.append(tViewpoint_2d_l)
                # print('tViewpoint_2d_l', tViewpoint_2d_l)

                #Right eye gaze - method one
                tViewpoint_2d_r = self.getEyeGaze_method_one(self.mEye_centers_r[index], tR, tT)
                self.mViewpoint_2d_r.append(tViewpoint_2d_r)
                # print('tViewpoint_2d_r', tViewpoint_2d_r)
                timelap_check('2-5.eye gaze - method one ', time_s)

            if (tSelect // 1000 % 10 == 1):
                time_s = time.time()
                tpoint_2d_l = self.getEyeGaze_method_two_EyeModel(self.mEye_centers_l[index], tR, tT, self.ref_p3dmodel[C_L_EYE], POI[0][C_L_EYE], k=1.31, k0 =0.53 )
                self.mVpoint_2d_l.append(tpoint_2d_l)

                tpoint_2d_r = self.getEyeGaze_method_two_EyeModel(self.mEye_centers_r[index], tR, tT, self.ref_p3dmodel[C_R_EYE], POI[0][C_R_EYE], k=1.31, k0 =0.53 )
                self.mVpoint_2d_r.append(tpoint_2d_r)
                timelap_check('2-6.eye gaze - method two ', time_s)

            if (tSelect // 100000 % 10 == 1):
                time_s = time.time()
                #Left eye gaze - method one
                tViewpoint_2d_l = self.getEyeGaze_method_three(self.mEye_centers_l[index], tR, tT)
                self.mViewpoint_2d_l_three.append(tViewpoint_2d_l)
                # print('tViewpoint_2d_l', tViewpoint_2d_l)

                #Right eye gaze - method one
                tViewpoint_2d_r = self.getEyeGaze_method_three(self.mEye_centers_r[index], tR, tT)
                self.mViewpoint_2d_r_three.append(tViewpoint_2d_r)
                # print('tViewpoint_2d_r', tViewpoint_2d_r)
                timelap_check('2-5.eye gaze - method one ', time_s)

        if(len(self.gEye_centers_r)):
            self.gEye_centers_r.pop(0)
        if(len(self.gEye_centers_l)):
            self.gEye_centers_l.pop(0)
        self.gEye_centers_r.append(self.mEye_centers_r)
        self.gEye_centers_l.append(self.mEye_centers_l)
        pass

    # this algo should be ready both eye values. (algo do not support one detected eye)
    def algo_ready(self, gray, availFrm):
        ret_l = False
        ret_r = False

        self.mEye_centers_r = []
        self.mEye_centers_l = []
        for index, POI in enumerate(self.faces_eye):
            # print('\nindex', index, '\nPOI', POI)
            # print(eye_corners.shape)

            # time_s = time.time()
            eye_corners = POI[2]
            # Right eye
            eye_center_point_r = getEyePos3(eye_corners, gray, 0)
            self.mEye_centers_r.append(eye_center_point_r)
            # Left eye
            eye_center_point_l = getEyePos3(eye_corners, gray, 1)
            self.mEye_centers_l.append(eye_center_point_l)
            # timelap_check('2-1.mEye_centers ', time_s)

        if (len(self.gEye_centers_r) < availFrm):
            self.gEye_centers_r.append(self.mEye_centers_r)
            if (len(self.gEye_centers_r) == availFrm):
                ret_r = True
        elif (len(self.gEye_centers_r) == availFrm):
            self.gEye_centers_r.pop(0)
            self.gEye_centers_r.append(self.mEye_centers_r)
            ret_r = True

        if (len(self.gEye_centers_l) < availFrm):
            self.gEye_centers_l.append(self.mEye_centers_l)
            if (len(self.gEye_centers_l) == availFrm):
                ret_l = True
        elif (len(self.gEye_centers_l) == availFrm):
            self.gEye_centers_l.pop(0)
            self.gEye_centers_l.append(self.mEye_centers_l)
            ret_l = True

        print('mEye_centers_r', self.mEye_centers_r)
        print('mEye_centers_l', self.mEye_centers_l)

        if(ret_r == True and ret_l == True):
            self.mEye_centers_r = self.get_pupil_center_with_LPF(self.gEye_centers_r, availFrm)
            self.mEye_centers_l = self.get_pupil_center_with_LPF(self.gEye_centers_l, availFrm)
            # self.get_pupil_center_with_LPF(self.gEye_centers_l)

        return ret_r, ret_l

    # def get_pupil_center_with_LPF(self, availFrm):

    def get_pupil_center_with_LPF(self, buffer, availFrm):
        # self.gEye_centers_l
        # self.gEye_centers_r

        # test2 = [
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
        #     [[(982.5, 395.0), [[975., 401.], [1000., 394.]]],
        #      [(456.5, 358.5), [[442., 365.], [467., 367.]]]]
        # ]

        pupil_cen_x = []
        pupil_cen_y = []
        corner_l_pupil_x = []
        corner_l_pupil_y = []
        corner_r_pupil_x = []
        corner_r_pupil_y = []

        for idx, tdata in enumerate(buffer):
            # print(idx, tdata)
            print(buffer[idx])
            tpupil_cen_x = []
            tpupil_cen_y = []
            tpupil_corner_l_x = []
            tpupil_corner_l_y = []
            tpupil_corner_r_x = []
            tpupil_corner_r_y = []

            for i, idata in enumerate(buffer[idx]):
                print(i, idata)
                print('----', idata[0], '-----', idata[1][0], idata[1][1])
                tpupil_cen_x.append(idata[0][0])
                tpupil_cen_y.append(idata[0][1])
                tpupil_corner_l_x.append(idata[1][0][0])
                tpupil_corner_l_y.append(idata[1][0][1])
                tpupil_corner_r_x.append(idata[1][1][0])
                tpupil_corner_r_y.append(idata[1][1][1])

                # print('tpupil_cen_x', tpupil_cen_x)


            pupil_cen_x.extend([tpupil_cen_x])
            pupil_cen_y.extend([tpupil_cen_y])
            corner_l_pupil_x.extend([tpupil_corner_l_x])
            corner_l_pupil_y.extend([tpupil_corner_l_y])
            corner_r_pupil_x.extend([tpupil_corner_r_x])
            corner_r_pupil_y.extend([tpupil_corner_r_y])

        print('pupil_cen_x', pupil_cen_x)
        print('pupil_cen_y', pupil_cen_y)

        # print(i)
        localeye_centers = []

        for e in range(i + 1):
            print('lpupil_cen_x_np_{}_column={}'.format(e, np.array(pupil_cen_x)[:, e]))
            print('lpupil_cen_y_np_{}_column={}'.format(e, np.array(pupil_cen_y)[:, e]))
            print('center', np.average(np.array(pupil_cen_x)[:, e]), np.average(np.array(pupil_cen_y)[:, e]) )
            print('pupil_corner', idata[1][0], idata[1][1])
            print('pupil_corner_1', idata[1])
            # localeye_centers.append([(np.mean(np.array(pupil_cen_x)[:, e]), np.mean(np.array(pupil_cen_y)[:, e])),np.array(idata[1])])
            temp = [[np.mean(np.array(corner_l_pupil_x)[:, e]),np.mean(np.array(corner_l_pupil_y)[:, e])],[np.mean(np.array(corner_r_pupil_x)[:, e]),np.mean(np.array(corner_r_pupil_y)[:, e])]]
            print('temp',temp)
            localeye_centers.append([(np.mean(np.array(pupil_cen_x)[:, e]), np.mean(np.array(pupil_cen_y)[:, e])),np.array(temp)])

            # print(np.array(buffer)[:, 0][:, 1][:, 0])
            # localeye_centers.append([(np.mean(np.array(pupil_cen_x)[:, e]), np.mean(np.array(pupil_cen_y)[:, e])),[np.array(buffer)[:, 0][:, 1][:, 0], np.array(buffer)[:, 0][:, 1][:, 1]]])
            # print(np.array(buffer)[:, 0][:, 1][:, 0][-1]))
            # print(np.array(buffer)[:, 0][:, 1][:, 1][-1])

            # low_pass_filter
            # ret_data_x = test_use_previous_to_one_result(np.array(lpupil_cen_x)[:, e],14)
            # ret_data_y = test_use_previous_to_one_result(np.array(lpupil_cen_y)[:, e], 14)
            # localeye_centers.append([(ret_data_x[14], ret_data_y[14]),np.array(idata[1])])
            # localeye_centers.append([(ret_data_x[7], ret_data_y[7]), np.array(idata[1])])
            # print('ret_data_x', ret_data_x)
            # print(1/0)

        # print('lpupil_cen_x_np_first_column', np.array(lpupil_cen_x)[:,0])
        # print('lpupil_cen_x_np_second_column', np.array(lpupil_cen_x)[:,1])
        # print('lpupil_cen_y_np', lpupil_cen_y)

        # print('rpupil_cen', rpupil_cen)
        print('localeye_centers', localeye_centers)
        return localeye_centers



    def algo_ready_next(self, gray, tSelect=0):
        # self.mEye_centers_r = []
        # self.mEye_centers_l = []
        self.mRT = []
        self.mEularAngle = []
        self.mLandmark_2d = []
        self.mEyeballgaze_l=[]
        self.mEyeballgaze_r = []
        self.mViewpoint_2d_l = []
        self.mViewpoint_2d_r = []
        self.mVpoint_2d_l = []
        self.mVpoint_2d_r = []
        self.mViewpoint_2d_l_three = []
        self.mViewpoint_2d_r_three = []

        # print('self.gEye_centers_l', self.gEye_centers_l)
        # print('self.gEye_centers_r', self.gEye_centers_r)
        # self.get_pupil_center_with_LPF(15)
        # print(1/0)
        for index, POI in enumerate(self.faces_eye):
            # print('\nindex', index, '\nPOI', POI)
            # print(eye_corners.shape)

            # time_s = time.time()
            # eye_corners = POI[2]
            # #Right eye
            # print('eye_corners',eye_corners)
            # eye_center_point_r = getEyePos3(eye_corners, gray, 0)
            # self.mEye_centers_r.append(eye_center_point_r)
            # print('mEye_centers_r', self.mEye_centers_r)
            # #Left eye
            # eye_center_point_l = getEyePos3(eye_corners, gray, 1)
            # self.mEye_centers_l.append(eye_center_point_l)
            # timelap_check('2-1.mEye_centers ', time_s)

            time_s = time.time()
            tR, tT, eulerAngle_degree = self.getWorldCoordFromFace(self.ref_p3dmodel, POI[0], self.cameraMatrix, self.distCoeffs)
            self.mRT.append([tR,tT])
            self.mEularAngle.append(eulerAngle_degree)
            print('tR',tR,'tT',tT)
            timelap_check('2-2.RT ', time_s)

            time_s = time.time()
            tlandmark_2d = self.getLandmark(self.ref_p3dmodel[C_NOSE], tR, tT)
            self.mLandmark_2d.append(tlandmark_2d)
            print('mLandmark_2d',  self.mLandmark_2d)
            timelap_check('2-3.Landmark ', time_s)

            if(tSelect//10%10 == 1):
                time_s = time.time()
                #Left eyeball gaze
                eyeballgaze_l = self.getEyeballCenterGaze(self.ref_p3dmodel[C_L_EYE], tR, tT)
                self.mEyeballgaze_l.append(eyeballgaze_l)
                # print('eyeballgaze_l', eyeballgaze_l)

                #Right eyeball gaze
                eyeballgaze_r = self.getEyeballCenterGaze(self.ref_p3dmodel[C_R_EYE], tR, tT)
                self.mEyeballgaze_r.append(eyeballgaze_r)
                # print('eyeballgaze_r', eyeballgaze_r)
                timelap_check('2-4.eyeball gaze ', time_s)

            if (tSelect // 100 % 10 == 1):
                time_s = time.time()
                #Left eye gaze - method one
                tViewpoint_2d_l = self.getEyeGaze_method_one(self.mEye_centers_l[index], tR, tT)
                self.mViewpoint_2d_l.append(tViewpoint_2d_l)
                # print('tViewpoint_2d_l', tViewpoint_2d_l)

                #Right eye gaze - method one
                tViewpoint_2d_r = self.getEyeGaze_method_one(self.mEye_centers_r[index], tR, tT)
                self.mViewpoint_2d_r.append(tViewpoint_2d_r)
                # print('tViewpoint_2d_r', tViewpoint_2d_r)
                timelap_check('2-5.eye gaze - method one ', time_s)

            if (tSelect // 1000 % 10 == 1):
                time_s = time.time()
                tpoint_2d_l = self.getEyeGaze_method_two_EyeModel(self.mEye_centers_l[index], tR, tT, self.ref_p3dmodel[C_L_EYE], POI[0][C_L_EYE], k=1.31, k0 =0.53 )
                self.mVpoint_2d_l.append(tpoint_2d_l)

                tpoint_2d_r = self.getEyeGaze_method_two_EyeModel(self.mEye_centers_r[index], tR, tT, self.ref_p3dmodel[C_R_EYE], POI[0][C_R_EYE], k=1.31, k0 =0.53 )
                self.mVpoint_2d_r.append(tpoint_2d_r)
                timelap_check('2-6.eye gaze - method two ', time_s)

            if (tSelect // 100000 % 10 == 1):
                time_s = time.time()
                #Left eye gaze - method one
                tViewpoint_2d_l = self.getEyeGaze_method_three(self.mEye_centers_l[index], tR, tT)
                self.mViewpoint_2d_l_three.append(tViewpoint_2d_l)
                # print('tViewpoint_2d_l', tViewpoint_2d_l)

                #Right eye gaze - method one
                tViewpoint_2d_r = self.getEyeGaze_method_three(self.mEye_centers_r[index], tR, tT)
                self.mViewpoint_2d_r_three.append(tViewpoint_2d_r)
                # print('tViewpoint_2d_r', tViewpoint_2d_r)
                timelap_check('2-5.eye gaze - method one ', time_s)
        pass


    def rendering(self, image, tSelect=0):
        for index, (p_leye, p_reye, p_mouthin, p_mouthout) in enumerate(self.faces_status):
            # print(index, '\n', p_leye, '\n', p_reye,'\n', p_mouthin,'\n', p_mouthout)
            # treye_text = "None"
            # tleye_text = "None"
            tleye_status, tleye_text, tleye_value = self.eye_aspect_ratio(p_leye)
            treye_status, treye_text, treye_value = self.eye_aspect_ratio(p_reye)
            tmouth_status, tmouth_text = self.mouth_open(p_mouthin, p_mouthout)
            cv2.putText(image,
                        'EyeRL=[{:s},{:s}],Mouth={:s}'.format(tleye_text, treye_text, tmouth_text),
                        (max(0,p_reye[0][0]-200), max(0,p_reye[0][1]-50)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=4)
            self.drowsiness_detect(image, tleye_value, p_reye[0])

        for index, POI in enumerate(self.faces_eye):

            #landmark
            draw_xyz_axis(image, self.mLandmark_2d[index][0], np.round(self.mLandmark_2d[index][1:4,-1]))
            # print('tlandmark_2d',  tlandmark_2d[0][0], np.round(tlandmark_2d[1:4,-1]))

            #face pitch, yaw, roll
            # print(POI[2][0][0][0], POI[2][0][0][1])
            cv2.putText(image,'pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(self.mEularAngle[index][0],self.mEularAngle[index][1],self.mEularAngle[index][2]),
                        # (10, 30),
                        (max(0, int(POI[2][0][0][0] - 200)), max(0, int(POI[2][0][0][1] - 80))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2, lineType=4)

            #eye center and eye corner
            if(tSelect%10 == 1):
                cv2.circle(image, (int(self.mEye_centers_r[index][0][0]), int(self.mEye_centers_r[index][0][1])), 2, (255, 0, 0), -1)
                cv2.circle(image, (int(self.mEye_centers_r[index][1][0][0]), int(self.mEye_centers_r[index][1][0][1])), 2, (0, 0, 255), -1)
                cv2.circle(image, (int(self.mEye_centers_r[index][1][1][0]), int(self.mEye_centers_r[index][1][1][1])), 2, (0, 0, 255), -1)
                cv2.circle(image, (int(self.mEye_centers_l[index][0][0]), int(self.mEye_centers_l[index][0][1])), 2, (255, 0, 0), -1)
                cv2.circle(image, (int(self.mEye_centers_l[index][1][0][0]), int(self.mEye_centers_l[index][1][0][1])), 2, (0, 0, 255), -1)
                cv2.circle(image, (int(self.mEye_centers_l[index][1][1][0]), int(self.mEye_centers_l[index][1][1][1])), 2, (0, 0, 255), -1)


            #Left/Right eyeball gaze
            if(tSelect//10%10 == 1):
                cv2.line(image, (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                         (int(self.mEyeballgaze_l[index][0][0][0]), int(self.mEyeballgaze_l[index][0][0][1])), (255, 0, 255), 2, -1)
                cv2.line(image, (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                         (int(self.mEyeballgaze_r[index][0][0][0]), int(self.mEyeballgaze_r[index][0][0][1])), (255, 0, 255), 2, -1)

            # Left/Right eye gaze - method one
            if (tSelect // 100 % 10 == 1):
                cv2.line(image,  (int(self.mEye_centers_l[index][0][0]),int(self.mEye_centers_l[index][0][1])),
                         (int(self.mEye_centers_l[index][0][0] - self.mViewpoint_2d_l[index][0]),
                          int(self.mEye_centers_l[index][0][1] - self.mViewpoint_2d_l[index][1])),
                         (255, 255, 0), 2, -1)
                cv2.line(image,  (int(self.mEye_centers_r[index][0][0]),int(self.mEye_centers_r[index][0][1])),
                         (int(self.mEye_centers_r[index][0][0] - self.mViewpoint_2d_r[index][0]),
                          int(self.mEye_centers_r[index][0][1] - self.mViewpoint_2d_r[index][1])),
                         (255, 255, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                #          (int(self.mEye_centers_l[index][0][0] - self.mViewpoint_2d_l[index][0]),
                #           int(self.mEye_centers_l[index][0][1] - self.mViewpoint_2d_l[index][1])),
                #          (255, 255, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                #          (int(self.mEye_centers_r[index][0][0] - self.mViewpoint_2d_r[index][0]),
                #           int(self.mEye_centers_r[index][0][1] - self.mViewpoint_2d_r[index][1])),
                #          (255, 255, 0), 2, -1)

            # Left/Right eye gaze - method two
            if (tSelect // 1000 % 10 == 1):
                cv2.line(image, (int(self.mEye_centers_l[index][0][0]),int(self.mEye_centers_l[index][0][1])),
                         (int(self.mVpoint_2d_l[index][2][0][0]), int(self.mVpoint_2d_l[index][2][0][1])), (0, 255, 255), 2, -1)
                cv2.line(image, (int(self.mEye_centers_r[index][0][0]),int(self.mEye_centers_r[index][0][1])),
                         (int(self.mVpoint_2d_r[index][2][0][0]), int(self.mVpoint_2d_r[index][2][0][1])), (0, 255, 255), 2, -1)
                # cv2.line(image, (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                #          (int(self.mVpoint_2d_l[index][2][0][0]), int(self.mVpoint_2d_l[index][2][0][1])), (0, 255, 255), 2, -1)
                # cv2.line(image, (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                #          (int(self.mVpoint_2d_r[index][2][0][0]), int(self.mVpoint_2d_r[index][2][0][1])), (0, 255, 255), 2, -1)

            # Left/Right eye gaze - method three
            if (tSelect // 100000 % 10 == 1):
                cv2.line(image,  (int(self.mEye_centers_l[index][0][0]),int(self.mEye_centers_l[index][0][1])),
                         (int(self.mEye_centers_l[index][0][0] + self.mViewpoint_2d_l_three[index][0]),
                          int(self.mEye_centers_l[index][0][1] + self.mViewpoint_2d_l_three[index][1])),
                         (0, 140, 0), 2, -1)
                cv2.line(image,  (int(self.mEye_centers_r[index][0][0]),int(self.mEye_centers_r[index][0][1])),
                         (int(self.mEye_centers_r[index][0][0] + self.mViewpoint_2d_r_three[index][0]),
                          int(self.mEye_centers_r[index][0][1] + self.mViewpoint_2d_r_three[index][1])),
                         (0, 140, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                #          (int(self.mEye_centers_l[index][0][0] + self.mViewpoint_2d_l_three[index][0]),
                #           int(self.mEye_centers_l[index][0][1] + self.mViewpoint_2d_l_three[index][1])),
                #          (0, 140, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                #          (int(self.mEye_centers_r[index][0][0] + self.mViewpoint_2d_r_three[index][0]),
                #           int(self.mEye_centers_r[index][0][1] + self.mViewpoint_2d_r_three[index][1])),
                #          (0, 140, 0), 2, -1)

            if (tSelect // 10000 % 10 == 1):
                for (sX, sY) in self.faces_point[index]:
                    cv2.circle(image, (sX, sY), 1, (255, 0, 0), -1)
        pass

    def rendering_with_filter(self, image, tSelect=0):
        for index, (p_leye, p_reye, p_mouthin, p_mouthout) in enumerate(self.faces_status):
            # print(index, '\n', p_leye, '\n', p_reye,'\n', p_mouthin,'\n', p_mouthout)
            # treye_text = "None"
            # tleye_text = "None"
            tleye_status, tleye_text, tleye_value = self.eye_aspect_ratio(p_leye)
            treye_status, treye_text, treye_value = self.eye_aspect_ratio(p_reye)
            tmouth_status, tmouth_text = self.mouth_open(p_mouthin, p_mouthout)
            cv2.putText(image,
                        'EyeRL=[{:s},{:s}],Mouth={:s}'.format(tleye_text, treye_text, tmouth_text),
                        (max(0,p_reye[0][0]-200), max(0,p_reye[0][1]-50)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=4)
            self.drowsiness_detect(image, tleye_value, p_reye[0])

        for index, POI in enumerate(self.faces_eye):

            #landmark
            draw_xyz_axis(image, self.mLandmark_2d[index][0], np.round(self.mLandmark_2d[index][1:4,-1]))
            # print('tlandmark_2d',  tlandmark_2d[0][0], np.round(tlandmark_2d[1:4,-1]))

            #face pitch, yaw, roll
            # print(POI[2][0][0][0], POI[2][0][0][1])
            cv2.putText(image,'pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(self.mEularAngle[index][0],self.mEularAngle[index][1],self.mEularAngle[index][2]),
                        # (10, 30),
                        (max(0, int(POI[2][0][0][0] - 200)), max(0, int(POI[2][0][0][1] - 80))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2, lineType=4)

            #eye center and eye corner
            if(tSelect%10 == 1):
                cv2.circle(image, (int(self.gEye_centers_r[-1][index][0][0]), int(self.gEye_centers_r[-1][index][0][1])), 2, (255, 0, 0), -1)
                cv2.circle(image, (int(self.gEye_centers_r[-1][index][1][0][0]), int(self.gEye_centers_r[-1][index][1][0][1])), 2, (0, 0, 255), -1)
                cv2.circle(image, (int(self.gEye_centers_r[-1][index][1][1][0]), int(self.gEye_centers_r[-1][index][1][1][1])), 2, (0, 0, 255), -1)
                cv2.circle(image, (int(self.gEye_centers_l[-1][index][0][0]), int(self.gEye_centers_l[-1][index][0][1])), 2, (255, 0, 0), -1)
                cv2.circle(image, (int(self.gEye_centers_l[-1][index][1][0][0]), int(self.gEye_centers_l[-1][index][1][0][1])), 2, (0, 0, 255), -1)
                cv2.circle(image, (int(self.gEye_centers_l[-1][index][1][1][0]), int(self.gEye_centers_l[-1][index][1][1][1])), 2, (0, 0, 255), -1)


            #Left/Right eyeball gaze
            if(tSelect//10%10 == 1):
                cv2.line(image, (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                         (int(self.mEyeballgaze_l[index][0][0][0]), int(self.mEyeballgaze_l[index][0][0][1])), (255, 0, 255), 2, -1)
                cv2.line(image, (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                         (int(self.mEyeballgaze_r[index][0][0][0]), int(self.mEyeballgaze_r[index][0][0][1])), (255, 0, 255), 2, -1)

            # Left/Right eye gaze - method one
            if (tSelect // 100 % 10 == 1):
                cv2.line(image,  (int(self.gEye_centers_l[-1][index][0][0]),int(self.gEye_centers_l[-1][index][0][1])),
                         (int(self.gEye_centers_l[-1][index][0][0] - self.mViewpoint_2d_l[index][0]),
                          int(self.gEye_centers_l[-1][index][0][1] - self.mViewpoint_2d_l[index][1])),
                         (255, 255, 0), 2, -1)
                cv2.line(image,  (int(self.gEye_centers_r[-1][index][0][0]),int(self.gEye_centers_r[-1][index][0][1])),
                         (int(self.gEye_centers_r[-1][index][0][0] - self.mViewpoint_2d_r[index][0]),
                          int(self.gEye_centers_r[-1][index][0][1] - self.mViewpoint_2d_r[index][1])),
                         (255, 255, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                #          (int(self.mEye_centers_l[index][0][0] - self.mViewpoint_2d_l[index][0]),
                #           int(self.mEye_centers_l[index][0][1] - self.mViewpoint_2d_l[index][1])),
                #          (255, 255, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                #          (int(self.mEye_centers_r[index][0][0] - self.mViewpoint_2d_r[index][0]),
                #           int(self.mEye_centers_r[index][0][1] - self.mViewpoint_2d_r[index][1])),
                #          (255, 255, 0), 2, -1)

            # Left/Right eye gaze - method two
            if (tSelect // 1000 % 10 == 1):
                cv2.line(image, (int(self.gEye_centers_l[-1][index][0][0]),int(self.gEye_centers_l[-1][index][0][1])),
                         (int(self.mVpoint_2d_l[index][2][0][0]), int(self.mVpoint_2d_l[index][2][0][1])), (0, 255, 255), 2, -1)
                cv2.line(image, (int(self.gEye_centers_r[-1][index][0][0]),int(self.gEye_centers_r[-1][index][0][1])),
                         (int(self.mVpoint_2d_r[index][2][0][0]), int(self.mVpoint_2d_r[index][2][0][1])), (0, 255, 255), 2, -1)
                # cv2.line(image, (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                #          (int(self.mVpoint_2d_l[index][2][0][0]), int(self.mVpoint_2d_l[index][2][0][1])), (0, 255, 255), 2, -1)
                # cv2.line(image, (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                #          (int(self.mVpoint_2d_r[index][2][0][0]), int(self.mVpoint_2d_r[index][2][0][1])), (0, 255, 255), 2, -1)

            # Left/Right eye gaze - method three
            if (tSelect // 100000 % 10 == 1):
                cv2.line(image,  (int(self.gEye_centers_l[-1][index][0][0]),int(self.gEye_centers_l[-1][index][0][1])),
                         (int(self.gEye_centers_l[-1][index][0][0] + self.mViewpoint_2d_l_three[index][0]),
                          int(self.gEye_centers_l[-1][index][0][1] + self.mViewpoint_2d_l_three[index][1])),
                         (0, 140, 0), 2, -1)
                cv2.line(image,  (int(self.gEye_centers_r[-1][index][0][0]),int(self.gEye_centers_r[-1][index][0][1])),
                         (int(self.gEye_centers_r[-1][index][0][0] + self.mViewpoint_2d_r_three[index][0]),
                          int(self.gEye_centers_r[-1][index][0][1] + self.mViewpoint_2d_r_three[index][1])),
                         (0, 140, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_L_EYE][0]),int(POI[0][C_L_EYE][1])),
                #          (int(self.mEye_centers_l[index][0][0] + self.mViewpoint_2d_l_three[index][0]),
                #           int(self.mEye_centers_l[index][0][1] + self.mViewpoint_2d_l_three[index][1])),
                #          (0, 140, 0), 2, -1)
                # cv2.line(image,  (int(POI[0][C_R_EYE][0]),int(POI[0][C_R_EYE][1])),
                #          (int(self.mEye_centers_r[index][0][0] + self.mViewpoint_2d_r_three[index][0]),
                #           int(self.mEye_centers_r[index][0][1] + self.mViewpoint_2d_r_three[index][1])),
                #          (0, 140, 0), 2, -1)

            if (tSelect // 10000 % 10 == 1):
                for (sX, sY) in self.faces_point[index]:
                    cv2.circle(image, (sX, sY), 1, (255, 0, 0), -1)
        pass

    # Given the data from a faceExtract
    def getWorldCoordFromFace(self, ref_point, image_point, cameraMatrix, distCoeffs):
        # print("\n//////////////getCoordFromFace")
        # SOLVER FOR PNPs
        retval, rvec, tvec = cv2.solvePnP(ref_point, image_point, cameraMatrix, distCoeffs);
        # print('retval', retval)
        # rvec[0] = rvec[0]+3.14/10 # roll임 - world coordinate가 y가 위로 +일경우
        # rvec[1] = rvec[1]+3.14/10 # yaw임 (얼굴이 왼쪽으로 +a , 얼굴이 오른쪽으로 -a)
        # rvec[2] = rvec[2]+3.14/10 # pitch임 (얼굴이 위쪽으로 +a , 얼굴이 아래쪽으로 -a)


        rvec_3x3 = cv2.Rodrigues(rvec)[0]
        # print(rvec_3x3)
        pitch_yaw_roll = rotationMatrixToEulerAngles(rvec_3x3)
        pitch = -pitch_yaw_roll[0] * rad2Deg
        yaw = -pitch_yaw_roll[1] * rad2Deg
        roll = pitch_yaw_roll[2] * rad2Deg

        #ref_point의 축이 변경되면, pitch yaw roll이 변경될수 있음
        # pitch = -math.degrees(math.asin(math.sin(rvec[0])))
        # yaw = -math.degrees(math.asin(math.sin(rvec[1])))
        # roll = math.degrees(math.asin(math.sin(rvec[2])))

        #2번째
        # rvec_matrix = cv2.Rodrigues(rvec)[0]
        # proj_matrix = np.hstack((rvec_matrix, tvec))
        # eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
        # print('eulerAngles', eulerAngles[6])
        # pitch, yaw, roll = [math.radians(_) for _ in eulerAngles[6]]
        # pitch = math.degrees(math.asin(math.sin(pitch)))
        # yaw = math.degrees(math.asin(math.sin(yaw)))
        # roll = -math.degrees(math.asin(math.sin(roll)))

        print('pitch_ {:.02f}, yaw_ {:.02f}, roll_ {:.02f}'.format(pitch, yaw, roll))

        # cv2.putText(img, '^pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(math.degrees(math.asin(math.sin(rvec[0]))), math.degrees(math.asin(math.sin(rvec[1]))), -math.degrees(math.asin(math.sin(rvec[2])))),
        #             (int(eyeData[0][0] - 250), int(eyeData[0][1] - 120)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=8)
        # cv2.putText(img, '^pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(math.degrees(math.asin(math.sin(rvec[0]))),
        #                                                                     math.degrees(math.asin(math.sin(rvec[1]))),
        #                                                                     -math.degrees(
        #                                                                         math.asin(math.sin(rvec[2])))),
        #             (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=8)


        return rvec, tvec, (pitch, yaw, roll)

    def getLandmark(self, ref_point, rvec, tvec):

        tAxis = np.float32([[0, 0, 0],
                           [10, 0, 0],
                           [0, 10, 0],
                           [0, 0, 10]]) + ref_point  #nose

        landmark_points, jac = cv2.projectPoints(tAxis, rvec, tvec, self.cameraMatrix, self.distCoeffs)
        return landmark_points

    def getEyeGaze_method_one(self, eyeData, rvec, tvec):
        rt, jacobian = cv2.Rodrigues(rvec)
        rot2 = rotMatFromEye(eyeData)

        headPos = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
        headDir = np.dot(rot2, np.dot(rt, [0, 0, 1]))
        camPlaneOrthVector = np.array([0, 0, 1])
        pointOnPlan = np.array([0, 0, 0])

        tview_points = intersectionWithPlan(headPos, headDir, camPlaneOrthVector, pointOnPlan)
        # tview_points = line_plane_collision(camPlaneOrthVector, pointOnPlan, headDir, headPos)

        print('tview_point', tview_points)

        return tview_points

    def getEyeballCenterGaze(self, ref_point, rvec, tvec):
        temp_point = ref_point.copy()
        temp_point[2] = 25

        print(temp_point)
        eyeballCenterGaze, jac = cv2.projectPoints(temp_point, rvec, tvec, self.cameraMatrix, self.distCoeffs)
        # print('eyeballCenterGaze', eyeballCenterGaze)
        return eyeballCenterGaze

    '''
    it is for 3d 
    K = 1.31    #distance between eyeball center and pupil center
    K0 = 0.53    #cornea radius
    '''
    def getEyeGaze_method_two_EyeModel(self, pupilC_pnt, rvec, tvec, ref_eyeC_pnt, img_eyeC_pnt, k=1.31, k0 = 0.53):
        eyeConst = 10
        # print("\n\n///getEyeGaze_method_two_EyeModel//////////\n")
        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, tvec))
        # eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
        # print('eulerAngles', eulerAngles[6])

        retGap, retGapCoord = self.extract_eyeballcenter_to_pupilcenter(img_eyeC_pnt, pupilC_pnt[0], proj_matrix, self.cameraMatrix, self.distCoeffs)
        # print(np.array([(3.0, 0.0, 25.0)] + retGap.T))
        # rvec_pupil = cv2.Rodrigues(np.dot(rot2,rt))[0]
        # lpupil_end_point2D, jacobian = cv2.projectPoints(np.array([(3.0, 0.0, 25.0)]), rvec_pupil, tvec+retGap, cameraMatrix, distCoeffs)

        tretGap = np.array(retGap)
        # tretGap[2] = 0
        tretGapCoord = np.array(retGapCoord.T[1])[0][0:3]
        tretGapCoord[2] = 0
        print('retGapCoord', [np.array(retGapCoord.T[1])[0][0:3]])

        tc_eyeball = np.array([ref_eyeC_pnt]) - np.array([(0, 0, k)])
        tc_pupil= [np.array(retGapCoord.T[1])[0][0:3]]
        tc_eyemodel = np.array([ref_eyeC_pnt])

        print('tc_eyeball',tc_eyeball)
        print('tc_pupil',tc_pupil)
        print('tc_pupil-tc_eyeball',(tc_pupil - tc_eyeball))
        print('tc_eyemodel-tc_eyeball',(tc_eyemodel - tc_eyeball))

        eyeGazePoint_3d = np.array([tc_eyeball, tc_pupil, (tc_pupil - tc_eyeball) * eyeConst + tc_pupil])
        print('3d eye gaze vector\n', eyeGazePoint_3d)

        eyeGazePoint_2d, jac = cv2.projectPoints(eyeGazePoint_3d, rvec, tvec,
                                                         self.cameraMatrix, self.distCoeffs)
        print('2d eye gaze point\n',eyeGazePoint_2d)
        # leyeball_to_pupil_point2D = intersectionWithPlan(origin, np.dot(cv2.Rodrigues(xx)[0], np.dot(rt, [0, 0, 1])), camPlaneOrthVector, pointOnPlan)
        # print('leyeball_to_pupil_point2D',leyeball_to_pupil_point2D)

        # tview_point = intersectionWithPlan(origin , headDir, camPlaneOrthVector, pointOnPlan)
        # print('tview_point',np.array(tview_point).ravel())

        return eyeGazePoint_2d

    def extract_eyeballcenter_to_pupilcenter(self, img_eyeC_pnt, pupilcenter, tproj_matrix, tcamera_matrix, tdist_coeffs):
        print("//////////sub_eyecenter_and_pupilcenter")
        print('image eyecenter ({:04.2f} {:04.2f}), pupil center ({:04.2f}, {:04.2f}) '.format(img_eyeC_pnt[0], img_eyeC_pnt[1], pupilcenter[0], pupilcenter[1]))
        # print('//////', np.asmatrix(proj_matrix).I)

        undist_ratio = cv2.undistortPoints(np.float32([[img_eyeC_pnt[0], img_eyeC_pnt[1]], [pupilcenter[0], pupilcenter[1]]]), tcamera_matrix, tdist_coeffs)
        undist_ratio = undist_ratio.reshape(-1, 2)

        undist_p = np.ones((undist_ratio.shape[0], undist_ratio.shape[1]+1))
        undist_p[:,:-1] = undist_ratio
        # print('undist_p', undist_p)
        # print('calc_cameramatix', tcamera_matrix * np.asmatrix(undist_p.T))

        ret_undist_p = (tcamera_matrix * np.asmatrix(undist_p.T)).T
        b = np.asmatrix(tproj_matrix).I * np.asmatrix(tcamera_matrix).I * ret_undist_p.T

        # b = np.asmatrix(tproj_matrix).I * np.asmatrix(tcamera_matrix).I * np.float32(
        #     [[img_eyeC_pnt[0], img_eyeC_pnt[1], 1], [pupilcenter[0], pupilcenter[1], 1]]).T
        b = b / b[3]

        print(b)
        print(np.subtract(b[0:-1, 1], b[0:-1, 0]))

        return np.subtract(b[0:-1, 1], b[0:-1, 0]), b

    def getEyeGaze_method_three(self, eyeData, rvec, tvec):
        rt_3x3, jacobian = cv2.Rodrigues(rvec)
        rot2 = rotMatFromEye2(eyeData)

        rt_2 = np.dot(eulerAnglesToRotationMatrix(np.array([0, math.pi, 0])), np.array([1, 1, 1])).round(5)
        # rt = eulerAnglesToRotationMatrix(headOri_radian * rt_2)
        rt = rt_3x3 * rt_2
        # rot2_mid = eulerAnglesToRotationMatrix(mideye_roll_pitch_yaw_rad)
        # headDir_mid = np.dot(np.dot(rot2_mid, rt), [1, 0, 0])


        headPos = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
        headDir = np.dot(rot2, np.dot(rt, np.array([0, 0, 1])))
        camPlaneOrthVector = np.array([0, 0, 1])
        pointOnPlan = np.array([0, 0, 0])

        # tview_points = intersectionWithPlan(headPos, headDir, camPlaneOrthVector, pointOnPlan)
        tview_points = line_plane_collision(camPlaneOrthVector, pointOnPlan, headDir, headPos)

        print('tview_point', tview_points)

        return tview_points


    def eye_aspect_ratio(self, teye):
        ttext = "Not Detect"
        tstatus = RET_NOT_DETECT
        if(teye[0][0] == 0 or teye[0][1] == 0):
            return tstatus, ttext
        # 눈에 랜드마크 좌표를 찍어서 EAR값을 예측합니다.
        A = calc_dist(teye[1], teye[5])
        B = calc_dist(teye[2], teye[4])
        C = calc_dist(teye[0], teye[3])
        D = calc_dist(teye[1], teye[2])
        E = calc_dist(teye[4], teye[5])

        # print(A,B,C)
        ear = (A + B) / (2.0 * C)
        # ear = 2 * (A + B) - C
        slope = (D+E) / (A+B)
        print('eye_aspect_ratio', ear)
        # print('eye_slope', slope)

        if( ear < EYE_DROWSINESS_THRESH):
        # if (slope > EYE_CLOSE_THRESH):
            ttext = "Close"
            # ttext = "{:.2f}{:.2f}".format(ear,slope)
            tstatus = RET_CLOSE
        else:
            ttext = "Open"
            # ttext = "{:.2f}{:.2f}".format(ear,slope)
            tstatus = RET_OPEN

        return tstatus, ttext, ear

    def mouth_open(self, tmouth_in, tmouth_out):
        ttext = "Not Detect"
        tstatus = RET_NOT_DETECT

        # print("tmouth_in",tmouth_in)
        # print("tmouth_out",tmouth_out)

        if(len(self.faces_point[0]) ==21):
            p51 = tmouth_out[1]
            p62 = tmouth_in[0]
            p66 = tmouth_in[1]
            B = calc_dist(p51, p62)
            E = calc_dist(p66, p62)
            if ((B) < (E)):
                ttext = "Open"
                tstatus = RET_OPEN
            else:
                ttext = "Close"
                tstatus = RET_CLOSE

            return tstatus, ttext

        p50 = tmouth_out[2]
        p51 = tmouth_out[3]
        p52 = tmouth_out[4]
        p61 = tmouth_in[1]
        p62 = tmouth_in[2]
        p63 = tmouth_in[3]
        p67 = tmouth_in[7]
        p66 = tmouth_in[6]
        p65 = tmouth_in[5]
        A = calc_dist(p50, p61)
        B = calc_dist(p51, p62)
        C = calc_dist(p52, p63)
        D = calc_dist(p67, p61)
        E = calc_dist(p66, p62)
        F = calc_dist(p65, p63)
        if((A+B+C) < (D+E+F)):
            ttext = "Open"
            tstatus = RET_OPEN
        else:
            ttext = "Close"
            tstatus = RET_CLOSE

        return tstatus, ttext

    def drowsiness_detect(self,image, ear, pos):
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_DROWSINESS_THRESH:
            self.EyeCloseCounter += 1
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if self.EyeCloseCounter >= EYE_DROWSINESS_REPEAT:
                # if the alarm is not on, turn it on
                # if not ALARM_ON:
                #     ALARM_ON = True
                #     # check to see if an alarm file was supplied,
                #     # and if so, start a thread to have the alarm
                #     # sound played in the background
                #     if args["alarm"] != "":
                #         t = Thread(target=sound_alarm,
                #                    args=(args["alarm"],))
                #         t.deamon = True
                #         t.start()
                # draw an alarm on the frame
                # cv2.putText(image, "!!!!!!  DROWSINESS ALERT  !!!!!!", (10, 60),
                cv2.putText(image, "!!!!!!  DROWSINESS ALERT  !!!!!!", (max(0,pos[0]-200), max(0,pos[1]-110)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            self.EyeCloseCounter = 0
            # ALARM_ON = False

    def analyseFace_from_dlib(self, img, detector, predictor, quality=0, offset=(0, 0)):
        dets = detector(img)
        result = []
        result_other = []

        result_faces_points = []

        for k, d in enumerate(dets):
            instantFacePOI = np.zeros((7, 2), dtype=np.float32)
            eyeCorners = np.zeros((2, 2, 2), dtype=np.float32)

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            result_faces_points.append(shape_to_np(shape,offset = offset).tolist())
            # print("result_faces_points", result_faces_points)

            if (shape.num_parts == 21):  # custom training
                instantFacePOI[C_R_HEAR][0] = shape.part(0).x + offset[0]
                instantFacePOI[C_R_HEAR][1] = shape.part(0).y + offset[1]
                instantFacePOI[C_L_HEAR][0] = shape.part(1).x + offset[0]
                instantFacePOI[C_L_HEAR][1] = shape.part(1).y + offset[1]
                instantFacePOI[C_NOSE][0] = shape.part(2).x + offset[0]
                instantFacePOI[C_NOSE][1] = shape.part(2).y + offset[1]
                instantFacePOI[C_R_MOUTH][0] = shape.part(15).x + offset[0]
                instantFacePOI[C_R_MOUTH][1] = shape.part(15).y + offset[1]
                instantFacePOI[C_L_MOUTH][0] = shape.part(17).x + offset[0]
                instantFacePOI[C_L_MOUTH][1] = shape.part(17).y + offset[1]

                leftEyeX = 0
                leftEyeY = 0
                for i in range(3, 9):
                    if (i == 3 or i == 6):
                        continue
                    leftEyeX += shape.part(i).x
                    leftEyeY += shape.part(i).y
                leftEyeX = int(leftEyeX / 4.0)
                leftEyeY = int(leftEyeY / 4.0)
                eyeCorners[0][0] = [shape.part(3).x + offset[0], shape.part(3).y + offset[1]]
                eyeCorners[0][1] = [shape.part(6).x + offset[0], shape.part(6).y + offset[1]]
                instantFacePOI[C_R_EYE][0] = leftEyeX + offset[0]
                instantFacePOI[C_R_EYE][1] = leftEyeY + offset[1]
                rightEyeX = 0
                rightEyeY = 0
                for i in range(9, 15):
                    if (i == 9 or i == 12):
                        continue
                    rightEyeX += shape.part(i).x
                    rightEyeY += shape.part(i).y
                rightEyeX = int(rightEyeX / 4.0)
                rightEyeY = int(rightEyeY / 4.0)
                eyeCorners[1][0] = [shape.part(9).x + offset[0], shape.part(9).y + offset[1]]
                eyeCorners[1][1] = [shape.part(12).x + offset[0], shape.part(12).y + offset[1]]
                instantFacePOI[C_L_EYE][0] = rightEyeX + offset[0]
                instantFacePOI[C_L_EYE][1] = rightEyeY + offset[1]
                data = [instantFacePOI,
                        (int(d.left() + offset[0]), int(d.top() + offset[1]), int(d.right() + offset[0]),
                         int(d.bottom() + offset[1])), \
                        eyeCorners]
                result.append(data)

                p_lefteye = []
                p_righteye = []
                p_mouse_in = []
                p_mouse_out = []

                p_lefteye.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in LEFT_EYE_MINI])
                p_righteye.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in RIGHT_EYE_MINI])
                p_mouse_out.extend(
                    [[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in MOUTH_OUTLINE_MINI])
                p_mouse_in.extend(
                    [[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in MOUTH_INNER_MINI])

                result_other.append([p_lefteye, p_righteye, p_mouse_in, p_mouse_out])

            else:
                # oreille droite
                instantFacePOI[C_R_HEAR][0] = shape.part(0).x + offset[0]
                instantFacePOI[C_R_HEAR][1] = shape.part(0).y + offset[1]
                # oreille gauche
                instantFacePOI[C_L_HEAR][0] = shape.part(16).x + offset[0]
                instantFacePOI[C_L_HEAR][1] = shape.part(16).y + offset[1]
                # nez
                instantFacePOI[C_NOSE][0] = shape.part(30).x + offset[0]
                instantFacePOI[C_NOSE][1] = shape.part(30).y + offset[1]
                # bouche gauche
                instantFacePOI[C_R_MOUTH][0] = shape.part(48).x + offset[0]
                instantFacePOI[C_R_MOUTH][1] = shape.part(48).y + offset[1]
                # bouche droite
                instantFacePOI[C_L_MOUTH][0] = shape.part(54).x + offset[0]
                instantFacePOI[C_L_MOUTH][1] = shape.part(54).y + offset[1]

                leftEyeX = 0
                leftEyeY = 0
                # for i in range(36, 42):
                #     leftEyeX += shape.part(i).x
                #     leftEyeY += shape.part(i).y
                # leftEyeX = int(leftEyeX / 6.0)
                # leftEyeY = int(leftEyeY / 6.0)
                for i in range(37, 42):
                    if (i == 39):
                        continue
                    leftEyeX += shape.part(i).x
                    leftEyeY += shape.part(i).y
                leftEyeX = int(leftEyeX / 4.0)
                leftEyeY = int(leftEyeY / 4.0)
                eyeCorners[0][0] = [shape.part(36).x + offset[0], shape.part(36).y + offset[1]]
                eyeCorners[0][1] = [shape.part(39).x + offset[0], shape.part(39).y + offset[1]]

                instantFacePOI[C_R_EYE][0] = leftEyeX + offset[0]
                instantFacePOI[C_R_EYE][1] = leftEyeY + offset[1]

                rightEyeX = 0
                rightEyeY = 0
                # for i in range(42, 48):
                #     rightEyeX += shape.part(i).x
                #     rightEyeY += shape.part(i).y
                # rightEyeX = int(rightEyeX / 6.0)
                # rightEyeY = int(rightEyeY / 6.0)
                for i in range(43, 48):
                    if (i == 45):
                        continue
                    rightEyeX += shape.part(i).x
                    rightEyeY += shape.part(i).y
                rightEyeX = int(rightEyeX / 4.0)
                rightEyeY = int(rightEyeY / 4.0)
                eyeCorners[1][0] = [shape.part(42).x + offset[0], shape.part(42).y + offset[1]]
                eyeCorners[1][1] = [shape.part(45).x + offset[0], shape.part(45).y + offset[1]]
                instantFacePOI[C_L_EYE][0] = rightEyeX + offset[0]
                instantFacePOI[C_L_EYE][1] = rightEyeY + offset[1]
                data = [instantFacePOI, (
                    int(d.left() + offset[0]), int(d.top() + offset[1]), int(d.right() + offset[0]),
                    int(d.bottom() + offset[1])),
                        eyeCorners]
                result.append(data)

                p_lefteye = []
                p_righteye = []
                p_mouse_in = []
                p_mouse_out = []

                p_lefteye.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in LEFT_EYE])
                p_righteye.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in RIGHT_EYE])
                p_mouse_out.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in MOUTH_OUTLINE])
                p_mouse_in.extend([[shape.part(t).x + offset[0], shape.part(t).y + offset[1]] for t in MOUTH_INNER])

                result_other.append([p_lefteye, p_righteye, p_mouse_in, p_mouse_out])
                # print('result_other', result_other)

        return result, result_other, result_faces_points


if __name__ == '__main__':

    test2 = [
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]],
[[(982.5, 395.0), [[ 975.,  401.],[1000.,  394.]]],
 [(456.5, 358.5), [[442., 365.], [467., 367.]]]]
]
#
    # print(np.array(test2)[:, 0])
    print(np.array(test2)[:, 0][:, 0][:, 0])
    print(np.array(test2)[:, 0][:, 0][:, 1])
    print(np.array(test2)[:, 1][:, 0][:, 0])
    print(np.array(test2)[:, 1][:, 0][:, 1])

    # print(np.array(test2)[:, 0][:, 1][:, 0][-1])
    # print(np.array(test2)[:, 0][:, 1][:, 0][:, 1])
    # print(np.array(test2)[:, 0][:, 1][:, 1][-1])
    # print(np.array(test2)[:, 1])
    # print(1/0)
    # print(test)
    lpupil_cen_x = []
    lpupil_cen_y = []
    for idx, tdata in enumerate(test2):

        # print(idx, tdata)
        print(test2[idx])
        tpupil_cen_x = []
        tpupil_cen_y = []
        for i, idata in enumerate(test2[idx]):
            print(i, idata)
            print('----',idata[0],'-----',idata[1][0],idata[1][1])
            tpupil_cen_x.append(idata[0][0])
            tpupil_cen_y.append(idata[0][1])
            # tpupil_cen_x.extend(idata[0][0])
            # tpupil_cen_y.extend(idata[0][1])
            print('tpupil_cen_x', tpupil_cen_x)

        lpupil_cen_x.extend([tpupil_cen_x])
        lpupil_cen_y.extend([tpupil_cen_y])
        # for j in i:
        #     print(j)
        # print(i)
        # print('center', i[0],'eye_pos', i[1])
    print('lpupil_cen_x', lpupil_cen_x)
    print('lpupil_cen_y', lpupil_cen_y)

    # print(i)
    for e in range(i+1):
        print('lpupil_cen_x_np_{}_column={}'.format(e, np.array(lpupil_cen_x)[:, e]))

    # print('lpupil_cen_x_np_first_column', np.array(lpupil_cen_x)[:,0])
    # print('lpupil_cen_x_np_second_column', np.array(lpupil_cen_x)[:,1])
    # print('lpupil_cen_y_np', lpupil_cen_y)

    # print('rpupil_cen', rpupil_cen)
    print(1/0)


    distCoeffs = np.zeros((5, 1))
    # HARDCODED CAM PARAMS
    # width = 1280
    # height = 964
    # width = 797
    # height = 1200
    width = 2402
    height = 1201
    maxSize = max(width, height)
    # maxSize = 1470
    cameraMatrix = np.array([[maxSize, 0, width / 2.0], [0, maxSize, height / 2.0], [0, 0, 1]], np.float32)

    # TEST PICTURE
    img = cv2.imread('./sample/face_two_person.png')
    # img = cv2.imread('s_DSC05310.JPG')
    # img = cv2.imread('test2.png')
    # ellipse_eye_black_right_cam.png
    # ellipse_eye_white_left_cam.png
    # s_DSC05310.JPG

    test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # image = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    #
    #
    # gradientX = computeGradient(image)
    # gradientY = np.transpose(computeGradient(np.transpose(image)))
    # gradientMatrix = matrixMagnitude(gradientX, gradientY)
    #
    # gradientThreshold = computeDynamicThreshold(gradientMatrix, kGradientThreshold)
    # # Normalisation
    # for y in range(0, image.shape[0]):  # Iterate through rows
    #     for x in range(0, image.shape[1]):  # Iterate through columns
    #         if (gradientMatrix[y][x] > gradientThreshold):
    #             gradientX[y][x] = gradientX[y][x] / gradientMatrix[y][x]
    #             gradientY[y][x] = gradientY[y][x] / gradientMatrix[y][x]
    #         else:
    #             gradientX[y][x] = 0.0
    #             gradientY[y][x] = 0.0
    #
    # # Invert and blur befor algo
    # weight = cv2.GaussianBlur(image, (kWeightBlurSize, kWeightBlurSize), 0)
    # for y in range(0, weight.shape[0]):  # Iterate through rows
    #     for x in range(0, weight.shape[1]):  # Iterate through columns
    #         weight[y][x] = 255 - weight[y][x]
    #
    # outSum = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)  # create a receiver array
    # # for y in range(0, outSum.shape[0]):  # Iterate through rows
    # #     for x in range(0, outSum.shape[1]):  # Iterate through columns
    # #         if (gradientX[y][x] == 0.0 and gradientY[y][x] == 0.0):
    # #             continue
    # #         testPossibleCentersFormula(x, y, weight, gradientX[y][x], gradientY[y][x], outSum)
    #
    # out = computeGradient(image)
   # # plt.imshow(out)
   # # plt.title('my pictures')
   # # plt.show()

    # laplacian = cv2.Laplacian(image, cv2.CV_8U, ksize=5)
    # sobelx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
    # sobely = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)
    # plt.subplot(3, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 2, 2), plt.imshow(laplacian, cmap='gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 2, 3), plt.imshow(sobelx, cmap='gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 2, 4), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 2, 5), plt.imshow(out, cmap='gray')
    # plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    # plt.subplot(3, 2, 6), plt.imshow(outSum, cmap='gray')
    # plt.title('outSum'), plt.xticks([]), plt.yticks([])
    #
    #
    # plt.show()


    # Model for face detect
    predictor_path = './dlib/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    faces_data, _ , _ = analyseFace(test, detector, predictor)
    print(len(faces_data))
    # print(faces_data)

    eye_centers = []
    eye_centers2 = []
    for index, POI in enumerate(faces_data):
        # print('\nindex', index, '\nPOI', POI)
        eye_corners = POI[2]
        # print(eye_corners.shape)
        eye_center = getEyePos2(eye_corners, test, 0)
        eye_centers.append(eye_center)
        # print('eye_centers', eye_centers)
        cv2.circle(test, (int(eye_center[0][0]), int(eye_center[0][1])), 2, (255, 0, 0), -1)
        cv2.circle(test, (int(eye_center[1][0][0]), int(eye_center[1][0][1])), 2, (0, 0, 255), -1)
        cv2.circle(test, (int(eye_center[1][1][0]), int(eye_center[1][1][1])), 2, (0, 0, 255), -1)

        eye_center = getEyePos2(eye_corners, test, 1)
        eye_centers2.append(eye_center)
        cv2.circle(test, (int(eye_center[0][0]), int(eye_center[0][1])), 2, (255, 0, 0), -1)
        cv2.circle(test, (int(eye_center[1][0][0]), int(eye_center[1][0][1])), 2, (0, 0, 255), -1)
        cv2.circle(test, (int(eye_center[1][1][0]), int(eye_center[1][1][1])), 2, (0, 0, 255), -1)

    print('eye_center',eye_centers)

    # plt.figure(figsize=(16, 16))
    # plt.imshow(test)
    # plt.title('my picture')
    # plt.show()

    for index, POI in enumerate(faces_data):
        # viewPoint = getCoordFromFace(POI[0], eye_centers[index])
        # cv2.line(test, (int(eye_centers[index][0][0]), int(eye_centers[index][0][1])),
        #          (int(eye_centers[index][0][0] - viewPoint[0]), int(eye_centers[index][0][1] - viewPoint[1])), (0, 255, 0),
        #          2, -1)

        viewPoint, nose_vec, leye_vec, xyz_axis, lpupil_vec, lpupil_vec_from_eyeball, ltemp2 = getCoordFromFace(POI[0],
                                                                                                                eye_centers2[
                                                                                                                    index],
                                                                                                                test, cameraMatrix, distCoeffs)
        cv2.line(test, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
                 (int(eye_centers2[index][0][0] + viewPoint[0]), int(eye_centers2[index][0][1] + viewPoint[1])),
                 (255, 255, 0),
                 2, -1)
        cv2.line(test, (int(POI[0][2][0]), int(POI[0][2][1])),
                 (int(nose_vec[0][0][0]), int(nose_vec[0][0][1])), (0, 255, 255), 2, -1)
        # print(eye_centers2)
        # center of eye gaze
        cv2.line(test, (int((eye_centers2[index][1][1][0] + eye_centers2[index][1][0][0]) / 2),
                        int((eye_centers2[index][1][1][1] + eye_centers2[index][1][0][1]) / 2)),
                 (int(leye_vec[0][0][0]), int(leye_vec[0][0][1])), (255, 0, 255), 2, -1)
        # pupil gaze
        cv2.line(test, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
                 (int(lpupil_vec[0][0][0]), int(lpupil_vec[0][0][1])), (100, 0, 255), 2, -1)

        # cv2.line(test,(int(ltemp2[0][0][0]), int(ltemp2[0][0][1])),
        #         (int(ltemp2[2][0][0]), int(ltemp2[2][0][1])), (255, 0, 0), 2, -1)
        cv2.line(test, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
                 (int(ltemp2[2][0][0]), int(ltemp2[2][0][1])), (255, 0, 0), 2, -1)
        cv2.line(test, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
                 (int(eye_centers2[index][0][0] + lpupil_vec_from_eyeball[0]),
                  int(eye_centers2[index][0][1] + lpupil_vec_from_eyeball[1])), (100, 255, 100), 2, -1)

        # draw_xyz_axis(img, np.uint32(nose_end_point2D), np.uint32(imgpts))
        # print(xyz_axis)
        # draw_xyz_axis(test, np.array([[int(POI[0][2][0]), int(POI[0][2][1])]]), np.uint16(xyz_axis))
        draw_xyz_axis(test, np.array([[int(nose_vec[0][0][0]), int(nose_vec[0][0][1])]]), np.uint16(xyz_axis))

        # cv2.line(test,(int(POI[0][2][0]), int(POI[0][2][1])),
        #         (int(reye_vec[0][0][0]), int(reye_vec[0][0][1])), (255, 0, 255), 2, -1)

    # plt.figure(figsize=(16, 16))
    plt.imshow(test)
    plt.title('my picture')
    plt.show()