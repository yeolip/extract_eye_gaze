import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
# %matplotlib inline
import math
import dlib
import cv2
import operator

C_R_HEAR = 0
C_L_HEAR = 1
C_NOSE   = 2
C_R_MOUTH= 3
C_L_MOUTH= 4
C_R_EYE  = 5
C_L_EYE  = 6

degreeToRadian = math.pi/180
radianToDegree = 180/math.pi

kGradientThreshold = 10.0
kWeightBlurSize = 5;
maxEyeSize = 10;

# SOLVER FOR PNP
cameraMatrix = np.eye(3)  # A checker en fct de l'optique choisie
distCoeffs = np.zeros((5, 1))
eyeConst = 1.5

# IMAGE POI FOR 7 POINT
FacePOI = np.zeros((7, 2), dtype=np.float32)
ThreeDFacePOI = np.zeros((7, 3), dtype=np.float32)
# RIGHTHEAR
ThreeDFacePOI[0, 0] = -6
ThreeDFacePOI[0, 1] = 0
ThreeDFacePOI[0, 2] = -8
# LEFTHEAR
ThreeDFacePOI[1, 0] = 6
ThreeDFacePOI[1, 1] = 0
ThreeDFacePOI[1, 2] = -8
# NOSE
ThreeDFacePOI[2, 0] = 0
ThreeDFacePOI[2, 1] = -4
ThreeDFacePOI[2, 2] = 2.5
# RIGHTMOUTH
ThreeDFacePOI[3, 0] = -5
ThreeDFacePOI[3, 1] = -8
ThreeDFacePOI[3, 2] = 0
# LEFTMOUTH
ThreeDFacePOI[4, 0] = 5
ThreeDFacePOI[4, 1] = -8
ThreeDFacePOI[4, 2] = 0
# RIGHTEYE
ThreeDFacePOI[5, 0] = -3
ThreeDFacePOI[5, 1] = 0
ThreeDFacePOI[5, 2] = -1
# LEFTEYE
ThreeDFacePOI[6, 0] = 3
ThreeDFacePOI[6, 1] = 0
ThreeDFacePOI[6, 2] = -1

ThreeDFacePOI2 = np.zeros((7, 3), dtype=np.float32)
# RIGHTHEAR
ThreeDFacePOI2[0, 0] = -6
ThreeDFacePOI2[0, 1] = 0
ThreeDFacePOI2[0, 2] = -8
# LEFTHEAR
ThreeDFacePOI2[1, 0] = 6
ThreeDFacePOI2[1, 1] = 0
ThreeDFacePOI2[1, 2] = -8
# NOSE
ThreeDFacePOI2[2, 0] = 0
ThreeDFacePOI2[2, 1] = 4
ThreeDFacePOI2[2, 2] = 2.5
# RIGHTMOUTH
ThreeDFacePOI2[3, 0] = -5
ThreeDFacePOI2[3, 1] = 8
ThreeDFacePOI2[3, 2] = 0
# LEFTMOUTH
ThreeDFacePOI2[4, 0] = 5
ThreeDFacePOI2[4, 1] = 8
ThreeDFacePOI2[4, 2] = 0
# RIGHTEYE
ThreeDFacePOI2[5, 0] = -3.13
ThreeDFacePOI2[5, 1] = 0
ThreeDFacePOI2[5, 2] = -1
# LEFTEYE
ThreeDFacePOI2[6, 0] = 3.13
ThreeDFacePOI2[6, 1] = 0
ThreeDFacePOI2[6, 2] = -1



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

def analyseFace(img, detector, predictor, quality=1, offset=(0, 0)):
    dets = detector(np.array(img), quality)
    result = []
    for k, d in enumerate(dets):
        instantFacePOI = np.zeros((7, 2), dtype=np.float32)
        eyeCorners = np.zeros((2, 2, 2), dtype=np.float32)
        # Get the landmarks/parts for the face in box d.
        shape = predictor(np.array(img), d)
        # oreille droite
        instantFacePOI[0][0] = shape.part(0).x + offset[0];
        instantFacePOI[0][1] = shape.part(0).y + offset[1];
        # oreille gauche
        instantFacePOI[1][0] = shape.part(16).x + offset[0];
        instantFacePOI[1][1] = shape.part(16).y + offset[1];
        # nez
        instantFacePOI[2][0] = shape.part(30).x + offset[0];
        instantFacePOI[2][1] = shape.part(30).y + offset[1];
        # bouche gauche
        instantFacePOI[3][0] = shape.part(48).x + offset[0];
        instantFacePOI[3][1] = shape.part(48).y + offset[1];
        # bouche droite
        instantFacePOI[4][0] = shape.part(54).x + offset[0];
        instantFacePOI[4][1] = shape.part(54).y + offset[1];

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

        instantFacePOI[5][0] = leftEyeX + offset[0];
        instantFacePOI[5][1] = leftEyeY + offset[1];

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
        instantFacePOI[6][0] = rightEyeX + offset[0];
        instantFacePOI[6][1] = rightEyeY + offset[1];
        data = [instantFacePOI, (
        int(d.left() + offset[0]), int(d.top() + offset[1]), int(d.right() + offset[0]), int(d.bottom() + offset[1])),
                eyeCorners]
        result.append(data)
    return result


def computeGradient(img):
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  # create a receiver array
    if img.shape[0] < 2 or img.shape[1] < 2:  # TODO I'm not sure that secure out of range
        print("EYES too small")
        return out
    for y in range(0, out.shape[0]):
        out[y][0] = img[y][1] - img[y][0]
        for x in range(1, out.shape[1] - 1):
            out[y][x] = (img[y][x + 1] - img[y][x - 1]) / 2.0
        out[y][out.shape[1] - 1] = img[y][out.shape[1] - 1] - img[y][out.shape[1] - 2]
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
        eyeImg = eyeImage.copy()
    eyeImg = eyeImg.astype(np.float32)
    scaleValue = 1.0;
    if (eyeImg.shape[0] > maxEyeSize or eyeImg.shape[1] > maxEyeSize):
        scaleValue = max(maxEyeSize / float(eyeImg.shape[0]), maxEyeSize / float(eyeImg.shape[1]))
        eyeImg = cv2.resize(eyeImg, None, fx=scaleValue, fy=scaleValue, interpolation=cv2.INTER_AREA)

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
    thetay = -np.arcsin(eyeOffset[0]) * eyeConst
    thetax = np.arcsin(eyeOffset[1]) * eyeConst
    print('각도', thetax*radianToDegree, thetay*radianToDegree)
    # Aaand the rotation matrix
    rot = eulerAnglesToRotationMatrix([thetax, thetay, 0])
    # print rot
    return rot

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
    def __init__(self):
        twidth = 640
        theight = 480
        tmaxSize = max(twidth, theight)
        tK = np.array([[tmaxSize, 0, twidth / 2.0], [0, tmaxSize, theight / 2.0], [0, 0, 1]], np.float32)
        tD = np.zeros((5,1))
        self.initilaize_calib(tK, tD)
        self.initialize_p3dmodel(ThreeDFacePOI2)
        predictor_path = './shape_predictor_68_face_landmarks.dat'

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        pass

    def initilaize_calib(self, tCameraMatrix, tDistCoeffs):
        # cameraMatrix = np.eye(3)  # A checker en fct de l'optique choisie
        # distCoeffs = np.zeros((5, 1))
        self.cameraMatrix = tCameraMatrix
        self.distCoeffs = tDistCoeffs
        pass

    def initialize_p3dmodel(self, paramPOI):
        # # RIGHTHEAR
        # ThreeDFacePOI[0, 0] = -6
        # ThreeDFacePOI[0, 1] = 0
        # ThreeDFacePOI[0, 2] = -8
        # # LEFTHEAR
        # ThreeDFacePOI[1, 0] = 6
        # ThreeDFacePOI[1, 1] = 0
        # ThreeDFacePOI[1, 2] = -8
        # # NOSE
        # ThreeDFacePOI[2, 0] = 0
        # ThreeDFacePOI[2, 1] = -4
        # ThreeDFacePOI[2, 2] = 2.5
        # # RIGHTMOUTH
        # ThreeDFacePOI[3, 0] = -5
        # ThreeDFacePOI[3, 1] = -8
        # ThreeDFacePOI[3, 2] = 0
        # # LEFTMOUTH
        # ThreeDFacePOI[4, 0] = 5
        # ThreeDFacePOI[4, 1] = -8
        # ThreeDFacePOI[4, 2] = 0
        # # RIGHTEYE
        # ThreeDFacePOI[5, 0] = -3
        # ThreeDFacePOI[5, 1] = 0
        # ThreeDFacePOI[5, 2] = -1
        # # LEFTEYE
        # ThreeDFacePOI[6, 0] = 3
        # ThreeDFacePOI[6, 1] = 0
        # ThreeDFacePOI[6, 2] = -1

        self.ref_p3dmodel = paramPOI

    def temp_run(self, image):

        eye_centers = []
        eye_centers2 = []
        for index, POI in enumerate(self.faces_data):
            # print('\nindex', index, '\nPOI', POI)
            eye_corners = POI[2]
            # print(eye_corners.shape)
            eye_center = getEyePos2(eye_corners, image, 0)
            eye_centers.append(eye_center)
            # print('eye_centers', eye_centers)
            cv2.circle(image, (int(eye_center[0][0]), int(eye_center[0][1])), 2, (255, 0, 0), -1)
            cv2.circle(image, (int(eye_center[1][0][0]), int(eye_center[1][0][1])), 2, (0, 0, 255), -1)
            cv2.circle(image, (int(eye_center[1][1][0]), int(eye_center[1][1][1])), 2, (0, 0, 255), -1)

            eye_center = getEyePos2(eye_corners, image, 1)
            eye_centers2.append(eye_center)
            cv2.circle(image, (int(eye_center[0][0]), int(eye_center[0][1])), 2, (255, 0, 0), -1)
            cv2.circle(image, (int(eye_center[1][0][0]), int(eye_center[1][0][1])), 2, (0, 0, 255), -1)
            cv2.circle(image, (int(eye_center[1][1][0]), int(eye_center[1][1][1])), 2, (0, 0, 255), -1)


        for index, POI in enumerate(self.faces_data):

            tR, tT, eulerAngle_degree = self.getWorldCoordFromFace(self.ref_p3dmodel, POI[0], self.cameraMatrix, self.distCoeffs)

            cv2.putText(image,'pitch {:.02f}, yaw {:.02f}, roll {:.02f}'.format(eulerAngle_degree[0],eulerAngle_degree[1],eulerAngle_degree[2]),
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=4)

            tlandmark_2d = self.getLandmark(self.ref_p3dmodel[C_NOSE], tR, tT)
            draw_xyz_axis(image, tlandmark_2d[0], round(tlandmark_2d[1:4,-1]))
            # print('tlandmark_2d',  tlandmark_2d[0][0], np.uint16(tlandmark_2d[1:4,-1]))

            eyeballgaze = self.getEyeballCenterGaze(self.ref_p3dmodel[C_R_EYE], tR, tT)
            print('eyeballgaze', eyeballgaze)
            # eyeball gaze
            # cv2.line(image, (int((eye_centers2[index][1][1][0] + eye_centers2[index][1][0][0]) / 2),
            #                 int((eye_centers2[index][1][1][1] + eye_centers2[index][1][0][1]) / 2)),
            #          (int(eyeballgaze[0][0][0]), int(eyeballgaze[0][0][1])), (255, 0, 255), 2, -1)
            cv2.line(image, (int((eye_centers2[index][1][1][0] + eye_centers2[index][1][0][0]) / 2),
                            int((eye_centers2[index][1][1][1] + eye_centers2[index][1][0][1]) / 2)),
                     (int(eyeballgaze[0][0][0]), int(eyeballgaze[0][0][1])), (255, 0, 255), 2, -1)

            tview_2d = self.getEyeGaze_method_one(eye_centers2[index], tR, tT)
            print('tview_2d', tview_2d)

            # viewPoint, nose_vec, leye_vec, xyz_axis, lpupil_vec, lpupil_vec_from_eyeball, ltemp2 = getCoordFromFace(
            #     POI[0], eye_centers2[index], image, self.cameraMatrix, self.distCoeffs)
            # cv2.line(image, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
            #          (int(eye_centers2[index][0][0] + viewPoint[0]), int(eye_centers2[index][0][1] + viewPoint[1])),
            #          (255, 255, 0), 2, -1)
            # # cv2.line(image, (int(POI[0][2][0]), int(POI[0][2][1])),
            # #          (int(nose_vec[0][0][0]), int(nose_vec[0][0][1])), (0, 255, 255), 2, -1)
            # #

            # # # pupil gaze
            # # cv2.line(image, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
            # #          (int(lpupil_vec[0][0][0]), int(lpupil_vec[0][0][1])), (100, 0, 255), 2, -1)
            # #
            # # # cv2.line(image,(int(ltemp2[0][0][0]), int(ltemp2[0][0][1])),
            # # #         (int(ltemp2[2][0][0]), int(ltemp2[2][0][1])), (255, 0, 0), 2, -1)
            # cv2.line(image, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
            #          (int(ltemp2[2][0][0]), int(ltemp2[2][0][1])), (255, 0, 0), 2, -1)
            # # cv2.line(image, (int(eye_centers2[index][0][0]), int(eye_centers2[index][0][1])),
            # #          (int(eye_centers2[index][0][0] + lpupil_vec_from_eyeball[0]),
            # #           int(eye_centers2[index][0][1] + lpupil_vec_from_eyeball[1])), (100, 255, 100), 2, -1)
            #
            # # draw_xyz_axis(img, np.uint32(nose_end_point2D), np.uint32(imgpts))
            # # print(xyz_axis)
            # # draw_xyz_axis(image, np.array([[int(POI[0][2][0]), int(POI[0][2][1])]]), np.uint16(xyz_axis))


    def preprocess(self, image):
        self.faces_data = analyseFace(image, self.detector, self.predictor)
        print("# of detected : {:d} person".format(len(self.faces_data)))
        return len(self.faces_data)

    def randering(self):
        pass

    # Given the data from a faceExtract
    def getWorldCoordFromFace(self, ref_point, image_point, cameraMatrix, distCoeffs):
        print("\n//////////////getCoordFromFace")
        # SOLVER FOR PNPs
        retval, rvec, tvec = cv2.solvePnP(ref_point, image_point, cameraMatrix, distCoeffs);
        # print('retval', retval)
        # rvec[0] = rvec[0]+3.14/10 # roll임 - world coordinate가 y가 위로 +일경우
        # rvec[1] = rvec[1]+3.14/10 # yaw임 (얼굴이 왼쪽으로 +a , 얼굴이 오른쪽으로 -a)
        # rvec[2] = rvec[2]+3.14/10 # pitch임 (얼굴이 위쪽으로 +a , 얼굴이 아래쪽으로 -a)

        #ref_point의 축이 변경되면, pitch yaw roll이 변경될수 있음
        pitch = math.degrees(math.asin(math.sin(rvec[0])))
        yaw = math.degrees(math.asin(math.sin(rvec[1])))
        roll = -math.degrees(math.asin(math.sin(rvec[2])))

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

        origin = [tvec[0][0], tvec[1][0], tvec[2][0]]
        headDir = np.dot(rot2, np.dot(rt, [0, 0, 1]))
        camPlaneOrthVector = [0, 0, 1]
        pointOnPlan = [0, 0, 0]

        tview_points = intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
        print('tview_point', tview_points)

        return tview_points

    def getEyeballCenterGaze(self, ref_point, rvec, tvec):
        temp_point = ref_point.copy()
        temp_point[2] = 25

        print(temp_point)
        eyeballCenterGaze, jac = cv2.projectPoints(ref_point, rvec, tvec, cameraMatrix, distCoeffs)
        # print('eyeballCenterGaze', eyeballCenterGaze)
        return eyeballCenterGaze

        # nose_end_point2D, jacobian = cv2.projectPoints(np.array([ThreeDFacePOI2[2]]),rvec, tvec, cameraMatrix, distCoeffs)
        # # print(nose_end_point2D)
        #
        # leye_end_point2D, jacobian = cv2.projectPoints(np.array([(3.0, 0.0, 25.0)]), rvec, tvec, cameraMatrix, distCoeffs)
        # # print(reye_end_point2D)
        #
        #
        # # pose estimation
        # # self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        # retGap, retGapCoord = sub_eyecenter_and_pupilcenter(FacePOI, eyeData[0], cameraMatrix, proj_matrix)
        # print(np.array([(3.0, 0.0, 25.0)] + retGap.T))
        # rvec_pupil = cv2.Rodrigues(np.dot(rot2,rt))[0]
        # lpupil_end_point2D, jacobian = cv2.projectPoints(np.array([(3.0, 0.0, 25.0)]), rvec_pupil, tvec+retGap, cameraMatrix, distCoeffs)
        #
        # K = 1.31    #distance between eyeball center and pupil center
        # K0 = 0.53    #cornea radius
        #
        # tretGap = np.array(retGap)
        # # tretGap[2] = 0
        # tretGapCoord = np.array(retGapCoord.T[1])[0][0:3]
        # tretGapCoord[2] = 0
        # print('retGapCoord', np.array(retGapCoord.T[1])[0][0:3])
        #
        # taa = np.array([ThreeDFacePOI2[6]]) - np.array([(0, 0, K)])
        # tbb = np.array([ThreeDFacePOI2[6]]) + tretGap.T
        # # tbb = np.array([ThreeDFacePOI2[6]]) + tretGapCoord
        #
        # tcc = np.array([ThreeDFacePOI2[6]])
        # temp2 = cv2.Rodrigues(tcc - taa)[0]
        # print('temp2',temp2, tcc - taa)
        # print('taa',taa)
        # print('tbb',tbb)
        # print('tbb-taa',(tbb - taa))
        # print('tcc-taa',(tcc - taa))
        # aaa = np.array(tbb - taa)
        # bbb = np.array(tcc - taa)
        # calc_ang = np.dot(aaa[0],bbb[0])
        # print(calc_ang)
        # calc_ang2 = np.sqrt(aaa[0][0]*aaa[0][0]+aaa[0][1]*aaa[0][1]+aaa[0][2]*aaa[0][2]) * np.sqrt(bbb[0][0]*bbb[0][0]+bbb[0][1]*bbb[0][1]+bbb[0][2]*bbb[0][2])
        # tang = np.arccos(calc_ang/ calc_ang2)
        # print('tang',tang*radianToDegree)
        # temp  = cv2.Rodrigues(tbb - taa)[0]
        # print(temp)
        # print("norm", cv2.norm(tbb - taa))
        # xx = np.arccos((tbb - taa)/cv2.norm(tbb - taa))
        # print('x ang', xx * radianToDegree)
        # temp3d = np.array([taa, tbb, (tbb - taa) *10+ taa])
        # print(temp3d)
        #
        # tangle = math.atan2(cv2.norm(np.cross(aaa, bbb)), np.dot(aaa, bbb.T))
        # print(tangle*radianToDegree)
        #
        # leyeball_to_pupil_point2D2, jacobian = cv2.projectPoints(temp3d, rvec, tvec,
        #                                                  cameraMatrix, distCoeffs)
        # print('leyeball_to_pupil_point2D2',leyeball_to_pupil_point2D2)
        # leyeball_to_pupil_point2D = intersectionWithPlan(origin, np.dot(cv2.Rodrigues(xx)[0], np.dot(rt, [0, 0, 1])), camPlaneOrthVector, pointOnPlan)
        # print('leyeball_to_pupil_point2D',leyeball_to_pupil_point2D)
        #
        # # tview_point = intersectionWithPlan(origin , headDir, camPlaneOrthVector, pointOnPlan)
        # # print('tview_point',np.array(tview_point).ravel())
        #
        # return tview_point, nose_end_point2D, leye_end_point2D, imgpts, lpupil_end_point2D, leyeball_to_pupil_point2D, leyeball_to_pupil_point2D2

        # pass



if __name__ == '__main__':
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
    img = cv2.imread('ellipse_eye_white_left_cam.png')
    # img = cv2.imread('s_DSC05310.JPG')
    # img = cv2.imread('test2.png')
    # ellipse_eye_black_right_cam.png
    # ellipse_eye_white_left_cam.png
    # s_DSC05310.JPG

    test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(test.shape[2])
    # plt.imshow(test)
    # plt.title('my picture')
    # plt.show()

    # Model for face detect
    # predictor_path =  'faceLandmarkModel.dat'
    predictor_path = './shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    faces_data = analyseFace(test, detector, predictor)
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