import cv2
import time
import dlib
import numpy as np
import eye_tracker_05 as eyeTrk

cap = cv2.VideoCapture(0)
time.sleep(2) #warming up
if not cap.isOpened():
  exit()



#set camera param
tdistCoeffs = np.zeros((5, 1))
twidth = 640
theight = 480
tmaxSize = max(twidth, theight)
tcameraMatrix = np.array([[tmaxSize, 0, twidth / 2.0], [0, tmaxSize, theight / 2.0], [0, 0, 1]], np.float32)

#set 3d face model
faceModel3D = np.zeros((7, 3), dtype=np.float32)
# RIGHTHEAR
faceModel3D[0] = [-6.,   0.,  -8. ]
# LEFTHEAR
faceModel3D[1] = [ 6.,   0.,  -8. ]
# NOSE
faceModel3D[2] =  [ 0.,   4.,   2.5]
# RIGHTMOUTH
faceModel3D[3] = [-5.,   8.,   0. ]
# LEFTMOUTH
faceModel3D[4] = [ 5.,   8.,   0. ]
# RIGHTEYE
faceModel3D[5] = [-3.5,  0.,  -1. ]
# LEFTEYE
faceModel3D[6] = [ 3.5,  0.,  -1. ]

objEyeTrack = eyeTrk.eyeTracker()
objEyeTrack.initilaize_calib(tcameraMatrix, tdistCoeffs)
# objEyeTrack.initialize_p3dmodel(faceModel3D)

tcnt = 1
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

while True:
    ret, image = cap.read()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    # print(int(image.size/(image.shape[0]*(image.shape[1]))))
    # print(int(img.size / (img.shape[0] * (img.shape[1]))))
    available = objEyeTrack.preprocess(image)
    if(available > 0 ):
        objEyeTrack.temp_run(image )

    # faces = detector(image)
    # for face in faces:
    #     x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    #     #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    #
    #     landmarks = predictor(image, face)
    #     for n in range(0, 68):
    #         x = landmarks.part(n).x
    #         y = landmarks.part(n).y
    #         cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
    # image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imshow('image', image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('out{:03d}.png'.format(tcnt), image)
        tcnt+=1
        print("save")
    time.sleep(0.001)

cap.release()
cv2.destroyAllWindows()