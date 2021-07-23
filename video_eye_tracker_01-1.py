import cv2
import time
import dlib
import numpy as np
import eye_tracker_05 as eyeTrk

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

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
faceModel3D[0] = [-7.5,   0.,  -8. ]
# LEFTHEAR
faceModel3D[1] = [ 7.5,   0.,  -8. ]
# NOSE
faceModel3D[2] =  [ 0.,  3.5,   2.5]
# RIGHTMOUTH
faceModel3D[3] = [-3.,   6.,   0. ]
# LEFTMOUTH
faceModel3D[4] = [ 3.,   6.,   0. ]
# RIGHTEYE
faceModel3D[5] = [-3.5,  0.,  -1. ]
# LEFTEYE
faceModel3D[6] = [ 3.5,  0.,  -1. ]

predictor_path = "./dlib/eye_predictor.dat"

objEyeTrack = eyeTrk.eyeTracker(predictor_path)
objEyeTrack.initilaize_calib(tcameraMatrix, tdistCoeffs)
# objEyeTrack.initialize_p3dmodel(faceModel3D)

tcnt = 1
# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1("./dlib/mmod_human_face_detector.dat")
# predictor = dlib.shape_predictor("./dlib/shape_predictor_68_face_landmarks.dat")
# predictor = dlib.shape_predictor("./dlib/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("./dlib/type2_19_facemini.dat")

ttimecount = 0
FRAME_REPEAT = 30
tfps = 30
available = 0
viewType = 0

while True:
    if(ttimecount == 0):
        starttime = time.time()
    ttimecount += 1
    cap.retrieve()
    ret, image = cap.read()

    # if(ttimecount%2 == 0):
    #     continue

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not ret:
        break

    # tempWidth = 80
    # available = objEyeTrack.preprocess(img, (160-tempWidth,120-tempWidth,320+tempWidth,240+tempWidth))
    # available = objEyeTrack.preprocess(img, (0,0, 640 , 480))
    # if(available > 0 ):
    #     objEyeTrack.temp_run(image, img, gazeType=viewType )
    #     objEyeTrack.randering(image)

    faces = detector(img, 0)
    for i, face in enumerate(faces):
        # x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        x1, y1, x2, y2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        # landmarks = predictor(img, face)
        landmarks = predictor(img, face.rect)
        tlandmark = shape_to_np(landmarks)
        # for n in range(0, 68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        for (sX, sY) in tlandmark:
            cv2.circle(image, (sX, sY), 1, (255, 0, 0), -1)
    # image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.putText(image, 'FPS={:.1f} {:s}'.format(tfps, "      "),
                (10, 460),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)
    if (ttimecount >= FRAME_REPEAT):
        tfps = ttimecount / (time.time() - starttime)
        ttimecount = 0

    cv2.imshow('image', image)



    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('out{:03d}.png'.format(tcnt), image)
        tcnt+=1
        print("save")
    elif key == ord('1'):
        if(viewType%10==1):
            viewType -= 1
        else:
            viewType += 1
    elif key == ord('2'):
        if(viewType//10%10==1):
            viewType -= 10
        else:
            viewType += 10
    elif key == ord('3'):
        if(viewType//100%10==1):
            viewType -= 100
        else:
            viewType += 100
    elif key == ord('4'):
        if(viewType//1000%10==1):
            viewType -= 1000
        else:
            viewType += 1000
    elif key == ord('w'):
        predictor = dlib.shape_predictor("./dlib/type1_21_facefull.dat")
    elif key == ord('e'):
        predictor = dlib.shape_predictor("./dlib/shape_predictor_68_face_landmarks.dat")
    elif key == ord('r'):
        predictor = dlib.shape_predictor("./dlib/type2_19_facemini.dat")
    elif key == ord('t'):
        predictor = dlib.shape_predictor("./dlib/eye_predictor.dat")
    # time.sleep(0.001)

cap.release()
cv2.destroyAllWindows()

