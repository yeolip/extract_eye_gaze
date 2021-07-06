import queue as queue
import threading
import cv2
import time
import dlib
import numpy as np
import eye_tracker_06 as eyeTrk   #detect 68 point

PERPROMANCE_TEST = 0

def timelap_check(title, start):
    if(PERPROMANCE_TEST == 1):
        print('\tTimeLap - {:s} {:.6f}'.format(title, time.time() - start))

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW)
        # self.cap.open(name, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            exit()
        self.q = queue.Queue(maxsize=1)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
          ret, frame = self.cap.read()
          if not ret:
            break
          if not self.q.empty():
            try:
              self.q.get_nowait()   # discard previous (unprocessed) frame
            except Queue.Empty:
              pass
          self.q.put(frame)

    def read(self):
        return True, self.q.get()

    def retrieve(self):
        return self.cap.retrieve()

    def release(self):
        return self.cap.release()

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# time.sleep(1) #warming up
# if not cap.isOpened():
#   exit()



#set camera param
tdistCoeffs = np.zeros((5, 1))
twidth = 640
theight = 480
tmaxSize = max(twidth, theight)
tcameraMatrix = np.array([[tmaxSize, 0, twidth / 2.0], [0, tmaxSize, theight / 2.0], [0, 0, 1]], np.float32)


def select_camera_calib(num):
    tempDistCoeffs = np.zeros((5, 1))
    tempCameraMatrix = np.eye(3)

    if (num == 1):
        tempDistCoeffs[0][0] = -0.015136023194323986
        tempDistCoeffs[1][0] = 0.2177351340933552
        tempDistCoeffs[2][0] = 0.0025235154109215703
        tempDistCoeffs[3][0] = 0.0022730661434222452
        tempDistCoeffs[4][0] = -0.7167190677252845
        tempCameraMatrix[0][0] = 641.8333531354511
        tempCameraMatrix[1][1] = 641.8333531354511
        tempCameraMatrix[0][2] = 309.4109382434117
        tempCameraMatrix[1][2] = 258.82858265848694
        # print(tempDistCoeffs, tempCameraMatrix)
    return tempCameraMatrix, tempDistCoeffs
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

# predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"
# predictor_path = "./dlib/type2mini.dat"
# predictor_path = "./dlib/eye_predictor.dat"
predictor_path = "./dlib/type1_21_facefull.dat"

objEyeTrack = eyeTrk.eyeTracker(predictor_path)
tcameraMatrix, tdistCoeffs = select_camera_calib(1)
objEyeTrack.initilaize_calib(tcameraMatrix, tdistCoeffs)
# objEyeTrack.initilaize_training_path(predictor_path)
# objEyeTrack.initialize_p3dmodel(faceModel3D)

tcnt = 1
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
# predictor = dlib.shape_predictor("./dlib/type2mini.dat")

ttimecount = 0
FRAME_REPEAT = 30
tfps = 30
available = 0
viewType = 0

cap = VideoCapture(0)
# cap.release()
# cap = VideoCapture(0)
# while True:
  # frame = cap.read()
  # time.sleep(.5)   # simulate long processing
  # cv2.imshow("frame", frame)
  # if chr(cv2.waitKey(1)&255) == 'q':
  #   break

while True:
    if(ttimecount == 0):
        starttime = time.time()
    ttimecount += 1
    cap.retrieve()
    ret, image = cap.read()
    # image = cap.read()

    # if(ttimecount%2 == 0):
    #     continue
    # if ((ttimecount % 4 == 0) or (ttimecount % 4 == 1) or (ttimecount % 4 == 2)):
    #     continue

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # if not ret:
    #     print("Can't read frame")
    #     break
    # print('ret', ret)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    time_s = time.time()
    tempWidth = 120
    available = objEyeTrack.preprocess(img, (160-tempWidth,120-tempWidth,400+tempWidth,340+tempWidth))
    # available = objEyeTrack.preprocess(img, (0,0, 640 , 480))
    timelap_check('1.detect face ', time_s)

    if(available > 0 ):
        time_s = time.time()
        objEyeTrack.algo_run(img, tSelect=viewType )
        timelap_check('2.calc eye gaze ', time_s)

        time_s = time.time()
        objEyeTrack.rendering(image, tSelect=viewType )
        timelap_check('3.rendering ', time_s)


    # faces = detector(img)
    # for face in faces:
    #     x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    #
    #     landmarks = predictor(img, face)
    #     tlandmark = shape_to_np(landmarks)
    #     # for n in range(0, 68):
    #     #     x = landmarks.part(n).x
    #     #     y = landmarks.part(n).y
    #     #     cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
    #     for (sX, sY) in tlandmark:
    #         cv2.circle(image, (sX, sY), 1, (255, 0, 0), -1)
    # # image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if (ttimecount >= FRAME_REPEAT):
        tfps = ttimecount / (time.time() - starttime)
        ttimecount = 0

    cv2.putText(image, 'FPS={:.1f} {:s}'.format(tfps, "      "),
                (10, 460),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)


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
    elif key == ord('0'):
        if(viewType//10000%10==1):
            viewType -= 10000
        else:
            viewType += 10000
    elif key == ord('w'):
        objEyeTrack.initilaize_training_path("./dlib/type1_21_facefull.dat")
    elif key == ord('e'):
        objEyeTrack.initilaize_training_path("./dlib/shape_predictor_68_face_landmarks.dat")
    # time.sleep(0.001)

cap.release()
cv2.destroyAllWindows()

