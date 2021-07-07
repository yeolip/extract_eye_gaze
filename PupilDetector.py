
import dlib
import cv2
import numpy as np
import time
from PupilDetector2 import GradientIntersect

RIGHT_EYE = list(range(36, 42))  # 6
LEFT_EYE = list(range(42, 48))  # 6

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlib/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
time.sleep(1)  # warming up
if not cap.isOpened():
    exit()

ttimecount = 0
FRAME_REPEAT = 30
tfps = 30
available = 0
viewType = 0
loc = (0,0)

gi_l = GradientIntersect()
gi_r = GradientIntersect()
while True:
    if(ttimecount == 0):
        starttime = time.time()
    ttimecount += 1
    cap.retrieve()
    ret, image = cap.read()

    # if(ttimecount%2 == 0):
    #     continue

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not ret:
        break

    # tempWidth = 80
    # available = objEyeTrack.preprocess(img, (160-tempWidth,120-tempWidth,320+tempWidth,240+tempWidth))
    # available = objEyeTrack.preprocess(img, (0,0, 640 , 480))
    # if(available > 0 ):
    #     objEyeTrack.temp_run(image, img, gazeType=viewType )
    #     objEyeTrack.randering(image)

    faces = detector(gray)
    for face in faces:
        p_lefteye = []
        p_righteye = []

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        landmarks = predictor(gray, face)
        tlandmark = shape_to_np(landmarks)
        # for n in range(0, 68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        for (sX, sY) in tlandmark:
            cv2.circle(image, (sX, sY), 1, (255, 0, 0), -1)

        for n in LEFT_EYE:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            p_lefteye.append([x,y])
        for n in RIGHT_EYE:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            p_righteye.append([x,y])

        p_lefteye = np.array(p_lefteye)
        p_righteye = np.array(p_righteye)
        # print(p_lefteye, p_lefteye.shape)
        # p_lefteye.append([face[t] for t in LEFT_EYE])
        # p_righteye.append([face[t] for t in RIGHT_EYE])
        # for tpoint in p_lefteye:
        lx = np.min(p_lefteye, axis=0)[0] -3
        ly = np.min(p_lefteye, axis=0)[1] - 3  # lip_check
        lx2 = np.max(p_lefteye, axis=0)[0] +3
        ly2 = np.max(p_lefteye, axis=0)[1] + 3
        print(lx, ly, lx2, ly2)
        # leye_result = eye_aspect_ratio(tpoint)
        # p_lefteye_local = (p_lefteye - np.array([x, y]))
        # print('crop_lefteye', p_lefteye_local)
        clipping_gray_l = gray[ly:ly2, lx:lx2]
            # if (SAVE_PART_OF_EYES):
            #     cv2.imwrite('eyeL{:02d}.png'.format(tnum), clipping_gray)

        rx = np.min(p_righteye, axis=0)[0] -3
        ry = np.min(p_righteye, axis=0)[1] - 3  # lip_check
        rx2 = np.max(p_righteye, axis=0)[0] +3
        ry2 = np.max(p_righteye, axis=0)[1] + 3
        # print(lx, ly, lx2, ly2)
        clipping_gray_r = gray[ry:ry2, rx:rx2]

        if(available == 0):
            # gi_l = GradientIntersect()
            loc_l = gi_l.locate(clipping_gray_l)
            print('loc_l', loc_l)
            cv2.circle(image, (int(lx + loc_l[1]), int(ly + loc_l[0])), 2, (0, 0, 255), -1)
            available = 0

            # gi_r = GradientIntersect()
            loc_r = gi_r.locate(clipping_gray_r)
            print('loc_r', loc_r)
            cv2.circle(image, (int(rx + loc_r[1]), int(ry + loc_r[0])), 2, (0, 0, 255), -1)

            # cv2.imshow("result", clipping_gray)
            # cv2.waitKey(0)
        # elif(available == 1):
            # loc_l = gi_l.track(clipping_gray_l, loc_l)
            # loc_l = gi_l.track(gray, (int(lx + loc_l[1]), int(ly + loc_l[0])))
            # print('loc2',loc_l)
    # image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.putText(image, 'FPS={:.1f} {:s}'.format(tfps, "      "),
                (10, 460),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)
    if (ttimecount >= FRAME_REPEAT):
        tfps = ttimecount / (time.time() - starttime)
        ttimecount = 0

    cv2.imshow('image', image)



    # gi = GradientIntersect()
    # loc = gi.locate(gray)
    # print(loc)
    #
    # ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image', frame)
    #
    # loc = gi.track(gray, loc)
    # print(loc)



    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()