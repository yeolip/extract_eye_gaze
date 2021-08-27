import queue as queue
import threading
import cv2
import time
import os
# import dlib
import numpy as np
# import eye_tracker_06 as eyeTrk   #detect 68 point

PERPROMANCE_TEST = 0

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        # self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW)
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

cap = cv2.VideoCapture(0)
ffps = cap.get(cv2.CAP_PROP_FPS)
fwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

time.sleep(1) #warming up
if not cap.isOpened():
  exit()

def save_mov(filename, t_fps, t_width, t_height):
    print('framerate', t_fps, 'width', t_width, 'height', t_height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    ## not work -  fourcc = cv2.VideoWriter_fourcc(*'X264')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    tfile = os.path.splitext(filename)

    for num in range(500):
        tfilename = '%s' % (tfile[len(tfile) - 2]) + '%03d' % (num) + tfile[len(tfile) - 1]
        if not os.path.exists(tfilename):
             break
    print("파일을 저장합니다.", tfilename)

    # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    out = cv2.VideoWriter(tfilename, fourcc, t_fps, (t_width, t_height))
    return out


tcnt = 1
ttimecount = 0
FRAME_REPEAT = 30
tfps = 30
tfreg = 3
fenable_record = False
objsave_mov = False #cv2.VideoWriter()
# objsave_mov = save_mov('record_out.avi', int(ffps), int(fwidth), int(fheight))

while True:
    # global objsave_mov
    if(ttimecount == 0):
        starttime = time.time()
    ttimecount += 1
    cap.retrieve()
    ret, image = cap.read()

    if(ret == False):
        break

    if (ttimecount >= FRAME_REPEAT):
        tfps = ttimecount / (time.time() - starttime)
        ttimecount = 0

    # cv2.imshow('origin', image)
    # cv2.imshow('virtual', image2)
    # print(image.shape, image2.shape)
    # img1 = cv2.resize(image, (640+200,482+150))
    # img2 = cv2.resize(image2, (640+200,482+150))
    # cv2.putText(img1, 'origin_mov', #.format(tfps, "      "),
    #             (10, 30),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)

    # cv2.putText(img2, 'virtual_gaze_roi', #.format(tfps, "      "),
    #             (10, 30),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)

    if(fenable_record == True):
        objsave_mov.write(image)
    cv2.putText(image, 'key press', #.format(tfps, "      "),
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=2, lineType=8)
    cv2.putText(image, 's : save', #.format(tfps, "      "),
                (10, 60),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)
    cv2.putText(image, 'r : record', #.format(tfps, "      "),
                (10, 90),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)
    cv2.putText(image, 'q : exit', #.format(tfps, "      "),
                (10, 120),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)

    cv2.imshow('webcam view', image)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        tfile = ""
        for num in range(500):
            tfile = 'out{:03d}.png'.format(num)
            if not os.path.exists(tfile):
                break
        cv2.imwrite(tfile, image)
        # tcnt+=1
        print("Save file:",tfile)
    elif key == ord('r'):
        if(fenable_record == False):
            fenable_record = True
            print("Start recording")
            objsave_mov = save_mov('record_out.avi', int(ffps), int(fwidth), int(fheight))
        elif(fenable_record == True):
            fenable_record = False
            print("End recording")


cap.release()
cv2.destroyAllWindows()

