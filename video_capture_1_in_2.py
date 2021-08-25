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

# cap = cv2.VideoCapture("D:/Project/CVT/발표/result/1_cto_DRCAM_KOR40BU4578_20190219_114431_0002.mp4")
cap = cv2.VideoCapture("D:/Project/CVT/발표/DRCAM_KOR40BU4578_20190219_114431_origin_left.avi")
cap2 = cv2.VideoCapture("D:/Project/CVT/발표/result/3d_gaze_roi_result_13.mp4")
# cap = cv2.VideoCapture("D:/Project/CVT/발표/result/1_cto_DRCAM_KOR40BU4578_20190219_114431_0002.mp4")
# cap2 = VideoCapture("D:/Project/CVT/발표/result/3d_gaze_roi_result_12.mp4")

# cap2.set(cv2.CAP_PROP_FPS, 30)

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# time.sleep(1) #warming up
# if not cap.isOpened():
#   exit()
# if not cap2.isOpened():
#   exit()


tcnt = 1

ttimecount = 0
FRAME_REPEAT = 30
tfps = 30
tfreg = 3

# cap = VideoCapture(0)
# cap.release()
# cap = VideoCapture(0)
# while True:
  # frame = cap.read()
  # time.sleep(.5)   # simulate long processing
  # cv2.imshow("frame", frame)
  # if chr(cv2.waitKey(1)&255) == 'q':
  #   break

def save_mov(filename):
    t_fps = 30
    t_width = 840+840
    t_height = 482+150
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    ## not work -  fourcc = cv2.VideoWriter_fourcc(*'X264')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    tfile = os.path.splitext(filename)

    for num in range(100):
        tfilename = '%s' % (tfile[len(tfile) - 2]) + '%03d' % (num) + tfile[len(tfile) - 1]
        if not os.path.exists(tfilename):
             break
    print("파일을 저장합니다.", tfilename)
    # print(1/0)
    # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    out = cv2.VideoWriter(tfilename, fourcc, t_fps, (t_width, t_height))

    return out

for a in range(28):
    cap.retrieve()
    ret, image = cap.read()

objsave_mov = save_mov('output.avi')

while True:
    if(ttimecount == 0):
        starttime = time.time()
    ttimecount += 1
    cap.retrieve()
    ret, image = cap.read()

    if(ret == False):
        break

    cap2.retrieve()
    ret2, image2 = cap2.read()

    if(ret2 == False):
        break

    # image = cv2.imread('./sample/face_two_person.png')
    # if(ttimecount%2==0):
    #     image = cv2.imread('./out009.png')
    # else:
    #     image = cv2.imread('./out011.png')
    # image = cv2.imread('./out012.png')

    # image = cv2.imread('./sample/distort/distort_01.png')
    # test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)






    if (ttimecount >= FRAME_REPEAT):
        tfps = ttimecount / (time.time() - starttime)
        ttimecount = 0

    # cv2.putText(image, 'origin_mov', #.format(tfps, "      "),
    #             (10, 30),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), thickness=2, lineType=8)
    #
    # cv2.putText(image2, 'virtual_gaze_roi', #.format(tfps, "      "),
    #             (10, 30),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), thickness=2, lineType=8)


    # cv2.imshow('origin', image)
    # cv2.imshow('virtual', image2)
    # print(image.shape, image2.shape)
    img1 = cv2.resize(image, (640+200,482+150))
    img2 = cv2.resize(image2, (640+200,482+150))
    cv2.putText(img1, 'origin_mov', #.format(tfps, "      "),
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)

    cv2.putText(img2, 'virtual_gaze_roi', #.format(tfps, "      "),
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=8)

    # cv2.imshow('origin vs virtual', cv2.hconcat([img1,img2]))
    objsave_mov.write(cv2.hconcat([img1,img2]))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('out{:03d}.png'.format(tcnt), image)
        tcnt+=1
        print("save")


cap.release()
cap2.release()
cv2.destroyAllWindows()

