# face_two_person.png
# ellipse_eye_black_right_cam.png
#Image0_28377_5718597709.png
# mouse_open.png

# https://roadcom.tistory.com/30
#dependency opencv-python 3.2
#python version 3.6
from collections import OrderedDict
import cv2
import os,re,sys
import numpy as np
import math
import dlib
import matplotlib.pyplot as plt

SAVE_PART_OF_EYES = 1

EYE_CLOSE_THRESH = 0.15
EYE_CLOSE_REPEAT = 15
#초당 30 frame일경우 15 frame까지 눈이 작게 뜨는 경우

# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))  # 5
LEFT_EYEBROW = list(range(22, 27))  # 5
RIGHT_EYE = list(range(36, 42))  # 6
LEFT_EYE = list(range(42, 48))  # 6
NOSE = list(range(27, 36))  # 9
MOUTH_OUTLINE = list(range(48, 60))
MOUTH_INNER = list(range(60, 68))  # 나는 60번이 INNER로 보임
JAWLINE = list(range(0, 17))

def calc_dist(p1, p2):
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return distance

def mouth_open(tmouth_in, tmouth_out):
    print("tmouth_in",tmouth_in)
    print("tmouth_out",tmouth_out)
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
        print("Mouth Open")
        return True
    else:
        print("Mouth Close")
        return False


def eye_aspect_ratio(teye):
    # 눈에 랜드마크 좌표를 찍어서 EAR값을 예측합니다.
    A = calc_dist(teye[1], teye[5])
    B = calc_dist(teye[2], teye[4])
    C = calc_dist(teye[0], teye[3])

    print(A,B,C)
    ear = (A + B) / (2.0 * C)
    print('eye_aspect_ratio', ear)
    return ear

def divide_by_face_comfornent(num_of_person, pface):
    p_lefteye = []
    p_righteye = []
    p_nose = []
    p_mouse_in = []
    p_mouse_out = []
    p_face_landmark = []

    print(pface)
    for i, one_face in enumerate(pface):
        print('person ',i)
        # print([ one_face[t] for t in RIGHT_EYE ])
        # print([ one_face[t] for t in LEFT_EYE ])
        # print([ one_face[t] for t in NOSE ])
        # print([ one_face[t] for t in MOUTH_OUTLINE ])
        # print([ one_face[t] for t in MOUTH_INNER ])
        p_lefteye.append([ one_face[t] for t in LEFT_EYE ])
        p_righteye.append([ one_face[t] for t in RIGHT_EYE ])
        p_nose.append([ one_face[t] for t in NOSE ])
        p_mouse_out.append([ one_face[t] for t in MOUTH_OUTLINE ])
        p_mouse_in.append([one_face[t] for t in MOUTH_INNER])
        p_face_landmark.append(list(np.mean([one_face[t] for t in EYE_4_POINT], axis=0)))

    print('p_lefteye', p_lefteye)
    print('p_righteye', p_righteye)
    print('p_nose', p_nose)
    print('p_mouse_in', p_mouse_in)
    print('p_mouse_out', p_mouse_out)
    print('p_face_landmark', p_face_landmark)

    return p_lefteye, p_righteye, p_nose, p_mouse_in, p_mouse_out, p_face_landmark


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.20):
    facial_features_cordinates = {}

    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts
        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # return the output image
    print(facial_features_cordinates)
    return output


def detect_face_eye_using_dlib(gray):
    # create face detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./dlib/shape_predictor_68_face_landmarks.dat')

    face_list = []
    eye_list=[]
    # Get faces (up-sampling=1)
    face_detector = detector(gray, 1)
    # the number of face detected
    print("The number of faces detected : {}".format(len(face_detector)))
    # if(faces is not ()):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    face_list = []
    eye_list=[]
    for face in face_detector:
        # face wrapped with rectangle
        print(face)
        cv2.rectangle(img2, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 0, 255), 3)

        # make prediction and transform to numpy array
        landmarks = predictor(img, face)  # 얼굴에서 68개 점 찾기

        landmark_list = np.array([[landmarks.part(t).x, landmarks.part(t).y] for t in ALL]).reshape(-1,2)
        face_list.append([[landmarks.part(t).x, landmarks.part(t).y] for t in ALL])
        # shape = np.array(shape).reshape(-1,2)
        print('landmark_list', landmark_list)
        img = visualize_facial_landmarks(img, landmark_list)
        cv2.imshow("Image", img)

        for i, (px, py) in enumerate(landmark_list):
            cv2.circle(img2, (px, py), 2, (0, 255, 0), -1)
            cv2.putText(img2, "{:02d}".format(i), (px, py), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        cv2.rectangle(img2, (np.min(landmark_list, axis=0)[0], np.min(landmark_list, axis=0)[1]), (np.max(landmark_list, axis=0)[0], np.max(landmark_list, axis=0)[1]),
                      (255, 0, 0), 3)


    cv2.imshow('result', img2)
    print("landmark_list", landmark_list)
    print("face_list", face_list)
    # cv2.waitKey(0)
    return len(face_detector), face_list



def detect_face_eye_using_opencv(gray):
    face_list = []
    eye_list=[]
    face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haar/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print('faces', type(faces))
    if(faces is not ()):
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        for j,(x,y,w,h) in enumerate(faces):
            print('face_{}[{},{},{},{}]'.format(j, x, y, x+w, y+h))
            face_list.append([x, y, x+w, y+h])
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1)
            # print(len(eyes))
            for i,(ex,ey,ew,eh) in enumerate(eyes):
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                print('  eye_{}[{},{},{},{}]'.format(i,x+ex,y+ey,x+ex+ew,y+ey+eh))
                eye_list.append([x+ex,y+ey,x+ex+ew,y+ey+eh])
        cv2.imshow("face", img)
        cv2.waitKey(0)
    else:
        print('Face None')
    return len(face_list), face_list, eye_list

def fit_circle(data):
    #data [[x1,y],[x2,y2]...]
    xs = data[:,0].reshape(-1,1)
    ys = data[:,1].reshape(-1,1)

    J = np.mat(np.hstack((-2 * xs,-2 * ys,np.ones_like(xs,dtype=np.float))))
    Y = np.mat(-xs ** 2 - ys ** 2)
    X = (J.T * J).I * J.T * Y

    cx = X[0,0]
    cy = X[1,0]
    c = X[2,0]
    r = np.sqrt(cx ** 2 + cy ** 2 - c)
    return (cx,cy,r)

def fit_rotated_ellipse(data):
    xs = data[:,0].reshape(-1,1)
    ys = data[:,1].reshape(-1,1)

    J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
    Y = np.mat(-1*xs**2)
    P= (J.T * J).I * J.T * Y

    a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
    # To do implementation
    #a,b,c,d,e,f 를 통해 theta, 중심(cx,cy) , 장축(major), 단축(minor) 등을 뽑아 낼 수 있어요

    theta = 0.5* np.arctan(b/(a-c))
    # if(theta<0):
    #     theta = math.pi + theta
    cx = (2*c*d - b*e)/(b**2-4*a*c)
    cy = (2*a*e - b*d)/(b**2-4*a*c)
    cu = a*cx**2 + b*cx*cy + c*cy**2 -f
    w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
    h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))
    print("a",a, "b",b,'c',c,'d',d,'e',e,'f',f)
    print("theta",theta, "cx",cx,'cy',cy,'cu',cu,'w',w,'h',h)

    return (cx,cy,w,h,theta)

def fit_rotated_ellipse_svd(data, th):
    # data = np.array([
    #     [43, 11],
    #     [ 44, 10],
    #     [44, 11],
    #     [44, 18],
    #     [45, 9 ],
    #     [45, 10],
    #     [45, 18],
    #     [45, 19],
    #     [46, 10],
    #     [46, 19],
    #     [47, 10],
    #     [47, 19],
    #     [48, 10],
    #     [48, 19],
    #     [49, 10],
    #     [49, 11],
    #     [49, 18],
    #     [49, 19],
    #     [50, 11],
    #     [50, 12],
    #     [50, 17],
    #     [50, 18],
    #     [51, 13],
    #     [51, 14],
    #     [51, 15],
    #     [51, 16],
    #     [51, 17]])

    # data = np.array([
    #          [32, 5 ],
    #          [32, 6 ],
    #          [32, 12],
    #          [32, 13],
    #          [33, 4 ],
    #          [33, 5 ],
    #          [33, 13],
    #          [34, 5 ],
    #          [34, 13],
    #          [35, 4 ],
    #          [35, 13],
    #          [36, 5 ],
    #          [36, 13],
    #          [37, 5 ],
    #          [37, 12],
    #          [37, 13],
    #          [38, 5 ],
    #          [38, 6 ],
    #          [38, 11],
    #          [38, 12],
    #          [39, 6 ],
    #          [39, 7 ],
    #          [39, 8 ],
    #          [39, 9 ],
    #          [39, 10],
    #          [39, 11],
    #          [59, 11]
    #         ])

    xs = data[:,0].reshape(-1,1)
    ys = data[:,1].reshape(-1,1)
    print('xs',xs)
    print('ys',ys)

    a = b  = c = d = e = 0
    f = 1.0

    print('leng',len(data))
    U = np.zeros((len(data), 5))
    # print('U', U)
    for i, (pX, pY) in enumerate(data):
        # print(pnt[0], pnt[1])
        print(pX, pY)
        U[i][0] = pX * pX
        U[i][1] = pX * pY
        U[i][2] = pY * pY
        U[i][3] = pX
        U[i][4] = pY
    print('final u', U)

    B = np.zeros((len(data), 1))
    for i, (pX, pY) in enumerate(data):
        B[i][0] = -f

    # u1, s1, Vt1 = np.linalg.svd(U)
    # print('svd shape', u1.shape, s1.shape, Vt1.shape)
    w, u, Vt = cv2.SVDecomp(U)
    print('svd shape', u.shape, w.shape, Vt.shape)

    print('Vt', Vt)
    print('u', u)
    print('w', w)

    # print('u',u)
    # print('s',s)
    # print('Vt',Vt)

    D = np.eye(5,5)
    D[0][0] = w[0]
    D[1][1] = w[1]
    D[2][2] = w[2]
    D[3][3] = w[3]
    D[4][4] = w[4]

    # print(np.linalg.inv(D))
    # R = Vt.T * U.T
    print('shape', Vt.T.shape, np.linalg.inv(D).shape, u.T.shape, B.shape)
    # np.matrix

    X = Vt.T *  np.asmatrix(D).I * u.T * B
    # X = Vt.T * np.linalg.inv(D) * np.ones((5,27)) * B

    # X = Vt.T *  np.invert(np.asmatrix(D)) * np.asmatrix(u.T) * B
    print('X',X, '\nU',U)
    a = X[0][0]
    b = X[1][0]
    c = X[2][0]
    d = X[3][0]
    e = X[4][0]

    #X a [[0.00024428]] b [[0.00149567]] c [[-0.00091559]] d [[-0.03855311]] e [[-0.0368046]]

    print('a',a,'b', b,'c', c,'d', d,'e', e)
    # print( 'ret1', Vt.T *  np.asmatrix(D).I )
    # print( 'ret2', np.asmatrix(u.T) * B)
    # print('ret3', Vt.T )
    # print('ret4', np.linalg.inv(D))

    img = cv2.cvtColor(th,cv2.COLOR_GRAY2RGB)
    # cv2.rectangle(img, (np.min(data, axis=0)[0], np.min(data, axis=0)[1]),
    #               (np.max(data, axis=0)[0], np.max(data, axis=0)[1]),
    #               (255, 0, 0), 1)
    # cv2.waitKey(0)

    # double  Ay, By, Cy
    # double DD;
    # double xe1, xe2, ye1, ye2;
    # double resi;
    # unsigned char  oriintensity0, oriintensity1;

    minX = np.min(data, axis=0)[0]
    maxX = np.max(data, axis=0)[0]
    minY = np.min(data, axis=0)[1]
    maxY = np.max(data, axis=0)[1]
    print("minX", minX, minY, maxX, maxY)
    width = th.shape[0]
    print('width', width)
    # for (int j = (int) minX; j <= (int) maxX; j++)

    for j in range(minX, maxX+1, 1):
        print(j)
        Ay = c
        By = b * j + e # j = x
        Cy = a * j * j + d * j + f

        DD = By * By - 4 * Ay * Cy

        if (DD < 0):
            # DD = -DD
            print("contin2")
            continue

        ye1 = (-(By) + math.sqrt(DD)) / (2 * Ay)
        ye2 = (-(By) - math.sqrt(DD)) / (2 * Ay)

        cv2.circle(img, ( j, (int)(ye1 + 0.5)), 1, (0, 255, 0), 1)
        print('point',j, (int)(ye1 + 0.5))
        cv2.circle(img, ( j, (int)(ye2 + 0.5)), 1, (0, 255, 0), 1)
        print('point', j, (int)(ye2 + 0.5))

        # th[(int)(ye1 + 0.5)][j] = 0;
        # th[(int)(ye2 + 0.5)][j] = 0;
        # img[width * (int)(ye1 + 0.5) + j] = (0,255,0);
        # img[width * (int)(ye2 + 0.5) + j] = (0,255,0);

    # for (int i = (int)minY; i <= (int)maxY; i++)
    for i in range(minY, maxY + 1, 1):
        print(i)
        Ax = a
        Bx = b * i + d # i = y
        Cx = c * i * i + e * i + f

        DD = Bx * Bx - 4 * Ax * Cx

        if (DD < 0):
            # DD = -DD
            print("contin")
            continue

        xe1 = (-(Bx) + math.sqrt(DD)) / (2 * Ax)
        xe2 = (-(Bx) - math.sqrt(DD)) / (2 * Ax)

        cv2.circle(img, ( (int)(xe1 + 0.5), i), 1, (255, 0, 0), 1)
        print('point2',(int)(xe1 + 0.5), i)
        cv2.circle(img, ( (int)(xe2 + 0.5), i), 1, (255, 0, 0), 1)
        print('point2',(int)(xe2 + 0.5), i)

        # th[i][(int)(xe1 + 0.5)] = 0
        # th[i][(int)(xe2 + 0.5)] = 0


    cv2.imshow('fit_rotated_ellipse_4', th)
    cv2.imshow('fit_rotated_ellipse_', img)
    cv2.waitKey(0)

    # print(1/0)


    # return (cx,cy,w,h,theta)

def calc_and_draw_ellipse_full(img_path):
    color_list = [(238,0,0),(0,252,124),(142,56,142)]
    # color_list = [(255,0,0),(0,255,0),(255,255,0)]


    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    retv, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_of_face, lface, leye = detect_face_eye_using_opencv(gray)
    contours , _ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("gray", gray)
    cv2.imshow("th", th)
    print(contours)
    for con in contours:
        approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
        area = cv2.contourArea(con)
        print("shape of approx", approx.shape)
        print("area", area)
        if(len(approx) > 8 and area > 50 and area < 10000):

            a,b,r = fit_circle(con.reshape(-1,2))
            cv2.drawContours(src,[con],0,color_list[2],2)
            cv2.circle(src,(int(a),int(b)),int(r),color_list[1])
            cx,cy,w,h,theta = fit_rotated_ellipse(con.reshape(-1,2))
            print(cx,cy,w,h,theta)
            if (math.isnan(h) == True or math.isnan(w) == True or math.isnan(cx) == True  or math.isnan(cx) == True or theta < 0):
                print("Nan")
            else:
                cv2.ellipse(src, (int(cx), int(cy)), (int(w), int(h)), theta * 180.0 / np.pi, 0.0, 360.0, color_list[0],2)

    cv2.imshow("result", src)
    cv2.waitKey(0)
    cv2.imwrite(img_path + 'out.png',src)

def calc_and_draw_ellipse(gray):
    color_list = [(238,0,0),(0,252,124),(142,56,142),(0,0,255)]
    # color_list = [(255,0,0),(0,255,0),(255,255,0)]

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # check2 = cv2.equalizeHist(gray)
    th = cv2.Canny(gray, 50, 127)
    # th = cv2.Canny(gray, 61, 138)
    # retv, th = cv2.threshold(gray, 25, 223, cv2.THRESH_BINARY)
    # retv, th = cv2.threshold(check2, 25, 223, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # retv, th = cv2.threshold(check2, 180, 255, cv2.THRESH_BINARY + cv2.ADAPTIVE_THRESH_MEAN_C)
    # th2 =  cv2.adaptiveThreshold(check2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("th2", th)
    # cv2.waitKey(0)

    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # cv2.imshow("calcHist", gray)
    # plt.plot(hist, color='black')
    # plt.show()


    contours , _ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("gray", gray)
    # cv2.imshow("th", th)
    print(len(contours)," detected\n", contours)
    for con in contours:
        approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
        area = cv2.contourArea(con)
        print("shape of approx", approx.shape)
        print("area", area)
        if(len(approx) > 13 and area > 6 and area < 8000):

            a,b,r = fit_circle(con.reshape(-1,2))
            cv2.drawContours(img,[con],0,color_list[2],2)
            cv2.circle(img,(int(a),int(b)),int(r),color_list[1])
            cv2.rectangle(img, (np.min(con.reshape(-1,2), axis=0)[0], np.min(con.reshape(-1,2), axis=0)[1]),
                          (np.max(con.reshape(-1,2), axis=0)[0], np.max(con.reshape(-1,2), axis=0)[1]),
                          (255, 0, 0), 1)
            fit_rotated_ellipse_svd(con.reshape(-1, 2), th)

            cx, cy, w, h, theta = fit_rotated_ellipse(con.reshape(-1, 2))
            print(cx, cy, w, h, theta)
            if (math.isnan(h) == True or math.isnan(w) == True or math.isnan(cx) == True or math.isnan(cy) == True ): #or theta < 0
                print("Nan")
            else:
                cv2.ellipse(img, (int(cx), int(cy)), (int(w), int(h)), theta * 180.0 / np.pi, 0.0, 360.0, color_list[3], 1)
            cv2.imshow("result", img)
            cv2.waitKey(0)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    # cv2.imwrite(img_path + 'out.png',img)

def main(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # retv, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # num_of_face, pfaces = detect_face_eye_using_dlib(gray)

    calc_and_draw_ellipse(gray)
    cv2.imshow("final", gray)
    cv2.waitKey(0)


if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('usage : {0} < image_abs_path >'.format(sys.argv[0]))
        exit(0)
    main(sys.argv[1])