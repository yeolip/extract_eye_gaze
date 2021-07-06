# extract_eye_gaze
차량안에 실내카메라를 이용한 운전자 모니터링 프로젝트를 진행중이며, 
그안에서 stereo camera calibration을 담당하고있다.
Head Eye tracking의 경우, 외부 업체의 알고리즘을 사용하는데,
그 기술등이 은닉되어 있고, 구현 방법등이 궁금하여, 
Eye tracking을 하기위한 여러 오픈소스를 통해 구현해 보았다.
딥러닝을 통한 Head pos나 Eye gaze를 계산하는 방법들도 있지만,
카메라의 화각이나 위치에 따를 re-트레이닝이 필요할것으로 판단되어
범용적이지 못할것 같아, gaze를 계산에 의해 도출하는 방법을 개인검토 차원에서 사용해보았다. 

1. face를 찾기위해, dlib 사용
2. camera intrinsic calibration 진행
2. 2d face feature와 camera intrinsic param을 이용한 3d face model 좌표 추정  
3. face rotation 도출(base on euler angle)
4. 2d eye crop을 통한 pupil center 찾기
5. eye gaze base code를 통한 eye pupil에서 camera plain coord로의 gaze vector
6. eyeball center에서 eye pupil으로의 gaze vector

#reference and code

https://github.com/BenjaminPoilve/Eye-Gaze-Estimator Eye gaze base code
https://github.com/indigopyj/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV Head rotation
https://github.com/davisking/dlib-models extract face model using dlib
Fabian Timm and Erhardt Barth - “Accurate Eye Centre Localisation by Means of Gradients” - find pupil pos
https://github.com/jonnedtc/PupilDetector speed up pupilDetector
https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/ how to train face feature on dlib
https://ibug.doc.ic.ac.uk/resources/300-W/ - dlib training example
Li Jianfeng and Li Shigang - “Eye-Model-Based Gaze Estimation by RGB-D Camera” -  calculate vector from center of eyeball to pupil center 
https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c


