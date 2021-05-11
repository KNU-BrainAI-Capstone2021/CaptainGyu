import dlib
import skvideo
path = "C:\\FFmpeg\\bin"
skvideo.setFFmpegPath(path)
import skvideo.io
import cv2
import time


# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))



detector = dlib.get_frontal_face_detector()

vid_in = cv2.VideoCapture('./input/test.mp4')

fourcc = cv2.VideoWriter_fourcc(*'FMP4')
out = cv2.VideoWriter('dlib_tracked.mp4', fourcc, 20, (640, 480),isColor=True)

est_FPS = 0
while True:
    # Get frame from video
    # get success : ret = True / fail : ret= False

    ret, image_o = vid_in.read()


   # resize the video
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # estimate FPS start
    preTime = time.time()

    # Get faces (up-sampling=1)
    face_detector = detector(img_gray, 1)

    # loop as the number of face
    # one loop belong to one face
    for face in face_detector:
        # face wrapped with rectangle
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 0, 255), 3)


    dstTime = time.time()
    est_FPS = (1./(dstTime - preTime))

    cv2.imshow('result', image)
    out.write(image)
    print("dlib FPS : ", est_FPS)
    # wait for keyboard input
    key = cv2.waitKey(1)

    # if esc,
    if key == 27:
        break

vid_in.release()
out.release()
cv2.destroyAllWindows()