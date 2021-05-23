import dlib
import skvideo.io
import cv2

## face detector와 landmark predictor 정의
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

## 비디오 읽어오기
cap = skvideo.io.vreader('test.mp4')

## 각 frame마다 얼굴 찾고, landmark 찍기
for frame in cap:    
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    r = 200. / img.shape[1]
    dim = (200, int(img.shape[0] * r))    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    rects = detector(resized, 1)
    for i, rect in enumerate(rects):
        l = rect.left()
        t = rect.top()
        b = rect.bottom()
        r = rect.right()
        shape = predictor(resized, rect)
        '''
        for j in range(68):
            x, y = shape.part(j).x, shape.part(j).y
            cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)'''
        cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()
