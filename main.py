import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
a=0.4                                   #번호판 반투명 상수
number_img = cv2.imread('number.jpg')   #번호판 이미지
answer_point=(0,0)                      #손가락 끝점 좌표
none_time = time.time()                 #시간체크용
flag =False                             #손인식 3초 이상시 Flag On
number_flag = False                     #손인식 3초 이상시 번호판 로딩
number_x =0                             #손인식 3초 후 손중심점 x좌표
number_y =0                             #손인식 3초 후 손중심점 y좌표
text = ""                               #번호 저장용
text_com = ""                           #엔터키 입력시 번호 상단에 표시용
timer1=0                                #번호입력 시간 체크용
timer1_None = 0
now_time = 0
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")      #얼군부분 제거시 사용

width = number_img.shape[1]
height = number_img.shape[0]


def detect(img, cascade):
    # 얼굴 부분 제거
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def removeFaceAra(img, cascade):
    # 얼굴 부분 제거
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, cascade)

    height, width = img.shape[:2]

    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

    return img

def img_set(origin):
    # 살색 검출을 위한 전처리
    #origin = removeFaceAra(origin,cascade)
    hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 30], dtype="uint8")
    upper = np.array([15, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsv, lower, upper)

    blurred = cv2.blur(skinRegionHSV, (3, 3))
    ret, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def num_img(x,y):
    # 번호판 이미지 비디오에 삽입
    global flag
    global number_flag
    try:
        dst = frame[y-height: y, x-width: x]
        b = 1.0 - a
        dst = cv2.addWeighted(number_img, a, dst, b, 0)
        frame[y-height: y, x-width: x] = dst
    except:
        flag = False
        number_flag = False

    return frame

def detect_contour(thresh):
    #컨투어 후 가장큰 범위를 가진 컨투어를 반환
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    max_contour = None
    max_area = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (w * h) * 0.4 > area:
            continue
        if w > h:
            continue
        if area > max_area:
            max_area = area
            max_contour = contour
    if max_area < 10000:
        max_area = -1

    return max_area, max_contour

def calculateAngle(A, B):
    #convexHull에 사용할 각도계산
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
    return angle

def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

def getFingerPosition(max_contour, img_result):
    #손가락 검출
    points1 = []
    M = cv2.moments(max_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)

    points = points1 + points2
    points = list(set(points))

    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)
    return 1, new_points

def time_chack(x, y, w, h):
    #번호판 눌린시간 체크
    global start
    global timer1
    if x<w and w-62<x :
        if y<h and h-43<y:
            now_time = time.time()-start
            timer1 += now_time
    if timer1>1:
        timer1 =0
        return True
    else:
        return  False

def number_point(x,y, text):
    #번호판 이미지 좌표에서 번호 획득
    #x,y = 손 가락의 포인트위치 ,set_point = 번호판위치 포인트
    global width
    global height
    global number_x
    global number_y
    global flag
    global number_flag
    global text_com
    set_point_x = number_x - width
    set_point_y = number_y - height
    if (time_chack(x, y, set_point_x+62, set_point_y+43)):
        text += "1"
    elif(time_chack(x, y,set_point_x+124,set_point_y+43)):
        text += "2"
    elif(time_chack(x, y,set_point_x+186,set_point_y+43)):
        text += "3"
    elif(time_chack(x, y,set_point_x+62,set_point_y+86)):
        text += "4"
    elif(time_chack(x, y,set_point_x+124,set_point_y+86)):
        text += "5"
    elif(time_chack(x, y,set_point_x+186,set_point_y+86)):
        text += "6"
    elif(time_chack(x, y,set_point_x+62,set_point_y+129)):
        text += "7"
    elif(time_chack(x, y,set_point_x+124,set_point_y+129)):
        text += "8"
    elif(time_chack(x, y,set_point_x+186,set_point_y+129)):
        text += "9"
    elif(time_chack(x, y,set_point_x+124,set_point_y+172)):
        text += "0"
    elif(time_chack(x, y,set_point_x+248,set_point_y+86)):
        flag = False
        number_flag = False
        text_com = text
        text = ""
    elif(time_chack(x, y,set_point_x+248,set_point_y+43)):
        text = ""
    return text

while True:
    #프로그램 시작후 이미지 전처리와 컨투어
    start = time.time()
    ret, frame = cap.read()

    thresh = img_set(frame)

    max_area, max_contour = detect_contour(thresh)
    # 일정 사이즈 이상일때만 진입(노이즈)
    if max_area > 2500:
        mmt = cv2.moments(max_contour)

        cx = int(mmt['m10'] / mmt['m00'])
        cy = int(mmt['m01'] / mmt['m00'])

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 4)

        ret, points = getFingerPosition(max_contour, frame)

        if ret > 0 :
            for p0 in points:
                answer_point= p0
                if(p0[0]*p0[1] > answer_point[0]*answer_point[1]):
                    answer_point = p0
            cv2.circle(frame, answer_point, 5, [255, 0, 255], 3)

        text = number_point(answer_point[0], answer_point[1], text)
        #print(answer_point)
        cv2.putText(frame, text, (number_x - width, number_y - height - 20), 2, 1, (255, 0, 0), 2)

    #번호판 이미지 로딩 이미 번호판이 로딩되어 있다면 한번만 실행
    if max_area > 2500:
        start_time = time.time()
        if start_time-none_time>=3 and number_flag == False:
            number_x = cx
            number_y = cy
            flag = True
            number_flag = True
    else :
        none_time = time.time()
    if flag == True:
        frame = num_img(number_x, number_y)

    print(flag)
    print(number_flag)
    cv2.putText(frame, text_com, (400, 50), 2, 1, (255, 0, 0), 2)
    cv2.imshow('stream', frame)
    cv2.imshow('stream2', thresh)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()