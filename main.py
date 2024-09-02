# A u t h o r : S a i k a t  B h a t t a c h a r y y a

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

rpoints = [deque(maxlen=1024)]
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
kpoints = [deque(maxlen=1024)]
wpoints = [deque(maxlen=1024)]

red_index = 0
blue_index = 0
green_index = 0
black_index = 0
white_index = 0

colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0), (255, 255, 255)]
brush_size = 5

paintWindow = np.ones((471, 636, 3), np.uint8) * 255

button_positions = [(20, 20, 130, 60), (150, 20, 130, 60), (280, 20, 130, 60), (410, 20, 130, 60), (540, 20, 130, 60)]
button_texts = ["CLEAR", "RED", "BLUE", "GREEN", "BLACK"]

for i, (x, y, w, h) in enumerate(button_positions):
    cv2.rectangle(paintWindow, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, button_texts[i], (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

colorIndex = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i, (x, y, w, h) in enumerate(button_positions):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, button_texts[i], (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if (thumb[1] - center[1] < 30):
            rpoints.append(deque(maxlen=512))
            red_index += 1
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            kpoints.append(deque(maxlen=512))
            black_index += 1
            wpoints.append(deque(maxlen=512))
            white_index += 1

        elif center[1] <= 80:
            for i, (x, y, w, h) in enumerate(button_positions):
                if x < center[0] < x + w and y < center[1] < y + h:
                    if i == 0:
                        paintWindow[100:, :, :] = 255
                        rpoints = [deque(maxlen=1024)]
                        bpoints = [deque(maxlen=1024)]
                        gpoints = [deque(maxlen=1024)]
                        kpoints = [deque(maxlen=1024)]
                        wpoints = [deque(maxlen=1024)]
                        red_index = blue_index = green_index = black_index = white_index = 0
                    else:
                        colorIndex = i - 1
                    break
        else:
            if colorIndex is not None:
                if colorIndex == 0:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 1:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 2:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 3:
                    kpoints[black_index].appendleft(center)
                elif colorIndex == 4:
                    wpoints[white_index].appendleft(center)

    else:
        rpoints.append(deque(maxlen=512))
        red_index += 1
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        kpoints.append(deque(maxlen=512))
        black_index += 1
        wpoints.append(deque(maxlen=512))
        white_index += 1

    points = [rpoints, bpoints, gpoints, kpoints, wpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is not None and points[i][j][k] is not None:
                    color = colors[i]
                    if color == (255, 255, 255):
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], color, brush_size*2)
                    else:
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], color, brush_size)
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], color, brush_size)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
