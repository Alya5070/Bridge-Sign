import cv2
import math
import os
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/C"
TARGET_COUNT = 100
COUNTDOWN_SECONDS = 5
START_KEY = ord('s')
QUIT_KEY = ord('q')

os.makedirs(folder, exist_ok=True)

counter = 0
countdown_active = False
capture_active = False
countdown_start = 0.0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    hands, _ = detector.findHands(img, draw=False)
    imgOutput = img.copy()

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Boundary-safe cropping
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            continue

        # Make 3-channel white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:
            # Height is bigger -> scale by height
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            # Width is bigger -> scale by width
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        if capture_active:
            counter += 1
            cv2.imwrite(f'{folder}/Image.{time.time()}.jpg', imgWhite)
            cv2.putText(imgOutput, f"Saved {counter}/{TARGET_COUNT}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            if counter >= TARGET_COUNT:
                capture_active = False
        else:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Countdown handling
    if countdown_active:
        elapsed = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed
        if remaining > 0:
            cv2.putText(imgOutput, f"Starting in {math.ceil(remaining)}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            countdown_active = False
            capture_active = True
            counter = 0

    # Instructions overlay
    cv2.putText(imgOutput, f"Press 's' to start ({TARGET_COUNT} shots after {COUNTDOWN_SECONDS}s)", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(imgOutput, "Press 'q' to stop/quit", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if capture_active:
        cv2.putText(imgOutput, f"Capturing: {counter}/{TARGET_COUNT}", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1) & 0xFF
    if key == START_KEY and not countdown_active and not capture_active:
        countdown_active = True
        countdown_start = time.time()
    if key == QUIT_KEY:
        break
    if capture_active and counter >= TARGET_COUNT:
        cv2.putText(imgOutput, "Done!", (10, 510),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Image", imgOutput)
        cv2.waitKey(500)
        break

cap.release()
cv2.destroyAllWindows()
