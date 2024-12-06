import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_finger_tip = None

def is_palm_open(hand_landmarks):
    """
    Determines if the hand is open based on the relative positions of the landmarks.
    This checks if all fingers are extended (open palm gesture).
    """
    finger_tips = [8, 12, 16, 20]  
    finger_bases = [6, 10, 14, 18] 
  
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y:
            return False
    return True

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_palm_open(hand_landmarks):
                x_center = int(hand_landmarks.landmark[0].x * img.shape[1])
                y_center = int(hand_landmarks.landmark[0].y * img.shape[0])
                erase_radius = 50 
                cv2.circle(canvas, (x_center, y_center), erase_radius, (0, 0, 0), thickness=-1)  
            else:
              
                finger_tip_index = 8
                x = int(hand_landmarks.landmark[finger_tip_index].x * img.shape[1])
                y = int(hand_landmarks.landmark[finger_tip_index].y * img.shape[0])

                cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

                if prev_finger_tip is not None:
                    cv2.line(canvas, prev_finger_tip, (x, y), (255, 0, 0), 5)  # Blue line
                prev_finger_tip = (x, y)

    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    cv2.imshow("Air Drawing", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
