import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -------- Load Model -------- #

base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(29, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.load_weights("models/sign_model.h5")

# -------- Labels -------- #

labels = [
'A','B','C','D','E','F','G','H','I','J','K','L','M',
'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'del','nothing','space'
]

# -------- MediaPipe Setup -------- #

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------- Webcam -------- #

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list)-20, max(x_list)+20
            ymin, ymax = min(y_list)-20, max(y_list)+20

            # Prevent out-of-frame errors
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            hand = frame[ymin:ymax, xmin:xmax]

            if hand.size != 0:

                hand = cv2.resize(hand, (224,224))
                hand = hand / 255.0
                hand = np.expand_dims(hand, axis=0)

                pred = model.predict(hand, verbose=0)
                class_id = np.argmax(pred)
                confidence = np.max(pred)

                if confidence > 0.8:

                    label = labels[class_id]

                    cv2.putText(frame,
                                f"{label} {confidence:.2f}",
                                (xmin, ymin-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,255,0),
                                2)

            # Draw bounding box
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)

    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()