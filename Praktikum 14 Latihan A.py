import cv2
import mediapipe as mp

mpose = mp.solutions.pose
pose = mpose.Pose()
mdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = pose.process(imgrgb)

    if hasil.pose_landmarks:
        mdraw.draw_landmarks(img, hasil.pose_landmarks, mpose.POSE_CONNECTIONS)

        lm = hasil.pose_landmarks.landmark

        shoulder_kiri = lm[mpose.PoseLandmark.LEFT_SHOULDER.value]
        wrist_kiri = lm[mpose.PoseLandmark.LEFT_WRIST.value]

        shoulder_kanan = lm[mpose.PoseLandmark.RIGHT_SHOULDER.value]
        wrist_kanan = lm[mpose.PoseLandmark.RIGHT_WRIST.value]

        if wrist_kiri.y < shoulder_kiri.y:
            cv2.putText(img, "Tangan Kiri Terangkat", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if wrist_kanan.y < shoulder_kanan.y:
            cv2.putText(img, "Tangan Kanan Terangkat", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Deteksi Angkat Tangan",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()