import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Dropout, Input
import tensorflow.keras.backend as K

# quantization_config 무시하는 커스텀 Dense
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
from PIL import ImageFont, ImageDraw, Image

MODEL_PATH = os.environ.get("SIGNUS_KSL_MODEL_PATH", "artifacts/ksl_model/ksl_model_final.h5")
CLASS_PATH = os.environ.get("SIGNUS_KSL_CLASS_PATH", "artifacts/ksl_model/ksl_classes_final.pkl")

FACE_POINTS_ALL = [70, 105, 107, 336, 334, 300, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14, 78, 308, 82, 312]
FACE_SMALL_IDX  = [1, 4, 6, 7, 10, 11, 14, 15, 16, 17]
FACE_POINTS_RT  = [FACE_POINTS_ALL[i] for i in FACE_SMALL_IDX]

hand_angle_idx = [(0,1,2), (1,2,3), (2,3,4), (0,5,6), (5,6,7), (6,7,8),
                  (0,9,10), (9,10,11), (10,11,12), (0,13,14), (13,14,15), (14,15,16),
                  (0,17,18), (17,18,19), (18,19,20)]
pose_angle_idx = [(11, 13, 15), (12, 14, 16), (23, 11, 13), (24, 12, 14)]
face_small_indices = [1, 4, 6, 7, 10, 11, 14, 15, 16, 17]

def put_korean_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("malgun.ttf", font_size)
    except: font = ImageFont.load_default()
    b, g, r = color
    draw.text(position, text, font=font, fill=(r, g, b))
    return np.array(img_pil)

def calculate_angles(points, angle_indices):
    angles = []
    for A, B, C in angle_indices:
        pA, pB, pC = points[A], points[B], points[C]
        if np.all(pB == 0): angles.append(0.0); continue
        v1, v2 = pA - pB, pC - pB
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: angles.append(0.0)
        else:
            dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angles.append(np.arccos(dot))
    return np.array(angles, dtype=np.float32)

def extract_hybrid_features(results):
    lh   = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark])  if results.left_hand_landmarks  else np.zeros((21, 3))
    rh   = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    pose = np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark])       if results.pose_landmarks       else np.zeros((33, 3))
    face = np.array([[results.face_landmarks.landmark[i].x,
                      results.face_landmarks.landmark[i].y,
                      results.face_landmarks.landmark[i].z] for i in FACE_POINTS_RT]) if results.face_landmarks else np.zeros((10, 3))

    angles = np.concatenate([
        calculate_angles(lh, hand_angle_idx),
        calculate_angles(rh, hand_angle_idx),
        calculate_angles(pose, pose_angle_idx)
    ])

    l_sh, r_sh = pose[11], pose[12]
    origin = (l_sh + r_sh) / 2.0
    s_width = np.linalg.norm(l_sh - r_sh)
    if s_width < 0.1: s_width = 0.5

    l_wrist, r_wrist = pose[15], pose[16]
    rel_l_wrist = (l_wrist[:2] - origin[:2]) / s_width if not np.all(l_wrist == 0) else np.zeros(2)
    rel_r_wrist = (r_wrist[:2] - origin[:2]) / s_width if not np.all(r_wrist == 0) else np.zeros(2)
    rel_l_mid = (lh[12][:2] - origin[:2]) / s_width if not np.all(lh == 0) else np.zeros(2)
    rel_r_mid = (rh[12][:2] - origin[:2]) / s_width if not np.all(rh == 0) else np.zeros(2)

    nose = pose[0]
    rel_nose = (nose[:2] - origin[:2]) / s_width if not np.all(nose == 0) else np.zeros(2)

    face_rel = []
    for pt in face:
        if not np.all(pt == 0):
            face_rel.extend(((pt[:2] - nose[:2]) / s_width).tolist())
        else:
            face_rel.extend([0.0, 0.0])
    face_rel = np.array(face_rel, dtype=np.float32)

    return np.concatenate([angles, rel_l_wrist, rel_r_wrist, rel_l_mid, rel_r_mid, rel_nose, face_rel]).astype(np.float32)

def main():
    try:
        model = load_model(MODEL_PATH, compile=False, custom_objects={'Dense': CustomDense})
        actions = joblib.load(CLASS_PATH)
        MAX_FRAMES = model.input_shape[1]
        print(f"✅ 모델 로딩 완료! 클래스 수: {len(actions)}")
    except Exception as e:
        print(f"❌ 에러: {e}"); return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    sequence, is_recording, missing_frames = [], False, 0
    current_word = "준비 완료!"
    last_valid_angles = np.zeros(30, dtype=np.float32)

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = frame.copy()

            white_dot = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2)
            white_line = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)

            if results.pose_landmarks: mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, white_dot, white_line)
            if results.left_hand_landmarks: mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, white_dot, white_line)
            if results.right_hand_landmarks: mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, white_dot, white_line)
            if results.face_landmarks: mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, white_dot, white_line)

            hands_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)

            if hands_detected:
                missing_frames = 0
                if not is_recording: is_recording, sequence = True, []

                curr_feat = extract_hybrid_features(results)
                if np.all(curr_feat[:30] == 0.0) and not np.all(last_valid_angles == 0.0):
                    curr_feat[:30] = last_valid_angles
                else: last_valid_angles = curr_feat[:30].copy()

                sequence.append(curr_feat)
                image = put_korean_text(image, f"인식 중... ({len(sequence)})", (10, 30), 40, (0, 0, 255))
            else:
                if is_recording:
                    missing_frames += 1
                    if missing_frames > 10:
                        is_recording = False
                        if len(sequence) > 15:
                            input_data = np.array(sequence[3:-3], dtype=np.float32)
                            length = len(input_data)
                            padded = input_data[:MAX_FRAMES] if length > MAX_FRAMES else np.pad(input_data, ((0, MAX_FRAMES - length), (0, 0)), 'constant')
                            res = model.predict(np.expand_dims(padded, axis=0), verbose=0)[0]
                            top3 = np.argsort(res)[::-1][:3]
                            print("\n=== 예측 결과 ===")
                            for i in top3:
                                print(f"  {actions[i]}: {res[i]*100:.1f}%")
                            best_prob = res[np.argmax(res)]
                            if best_prob >= 0.8:
                                current_word = f"정답: {actions[np.argmax(res)]} ({best_prob*100:.1f}%)"
                            else:
                                current_word = "예측불가. 다시 동작해주세요"
                        sequence = []
                if not is_recording:
                    image = put_korean_text(image, current_word, (10, 30), 30, (255, 0, 0))

            cv2.imshow('KSL Sign Language - 182 Classes', image)
            if cv2.waitKey(10) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
