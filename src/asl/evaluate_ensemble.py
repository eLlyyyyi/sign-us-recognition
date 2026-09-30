import os
import re
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

MODEL_SAVE_DIR = os.environ.get("SIGNUS_ASL_MODEL_DIR", "artifacts/asl_model")
TEST_PATH = os.environ.get("SIGNUS_ASL_TEST_DIR", "data/asl/holdout/test")
FINAL_TXT_PATH = os.path.join(MODEL_SAVE_DIR, "result.txt")
EXP_NAME = "시도2_증강10배"
N_FOLDS = 4

hand_angle_idx = [(0,1,2), (1,2,3), (2,3,4), (0,5,6), (5,6,7), (6,7,8),
                  (0,9,10), (9,10,11), (10,11,12), (0,13,14), (13,14,15), (14,15,16),
                  (0,17,18), (17,18,19), (18,19,20)]
pose_angle_idx = [(11, 13, 15), (12, 14, 16), (23, 11, 13), (24, 12, 14)]
face_small_indices = [1, 4, 6, 7, 10, 11, 14, 15, 16, 17]

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

def convert_raw_to_hybrid_features(raw_frame):
    lh = raw_frame[:63].reshape(21, 3); rh = raw_frame[63:126].reshape(21, 3)
    pose = raw_frame[126:225].reshape(33, 3); face = raw_frame[225:291].reshape(22, 3)
    lh_angles = calculate_angles(lh, hand_angle_idx); rh_angles = calculate_angles(rh, hand_angle_idx); p_angles = calculate_angles(pose, pose_angle_idx)
    l_sh, r_sh = pose[11], pose[12]; origin = (l_sh + r_sh) / 2.0; s_width = np.linalg.norm(l_sh - r_sh)
    if s_width < 0.1: s_width = 0.5
    l_wrist, r_wrist = pose[15], pose[16]
    rel_l_wrist = (l_wrist[:2] - origin[:2]) / s_width if not np.all(l_wrist == 0) else np.zeros(2)
    rel_r_wrist = (r_wrist[:2] - origin[:2]) / s_width if not np.all(r_wrist == 0) else np.zeros(2)
    rel_l_mid = (lh[12][:2] - origin[:2]) / s_width if not np.all(lh == 0) else np.zeros(2)
    rel_r_mid = (rh[12][:2] - origin[:2]) / s_width if not np.all(rh == 0) else np.zeros(2)
    nose = pose[0]; rel_nose = (nose[:2] - origin[:2]) / s_width if not np.all(nose == 0) else np.zeros(2)
    face_rel = []
    for i in face_small_indices:
        pt = face[i]
        if not np.all(pt == 0): face_rel.extend(((pt[:2] - nose[:2]) / s_width).tolist())
        else: face_rel.extend([0.0, 0.0])
    return np.concatenate([lh_angles, rh_angles, p_angles, rel_l_wrist, rel_r_wrist, rel_l_mid, rel_r_mid, rel_nose, face_rel]).astype(np.float32)

def extract_person_id(file_name):
    stem = os.path.splitext(file_name)[0].upper()
    match = re.match(r"^(P\d+)_", stem)
    return match.group(1) if match else stem

def load_data(path):
    data, labels, groups = [], [], []
    for folder_name in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder_name)
        if not os.path.isdir(folder_path): continue
        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".npy"): continue
            raw_data = np.load(os.path.join(folder_path, file_name))
            if len(raw_data.shape) == 2 and raw_data.shape[1] == 291:
                hybrid = np.array([convert_raw_to_hybrid_features(f) for f in raw_data])
                data.append(hybrid); labels.append(folder_name); groups.append(extract_person_id(file_name))
    return data, labels, groups

def main():
    test_data, test_labels, _ = load_data(TEST_PATH)

    classes = joblib.load(os.path.join(MODEL_SAVE_DIR, "asl_classes_final.pkl"))
    encoder = LabelEncoder(); encoder.classes_ = classes; num_classes = len(classes)
    y_ts_labels = encoder.transform(test_labels)

    # ✅ 폴드1 모델 기준으로 입력 shape 확인
    fold1_path = os.path.join(MODEL_SAVE_DIR, f"{EXP_NAME}_fold1_best.h5")
    fold1_model = load_model(fold1_path)
    max_frames = fold1_model.input_shape[1]
    print(f"✅ 입력 shape 기준: {max_frames} 프레임")
    del fold1_model; tf.keras.backend.clear_session()

    X_ts = pad_sequences(test_data, maxlen=max_frames, dtype="float32", padding="post", value=0.0)
    y_ts = to_categorical(y_ts_labels, num_classes=num_classes)

    # ✅ 각 폴드 단일 테스트
    fold_accs = []
    ensemble_probs = np.zeros((len(X_ts), num_classes), dtype=np.float32)
    for fi in range(1, N_FOLDS + 1):
        fold_path = os.path.join(MODEL_SAVE_DIR, f"{EXP_NAME}_fold{fi}_best.h5")
        fold_model = load_model(fold_path)
        _, t_acc = fold_model.evaluate(X_ts, y_ts, verbose=0)
        fold_accs.append(t_acc)
        ensemble_probs += fold_model.predict(X_ts, verbose=0)
        print(f"Fold {fi} 테스트 정확도: {t_acc:.4f}")
        del fold_model; tf.keras.backend.clear_session()

    # ✅ 앙상블 결과
    ensemble_probs /= N_FOLDS
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = np.mean(ensemble_preds == y_ts_labels)
    print(f"\n앙상블 테스트 정확도: {ensemble_acc:.4f}")

    with open(FINAL_TXT_PATH, "a", encoding="utf-8") as f:
        f.write(f"{'=' * 50}\n{EXP_NAME} 앙상블 테스트 결과\n{'=' * 50}\n")
        for fi, acc in enumerate(fold_accs, 1):
            f.write(f"Fold {fi} 테스트 정확도: {acc:.4f}\n")
        f.write(f"앙상블 테스트 정확도: {ensemble_acc:.4f}\n{'=' * 50}\n\n")
    print(f"\n✅ 완료. result.txt 결과를 확인하세요.")

if __name__ == "__main__": main()
