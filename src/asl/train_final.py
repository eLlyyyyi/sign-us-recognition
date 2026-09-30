import os
import re
import gc
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization, Activation, Bidirectional, LSTM, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
MODEL_SAVE_DIR = os.environ.get("SIGNUS_ASL_MODEL_DIR", "artifacts/asl_model")
FINAL_TXT_PATH = os.path.join(MODEL_SAVE_DIR, "result.txt")

EXP_CONFIG = {
    "name": "시도2_증강10배",
    "description": "dropout(0.5) + 증강(10) / 정제 데이터",
    "train_val_path": os.environ.get("SIGNUS_ASL_TRAIN_VAL_DIR", "data/asl/holdout/train_val"),
    "test_path": os.environ.get("SIGNUS_ASL_TEST_DIR", "data/asl/holdout/test"),
    "patience": 10,
    "epochs": 80,
    "n_folds": 4,
}

# --- [특징점 연산용 인덱스 매핑 테이블] ---
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

def augment_safe_noise(data, noise_std=0.005):
    noisy_data = data.copy()
    noise = np.random.normal(0, noise_std, (data.shape[0], data.shape[1] - 34)).astype(np.float32)
    noisy_data[:, 34:] += noise
    return noisy_data

def augment_linear_speed(data, factor):
    old_size = len(data)
    new_size = max(1, int(old_size * factor))
    old_indices = np.arange(old_size)
    new_indices = np.linspace(0, old_size - 1, new_size)
    augmented_data = np.zeros((new_size, data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        augmented_data[:, i] = np.interp(new_indices, old_indices, data[:, i])
    return augmented_data

def augment_total_rotation(data, angle_degrees):
    rad = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotated_data = data.copy()
    for i in range(34, data.shape[1], 2):
        if i+1 < data.shape[1]:
            x, y = data[:, i], data[:, i+1]
            rotated_data[:, i] = x * cos_a - y * sin_a
            rotated_data[:, i+1] = x * sin_a + y * cos_a
    return rotated_data

def augment_flip_ksl_style(data):
    flipped = data.copy()
    flipped[:, :15], flipped[:, 15:30] = data[:, 15:30].copy(), data[:, :15].copy()
    flipped[:, 34:36], flipped[:, 36:38] = data[:, 36:38].copy(), data[:, 34:36].copy()
    flipped[:, 34] = -flipped[:, 34]; flipped[:, 36] = -flipped[:, 36]
    flipped[:, 38:40], flipped[:, 40:42] = data[:, 40:42].copy(), data[:, 38:40].copy()
    flipped[:, 38] = -flipped[:, 38]; flipped[:, 40] = -flipped[:, 40]
    flipped[:, 42] = -flipped[:, 42]
    for i in range(44, 64, 2):
        flipped[:, i] = -flipped[:, i]
    return flipped

def augment_time_shift(data, shift=5):
    padding = np.zeros((shift, data.shape[1]), dtype=np.float32)
    return np.concatenate([padding, data], axis=0)

def apply_clean_10x_matrix(data):
    f_data = augment_flip_ksl_style(data)
    return [
        data,                                           # 1. 원본
        augment_safe_noise(data),                       # 2. 노이즈
        augment_linear_speed(data, factor=0.75),        # 3. 고속화
        augment_linear_speed(data, factor=1.25),        # 4. 저속화
        f_data,                                         # 5. 좌우반전
        augment_linear_speed(data, factor=0.90),        # 6. 미세 고속화
        augment_linear_speed(data, factor=1.10),        # 7. 미세 저속화
        augment_total_rotation(data, angle_degrees=-4), # 8. 좌측 회전
        augment_total_rotation(data, angle_degrees=4),  # 9. 우측 회전
        augment_time_shift(data, shift=5),              # 10. 시간 이동
    ]

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

def build_model(input_shape, num_classes, dropout_rate=0.5):
    model = Sequential([
        Input(shape=input_shape), Dense(64),
        Conv1D(64, kernel_size=3, padding='same'), BatchNormalization(), Activation('relu'),
        Conv1D(128, kernel_size=3, padding='same'), BatchNormalization(), Activation('relu'),
        Conv1D(256, kernel_size=3, padding='same'), BatchNormalization(), Activation('relu'),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dense(128), BatchNormalization(), Activation('relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])
    return model

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("\n==================================================")
    print(f"🚀 {EXP_CONFIG['name']} 시작")
    print("==================================================")

    train_val_data, train_val_labels, train_val_groups = load_data(EXP_CONFIG["train_val_path"])
    test_data, test_labels, _ = load_data(EXP_CONFIG["test_path"])

    encoder = LabelEncoder()
    encoder.fit(train_val_labels)
    joblib.dump(encoder.classes_, os.path.join(MODEL_SAVE_DIR, "asl_classes_final.pkl"))
    classes = encoder.classes_
    num_classes = len(classes)

    train_val_data = np.array(train_val_data, dtype=object)
    train_val_labels_arr = np.array(train_val_labels)
    train_val_groups_arr = np.array(train_val_groups)
    unique_persons = np.unique(train_val_groups_arr)

    print("📐 global_max_frames 계산 중...")
    all_aug_list = []
    for d in train_val_data:
        for aug in apply_clean_10x_matrix(d):
            all_aug_list.append(aug)
    global_max_frames = max(len(x) for x in all_aug_list)
    del all_aug_list; gc.collect()
    print(f"✅ global_max_frames = {global_max_frames}")

    fold_scores, fold_logs = [], []
    best_fold_idx, best_fold_val_acc = -1, -1.0
    kf = KFold(n_splits=EXP_CONFIG["n_folds"], shuffle=True, random_state=42)

    for fold_idx, (train_p_idx, val_p_idx) in enumerate(kf.split(unique_persons)):
        train_persons = unique_persons[train_p_idx]; val_persons = unique_persons[val_p_idx]
        train_mask = np.isin(train_val_groups_arr, train_persons); val_mask = np.isin(train_val_groups_arr, val_persons)

        X_tr_list, y_tr_list = [], []
        for idx in np.where(train_mask)[0]:
            for aug_data in apply_clean_10x_matrix(train_val_data[idx]):
                X_tr_list.append(aug_data); y_tr_list.append(train_val_labels_arr[idx])

        X_val_list = [train_val_data[i] for i in np.where(val_mask)[0]]

        X_tr = pad_sequences(X_tr_list, maxlen=global_max_frames, dtype="float32", padding="post", value=0.0)
        y_tr = to_categorical(encoder.transform(y_tr_list), num_classes=num_classes)
        X_val = pad_sequences(X_val_list, maxlen=global_max_frames, dtype="float32", padding="post", value=0.0)
        y_val = to_categorical(encoder.transform(train_val_labels_arr[val_mask]), num_classes=num_classes)

        print(f"\n👉 [Fold {fold_idx+1}] 학습 시작 (데이터 {len(X_tr)}개)...")
        model = build_model(X_tr.shape[1:], num_classes, dropout_rate=0.5)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

        fold_model_path = os.path.join(MODEL_SAVE_DIR, f"{EXP_CONFIG['name']}_fold{fold_idx+1}_best.h5")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=EXP_CONFIG["patience"], restore_best_weights=True, verbose=0),
            ModelCheckpoint(fold_model_path, monitor="val_loss", save_best_only=True, verbose=0)
        ]

        history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EXP_CONFIG["epochs"], batch_size=32, callbacks=callbacks, verbose=1)

        best_epoch_idx = np.argmin(history.history['val_loss'])
        b_val_acc = history.history['val_accuracy'][best_epoch_idx]
        b_val_loss = history.history['val_loss'][best_epoch_idx]
        b_tr_acc = history.history['accuracy'][best_epoch_idx]
        b_tr_loss = history.history['loss'][best_epoch_idx]
        
        train_persons_list = ", ".join(sorted(train_persons))
        val_persons_list = ", ".join(sorted(val_persons))
        fold_scores.append(b_val_acc)
        
        log = (f"{fold_idx+1}폴드: 최종 검증 정확도={b_val_acc:.4f}, 최종 검증 오차={b_val_loss:.4f}, "
               f"학습={len(X_tr)}개/{len(train_persons)}명, 검증={len(X_val)}개/{len(val_persons)}명\n"
               f"  학습 사람: {train_persons_list}\n"
               f"  학습 파일 수: {len(X_tr)}개\n"
               f"  검증 사람: {val_persons_list}\n"
               f"  검증 파일 수: {len(X_val)}개\n"
               f"  [베스트 에폭 {best_epoch_idx+1}번] accuracy={b_tr_acc:.4f}, loss={b_tr_loss:.4f}, "
               f"val_accuracy={b_val_acc:.4f}, val_loss={b_val_loss:.4f}")
        
        # ✨ 기획 수선: 폴드가 끝날 때마다 베스트 에폭 결과 실시간 출력
        print(f"\n📢 [Fold {fold_idx+1} 완료 성적표]")
        print(log)
        print("-" * 50)
        
        fold_logs.append(log)
        if b_val_acc > best_fold_val_acc: best_fold_val_acc = b_val_acc; best_fold_idx = fold_idx + 1

        del model; tf.keras.backend.clear_session(); gc.collect()

    best_saved_path = os.path.join(MODEL_SAVE_DIR, f"{EXP_CONFIG['name']}_fold{best_fold_idx}_best.h5")
    final_best_model = load_model(best_saved_path)
    test_max_frames = final_best_model.input_shape[1]
    X_ts = pad_sequences(test_data, maxlen=test_max_frames, dtype="float32", padding="post", value=0.0)
    y_ts = to_categorical(encoder.transform(test_labels), num_classes=num_classes)
    t_loss, t_acc = final_best_model.evaluate(X_ts, y_ts, verbose=0)

    worst_fold_idx = int(np.argmin(fold_scores)) + 1
    worst_fold_acc = min(fold_scores)

    with open(FINAL_TXT_PATH, "a", encoding="utf-8") as f:
        f.write(f"{'=' * 50}\n실험명: {EXP_CONFIG['name']}\n설정: {EXP_CONFIG['description']}\n{'=' * 50}\n")
        f.write("K-Fold 검증 결과:\n")
        f.write(f"평균 검증 정확도: {np.mean(fold_scores):.4f}\n")
        for log_line in fold_logs: f.write(log_line + "\n")
        f.write(f"최저 Fold: {worst_fold_idx} ({worst_fold_acc:.4f})\n")
        f.write(f"베스트 Fold: {best_fold_idx} ({best_fold_val_acc:.4f})\n")
        f.write(f"Fold 최고-최저 차이: {best_fold_val_acc - worst_fold_acc:.4f}\n")
        f.write(f"베스트 Fold 테스트 정확도: {t_acc:.4f}\n")
        f.write(f"베스트 Fold 테스트 오차: {t_loss:.4f}\n{'=' * 50}\n\n")
    print(f"\n✅ {EXP_CONFIG['name']} 완료. result.txt 결과를 확인하세요.")

if __name__ == "__main__": main()
