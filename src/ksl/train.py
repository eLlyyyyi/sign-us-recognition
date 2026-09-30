import os
import numpy as np
import joblib
import optuna

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================
# 경로 설정
# ==========================================================
# 단어 데이터
WORD_DATA_PATH = os.environ.get("SIGNUS_KSL_WORD_DATA_DIR", "data/ksl/words")
WORD_LIST_FILE = os.environ.get("SIGNUS_KSL_WORD_LIST", "data/ksl/word_list.txt")

# 문장 데이터
SEN_DATA_PATH = os.environ.get("SIGNUS_KSL_SENTENCE_DATA_DIR", "data/ksl/sentences")
SEN_LIST_FILE = os.environ.get("SIGNUS_KSL_SENTENCE_LIST", "data/ksl/sentence_list.txt")

MODEL_DIR = os.environ.get("SIGNUS_KSL_MODEL_DIR", "artifacts/ksl_model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = 64

# 튜닝: REAL01~16
TUNE_REALS = list(range(1, 17))
# 최종 평가: REAL17~18
EVAL_REALS = [17, 18]

# 4-Fold (튜닝용, 4명씩 균등)
TUNE_FOLDS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

hand_angle_idx = [(0,1,2), (1,2,3), (2,3,4), (0,5,6), (5,6,7), (6,7,8),
                  (0,9,10), (9,10,11), (10,11,12), (0,13,14), (13,14,15), (14,15,16),
                  (0,17,18), (17,18,19), (18,19,20)]
pose_angle_idx = [(11, 13, 15), (12, 14, 16), (23, 11, 13), (24, 12, 14)]
face_small_indices = [1, 4, 6, 7, 10, 11, 14, 15, 16, 17]

def calculate_angles(points, angle_indices):
    angles = []
    for A, B, C in angle_indices:
        pA, pB, pC = points[A], points[B], points[C]
        if np.all(pB == 0):
            angles.append(0.0); continue
        v1, v2 = pA - pB, pC - pB
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            angles.append(0.0)
        else:
            dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angles.append(np.arccos(dot))
    return np.array(angles, dtype=np.float32)

def convert_raw_to_hybrid_features(raw_frame):
    lh   = raw_frame[:63].reshape(21, 3)
    rh   = raw_frame[63:126].reshape(21, 3)
    pose = raw_frame[126:225].reshape(33, 3)
    face = raw_frame[225:291].reshape(22, 3)

    lh_angles = calculate_angles(lh, hand_angle_idx)
    rh_angles = calculate_angles(rh, hand_angle_idx)
    p_angles  = calculate_angles(pose, pose_angle_idx)

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
    for i in face_small_indices:
        pt = face[i]
        if not np.all(pt == 0):
            face_rel.extend(((pt[:2] - nose[:2]) / s_width).tolist())
        else:
            face_rel.extend([0.0, 0.0])
    face_rel = np.array(face_rel, dtype=np.float32)

    return np.concatenate([
        lh_angles, rh_angles, p_angles,
        rel_l_wrist, rel_r_wrist, rel_l_mid, rel_r_mid,
        rel_nose, face_rel
    ]).astype(np.float32)

def augment_noise(data, noise_std=0.01):
    return data + np.random.normal(0, noise_std, data.shape).astype(np.float32)

def augment_speed_up(data, factor=0.75):
    length = len(data)
    new_length = max(1, int(length * factor))
    indices = np.linspace(0, length - 1, new_length).astype(int)
    return data[indices]

def augment_speed_down(data, factor=1.25):
    length = len(data)
    new_length = int(length * factor)
    indices = np.linspace(0, length - 1, new_length).astype(int)
    return data[indices]

def augment_flip(data):
    flipped = data.copy()
    flipped[:, :15], flipped[:, 15:30] = data[:, 15:30].copy(), data[:, :15].copy()
    flipped[:, 34:36], flipped[:, 36:38] = data[:, 36:38].copy(), data[:, 34:36].copy()
    flipped[:, 34] = -flipped[:, 34]
    flipped[:, 36] = -flipped[:, 36]
    flipped[:, 38:40], flipped[:, 40:42] = data[:, 40:42].copy(), data[:, 38:40].copy()
    flipped[:, 38] = -flipped[:, 38]
    flipped[:, 40] = -flipped[:, 40]
    flipped[:, 42] = -flipped[:, 42]
    for i in range(44, 64, 2):
        flipped[:, i] = -flipped[:, i]
    return flipped

def apply_augmentation(data_list):
    aug_list = []
    for data, label in data_list:
        aug_list.append((data, label))
        aug_list.append((augment_noise(data), label))
        aug_list.append((augment_speed_up(data), label))
        aug_list.append((augment_speed_down(data), label))
        aug_list.append((augment_flip(data), label))
    return aug_list

# ==========================================================
# 데이터 검증 함수
# ==========================================================
def verify_data(list_file, data_path):
    """목록과 데이터셋 폴더 일치 여부 확인"""
    target_words = set()
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                target_words.add(line.split(':', 1)[1].strip())
    folder_words = set()
    for folder_name in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder_name)):
            word = folder_name.split('_', 1)[1] if '_' in folder_name else folder_name
            folder_words.add(word)
    missing = target_words - folder_words
    extra = folder_words - target_words
    return missing, extra

# ==========================================================
# 데이터 로딩 함수
# ==========================================================
def load_all_data():
    all_reals = TUNE_REALS + EVAL_REALS
    person_data = {i: [] for i in all_reals}

    # --- 단어 데이터 로딩 ---
    print("📌 단어 데이터 로딩 중...")
    for folder_name in os.listdir(WORD_DATA_PATH):
        folder_path = os.path.join(WORD_DATA_PATH, folder_name)
        if not os.path.isdir(folder_path): continue
        word = folder_name.split('_', 1)[1] if '_' in folder_name else folder_name
        label = word  # 단어 이름으로 라벨

        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.npy'): continue
            parts = file_name.split('_')
            real_part = next((p for p in parts if p.startswith('REAL')), None)
            if real_part is None: continue
            real_num = int(real_part.replace('REAL', ''))
            if real_num not in person_data: continue
            path = os.path.join(folder_path, file_name)
            raw_data = np.load(path)
            if len(raw_data.shape) == 2 and raw_data.shape[1] == 291:
                hybrid = np.array([convert_raw_to_hybrid_features(f) for f in raw_data])
                person_data[real_num].append((hybrid, label))

    # --- 문장 데이터 로딩 ---
    print("📌 문장 데이터 로딩 중...")
    for folder_name in os.listdir(SEN_DATA_PATH):
        folder_path = os.path.join(SEN_DATA_PATH, folder_name)
        if not os.path.isdir(folder_path): continue
        sen = folder_name.split('_', 1)[1] if '_' in folder_name else folder_name
        label = sen  # 단어 의미로 라벨

        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.npy'): continue
            parts = file_name.split('_')
            real_part = next((p for p in parts if p.startswith('REAL')), None)
            if real_part is None: continue
            real_num = int(real_part.replace('REAL', ''))
            if real_num not in person_data: continue
            path = os.path.join(folder_path, file_name)
            raw_data = np.load(path)
            if len(raw_data.shape) == 2 and raw_data.shape[1] == 291:
                hybrid = np.array([convert_raw_to_hybrid_features(f) for f in raw_data])
                person_data[real_num].append((hybrid, label))

    return person_data

# ==========================================================
# Optuna 목적 함수
# ==========================================================
def objective(trial, person_data, person_aug, encoder, MAX_FRAMES):
    conv_filters  = trial.suggest_categorical('conv_filters', [32, 64, 128])
    lstm1_units   = trial.suggest_categorical('lstm1_units', [64, 128, 256])
    lstm2_units   = trial.suggest_categorical('lstm2_units', [64, 128, 256])
    dense_units   = trial.suggest_categorical('dense_units', [64, 128])
    dropout_rate  = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1)
    batch_size    = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001])

    NUM_CLASSES = len(encoder.classes_)
    fold_scores = []

    for fold_idx, val_reals in enumerate(TUNE_FOLDS):
        train_reals = [r for r in TUNE_REALS if r not in val_reals]

        X_val_list = [d for r in val_reals for d, _ in person_data[r]]
        y_val_list = [l for r in val_reals for _, l in person_data[r]]
        X_val = pad_sequences(X_val_list, maxlen=MAX_FRAMES, dtype='float32', padding='post', value=0.0)
        y_val = to_categorical(encoder.transform(y_val_list), num_classes=NUM_CLASSES)

        X_tr_list = [d for r in train_reals for d, _ in person_aug[r]]
        y_tr_list = [l for r in train_reals for _, l in person_aug[r]]
        X_tr = pad_sequences(X_tr_list, maxlen=MAX_FRAMES, dtype='float32', padding='post', value=0.0)
        y_tr = to_categorical(encoder.transform(y_tr_list), num_classes=NUM_CLASSES)

        from tensorflow.keras.optimizers import Adam
        model = Sequential([
            Input(shape=(MAX_FRAMES, FEATURES)),
            Conv1D(filters=conv_filters, kernel_size=3, activation='relu', padding='same'),
            LSTM(lstm1_units, return_sequences=True, activation='tanh'),
            LSTM(lstm2_units, return_sequences=False, activation='tanh'),
            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=50, batch_size=batch_size, callbacks=[es], verbose=0)

        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_scores.append(val_acc)

    return np.mean(fold_scores)

# ==========================================================
# 메인
# ==========================================================
def main():
    # 데이터 검증
    print("📌 데이터 검증 중...")

    word_missing, word_extra = verify_data(WORD_LIST_FILE, WORD_DATA_PATH)
    sen_missing, sen_extra = verify_data(SEN_LIST_FILE, SEN_DATA_PATH)

    has_error = False
    if word_missing or word_extra:
        print("❌ 단어 데이터 불일치!")
        if word_missing: print(f"  목록에 있는데 폴더 없음: {sorted(word_missing)}")
        if word_extra: print(f"  폴더에 있는데 목록 없음: {sorted(word_extra)}")
        has_error = True
    else:
        print("✅ 단어 데이터 일치!")

    if sen_missing or sen_extra:
        print("❌ 문장 데이터 불일치!")
        if sen_missing: print(f"  목록에 있는데 폴더 없음: {sorted(sen_missing)}")
        if sen_extra: print(f"  폴더에 있는데 목록 없음: {sorted(sen_extra)}")
        has_error = True
    else:
        print("✅ 문장 데이터 일치!")

    if has_error:
        print("\n❌ 데이터 불일치로 학습 중단!")
        return

    print("✅ 모든 데이터 일치! 학습 시작\n")

    # 데이터 로딩
    person_data = load_all_data()

    # 튜닝용 증강 (REAL01~16만)
    person_aug = {}
    for real_num in TUNE_REALS:
        person_aug[real_num] = apply_augmentation(person_data[real_num])

    all_lengths = [len(d) for real in TUNE_REALS for d, _ in person_aug[real]]
    all_lengths += [len(d) for real in EVAL_REALS for d, _ in person_data[real]]
    MAX_FRAMES = max(all_lengths)
    print(f"최대 프레임 수: {MAX_FRAMES}")

    all_labels = [l for real in TUNE_REALS for _, l in person_aug[real]]
    all_labels += [l for real in EVAL_REALS for _, l in person_data[real]]
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    NUM_CLASSES = len(encoder.classes_)
    print(f"클래스 수: {NUM_CLASSES}")

    # Optuna 튜닝
    print("\n🔍 Optuna 하이퍼파라미터 튜닝 시작 (50 trial)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, person_data, person_aug, encoder, MAX_FRAMES),
        n_trials=50,
        show_progress_bar=True
    )

    best_params = study.best_params
    print(f"\n{'='*50}")
    print(f"✅ 베스트 Trial 번호: {study.best_trial.number + 1} / 50")
    print(f"✅ 최적 하이퍼파라미터:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  OOF 평균 검증 정확도: {study.best_value:.4f}")
    print(f"{'='*50}")

    joblib.dump(best_params, os.path.join(MODEL_DIR, 'best_params.pkl'))

    # 최종 평가 (REAL17, 18)
    print("\n📊 최종 평가 (REAL17, 18)...")
    all_tune_aug = [item for real in TUNE_REALS for item in person_aug[real]]
    X_tr_list = [d for d, _ in all_tune_aug]
    y_tr_list = [l for _, l in all_tune_aug]
    X_tr = pad_sequences(X_tr_list, maxlen=MAX_FRAMES, dtype='float32', padding='post', value=0.0)
    y_tr = to_categorical(encoder.transform(y_tr_list), num_classes=NUM_CLASSES)

    X_eval_list = [d for real in EVAL_REALS for d, _ in person_data[real]]
    y_eval_list = [l for real in EVAL_REALS for _, l in person_data[real]]
    X_eval = pad_sequences(X_eval_list, maxlen=MAX_FRAMES, dtype='float32', padding='post', value=0.0)
    y_eval = to_categorical(encoder.transform(y_eval_list), num_classes=NUM_CLASSES)

    from tensorflow.keras.optimizers import Adam
    eval_model = Sequential([
        Input(shape=(MAX_FRAMES, FEATURES)),
        Conv1D(filters=best_params['conv_filters'], kernel_size=3, activation='relu', padding='same'),
        LSTM(best_params['lstm1_units'], return_sequences=True, activation='tanh'),
        LSTM(best_params['lstm2_units'], return_sequences=False, activation='tanh'),
        Dense(best_params['dense_units'], activation='relu'),
        Dropout(best_params['dropout_rate']),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    eval_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                       loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    eval_model.fit(X_tr, y_tr, validation_data=(X_eval, y_eval),
                   epochs=100, batch_size=best_params['batch_size'], callbacks=[es], verbose=1)

    _, eval_acc = eval_model.evaluate(X_eval, y_eval, verbose=0)
    print(f"\n✅ 최종 평가 정확도 (REAL17, 18): {eval_acc:.4f}")

    # 재학습: 18명 전체
    print("\n🔥 18명 전체로 재학습 시작...")
    all_reals = TUNE_REALS + EVAL_REALS
    person_aug_all = {}
    for real_num in TUNE_REALS:
        person_aug_all[real_num] = person_aug[real_num]
    for real_num in EVAL_REALS:
        person_aug_all[real_num] = apply_augmentation(person_data[real_num])

    all_aug = [item for real in all_reals for item in person_aug_all[real]]
    X_all_list = [d for d, _ in all_aug]
    y_all_list = [l for _, l in all_aug]

    MAX_FRAMES_ALL = max(len(d) for d in X_all_list)
    X_all = pad_sequences(X_all_list, maxlen=MAX_FRAMES_ALL, dtype='float32', padding='post', value=0.0)
    y_all = to_categorical(encoder.transform(y_all_list), num_classes=NUM_CLASSES)

    final_model = Sequential([
        Input(shape=(MAX_FRAMES_ALL, FEATURES)),
        Conv1D(filters=best_params['conv_filters'], kernel_size=3, activation='relu', padding='same'),
        LSTM(best_params['lstm1_units'], return_sequences=True, activation='tanh'),
        LSTM(best_params['lstm2_units'], return_sequences=False, activation='tanh'),
        Dense(best_params['dense_units'], activation='relu'),
        Dropout(best_params['dropout_rate']),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                        loss='categorical_crossentropy', metrics=['accuracy'])
    final_model.fit(X_all, y_all, epochs=80, batch_size=best_params['batch_size'], verbose=1)

    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'ksl_model_final.h5')
    FINAL_CLASS_PATH = os.path.join(MODEL_DIR, 'ksl_classes_final.pkl')
    final_model.save(FINAL_MODEL_PATH)
    joblib.dump(encoder.classes_, FINAL_CLASS_PATH)

    print(f"\n{'='*50}")
    print(f"✅ 최종 평가 정확도 (REAL17, 18): {eval_acc:.4f}")
    print(f"✅ 최종 모델 저장: {FINAL_MODEL_PATH}")
    print(f"✅ 라벨 저장: {FINAL_CLASS_PATH}")
    print(f"{'='*50}")

    LOG_PATH = os.path.join(MODEL_DIR, 'result_log.txt')
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"베스트 Trial 번호: {study.best_trial.number + 1} / 50\n")
        f.write(f"{'='*50}\n")
        f.write(f"최적 하이퍼파라미터:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"OOF 평균 검증 정확도: {study.best_value:.4f}\n")
        f.write(f"{'='*50}\n")
        f.write(f"최종 평가 정확도 (REAL17, 18): {eval_acc:.4f}\n")
        f.write(f"최종 모델 저장: {FINAL_MODEL_PATH}\n")
        f.write(f"라벨 저장: {FINAL_CLASS_PATH}\n")
        f.write(f"{'='*50}\n")
    print(f"✅ 결과 로그 저장: {LOG_PATH}")

if __name__ == "__main__":
    main()
