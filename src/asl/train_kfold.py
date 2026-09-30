# ================================================================
# ASL 인식 모델 학습 코드 - npy_v2 (260차원) 버전
# ================================================================

import os
import tensorflow as tf
import numpy as np
import joblib
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder


# ================================================================
# [설정]
# ================================================================

BASE_PATH       = os.environ.get("SIGNUS_ASL_FEATURE_DIR", "data/asl/features")
TRAIN_VAL_PATHS = [
    os.path.join(BASE_PATH, "train"),
    os.path.join(BASE_PATH, "val"),
]
TEST_PATH       = os.path.join(BASE_PATH, "test")

SEED = 2026
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

MODEL_DIR = os.environ.get("SIGNUS_ASL_MODEL_DIR", "artifacts/asl_model")
os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS = {
    'conv_filters' : 64,
    'lstm1_units'  : 128,
    'lstm2_units'  : 128,
    'dense_units'  : 128,
    'dropout_rate' : 0.5,
    'batch_size'   : 32,
    'learning_rate': 0.0005,
}

FEATURES       = 260          # ← 변경
MAX_FRAMES_CAP = 60
MIN_FRAMES     = 5
N_FOLDS        = 4

FOLDS = [
    ['P7',  'P10', 'P11', 'P12'],
    ['P20', 'P21', 'P26', 'P27'],
    ['P29', 'P30', 'P31', 'P33'],
    ['P37', 'P39', 'P40', 'P50', 'P51', 'P52'],
]
ALL_TRAINVAL_PERSONS = [p for fold in FOLDS for p in fold]


# ================================================================
# [피험자 자동 감지]
# ================================================================

def extract_person_id(file_name):
    base  = os.path.splitext(file_name)[0]
    parts = base.split('_')
    for part in parts:
        if part.startswith('P') and part[1:].isdigit():
            return part
    return None

def scan_persons(split_path):
    persons = set()
    if not os.path.isdir(split_path):
        return persons
    for word_folder in os.listdir(split_path):
        word_path = os.path.join(split_path, word_folder)
        if not os.path.isdir(word_path):
            continue
        for file_name in os.listdir(word_path):
            if not file_name.endswith('.npy'):
                continue
            pid = extract_person_id(file_name)
            if pid:
                persons.add(pid)
    return persons

def scan_persons_multi(split_paths):
    persons = set()
    for split_path in split_paths:
        persons.update(scan_persons(split_path))
    return persons

def make_folds(persons, n_folds=4):
    persons_sorted = sorted(persons, key=lambda x: int(x[1:]))
    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(persons_sorted):
        folds[i % n_folds].append(p)
    return folds


# ================================================================
# [데이터 로딩] 260차원 그대로 로딩 (변환 없음)
# ================================================================

def load_by_person(split_path, allowed_persons):
    person_data = {}
    if not os.path.isdir(split_path):
        print(f"❌ 경로 없음: {split_path}")
        return person_data

    for word_folder in sorted(os.listdir(split_path)):
        word_path = os.path.join(split_path, word_folder)
        if not os.path.isdir(word_path):
            continue
        label = word_folder

        for file_name in os.listdir(word_path):
            if not file_name.endswith('.npy'):
                continue
            person = extract_person_id(file_name)
            if person is None or person not in allowed_persons:
                continue

            data = np.load(os.path.join(word_path, file_name))
            if data.ndim != 2 or data.shape[1] != FEATURES or len(data) < MIN_FRAMES:
                continue

            if person not in person_data:
                person_data[person] = []
            person_data[person].append((data.astype(np.float32), label))

    return person_data

def load_by_person_multi(split_paths, allowed_persons):
    merged = {}
    for split_path in split_paths:
        split_data = load_by_person(split_path, allowed_persons)
        for person, items in split_data.items():
            merged.setdefault(person, []).extend(items)
    return merged


def load_test():
    data_list = []
    if not os.path.isdir(TEST_PATH):
        print(f"❌ test 경로 없음: {TEST_PATH}")
        return data_list

    for word_folder in sorted(os.listdir(TEST_PATH)):
        word_path = os.path.join(TEST_PATH, word_folder)
        if not os.path.isdir(word_path):
            continue
        label = word_folder
        for file_name in os.listdir(word_path):
            if not file_name.endswith('.npy'):
                continue
            data = np.load(os.path.join(word_path, file_name))
            if data.ndim != 2 or data.shape[1] != FEATURES or len(data) < MIN_FRAMES:
                continue
            data_list.append((data.astype(np.float32), label))

    return data_list


# ================================================================
# [데이터 증강]
# ================================================================

def aug_noise(d):
    return d + np.random.normal(0, 0.005, d.shape).astype(np.float32)

def aug_speed_up(d):
    n = max(1, int(len(d)*0.75))
    return d[np.linspace(0, len(d)-1, n).astype(int)]

def aug_speed_down(d):
    n = int(len(d)*1.25)
    return d[np.linspace(0, len(d)-1, n).astype(int)]

def aug_flip(d):
    """LH↔RH 스왑 (260차원 기준)"""
    f = d.copy()
    # lh_norm(0~62) ↔ rh_norm(63~125)
    f[:, 0:63],  f[:, 63:126] = d[:, 63:126].copy(), d[:, 0:63].copy()
    # lh_angles(126~138) ↔ rh_angles(139~151)
    f[:, 126:139], f[:, 139:152] = d[:, 139:152].copy(), d[:, 126:139].copy()
    # rel_pos x 부호 반전
    f[:, 152] = -f[:, 152]
    # lh_vel(254~256) ↔ rh_vel(257~259)
    f[:, 254:257], f[:, 257:260] = d[:, 257:260].copy(), d[:, 254:257].copy()
    # 정규화 좌표 x축 부호 반전
    f[:, 0:63:3] = -f[:, 0:63:3]
    f[:, 63:126:3] = -f[:, 63:126:3]
    return f

def aug_time_shift(d, shift=5):
    shift = min(shift, len(d) - 1)
    pad   = np.zeros((shift, d.shape[1]), dtype=np.float32)
    return np.concatenate([pad, d[:-shift]], axis=0)

def aug_frame_mask(d, mask_ratio=0.1):
    result  = d.copy()
    n_mask  = max(1, int(len(d) * mask_ratio))
    indices = np.random.choice(len(d), n_mask, replace=False)
    result[indices] = 0.0
    return result

def aug_scale(d, scale=0.9):
    result = d.copy()
    result[:, 0:126] = result[:, 0:126] * scale  # 손 정규화 좌표 스케일
    return result

def augment(data_list):
    result = []
    for data, label in data_list:
        result += [
            (data,                 label),
            (aug_noise(data),      label),
            (aug_speed_up(data),   label),
            (aug_speed_down(data), label),
            (aug_flip(data),       label),
            (aug_time_shift(data), label),
            (aug_frame_mask(data), label),
            (aug_noise(aug_flip(data)), label),
        ]
    return result


# ================================================================
# [모델 생성]
# ================================================================

def build_model(num_classes, max_frames):
    l2 = regularizers.l2(1e-4)
    model = Sequential([
        Input(shape=(max_frames, FEATURES)),
        Conv1D(PARAMS['conv_filters'], kernel_size=3, activation='relu',
               padding='same', kernel_regularizer=l2),
        BatchNormalization(),
        LSTM(PARAMS['lstm1_units'], return_sequences=True,  activation='tanh', kernel_regularizer=l2),
        LSTM(PARAMS['lstm2_units'], return_sequences=False, activation='tanh', kernel_regularizer=l2),
        Dense(PARAMS['dense_units'], activation='relu', kernel_regularizer=l2),
        BatchNormalization(),
        Dropout(PARAMS['dropout_rate']),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=PARAMS['learning_rate']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


# ================================================================
# [메인]
# ================================================================

def main():

    # ── STEP 1. 피험자 자동 감지 ──────────────────────────────
    print("=" * 55)
    print("STEP 1. 피험자 자동 감지")
    print("=" * 55)

    detected_persons = scan_persons_multi(TRAIN_VAL_PATHS)
    all_persons = [p for p in ALL_TRAINVAL_PERSONS if p in detected_persons]
    if not all_persons:
        print("❌ train/val 폴더에서 학습 대상 피험자(P번호)를 찾지 못했습니다.")
        return

    print(f"  감지된 피험자 ({len(all_persons)}명): {all_persons}")

    print(f"\n  {N_FOLDS}-Fold 구성:")
    for i, fold in enumerate(FOLDS):
        print(f"    Fold {i+1} 검증: {fold}")

    # ── STEP 2. 데이터 로딩 ───────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2. 데이터 로딩 (260차원)")
    print("=" * 55)

    person_data = load_by_person_multi(TRAIN_VAL_PATHS, set(all_persons))
    test_data   = load_test()

    loaded_persons = sorted(person_data.keys(), key=lambda x: int(x[1:]))
    print(f"  로딩된 피험자: {loaded_persons}")
    for p in loaded_persons:
        print(f"    {p}: {len(person_data[p])}개 샘플")

    total_trainval = sum(len(v) for v in person_data.values())
    print(f"  총 train+val 샘플: {total_trainval}개")
    print(f"  test 샘플: {len(test_data)}개")

    # ── STEP 3. 라벨 인코더 ───────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3. 라벨 인코더 준비")
    print("=" * 55)

    all_labels = [l for v in person_data.values() for _, l in v]
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    NUM_CLASSES = len(encoder.classes_)
    print(f"  클래스 수: {NUM_CLASSES}개")

    # 증강 후 길이를 기준으로 잡되, 과도한 zero padding을 막기 위해 상한을 둔다.
    all_trainval_raw = [item for v in person_data.values() for item in v]
    all_trainval_aug = augment(all_trainval_raw)
    aug_lengths = [len(d) for d, _ in all_trainval_aug]
    MAX_FRAMES = min(max(aug_lengths), MAX_FRAMES_CAP)
    over_cap = sum(length > MAX_FRAMES for length in aug_lengths)
    print(f"  최대 프레임 수: {MAX_FRAMES} (증강 후 기준, 상한 {MAX_FRAMES_CAP} 적용)")
    print(f"  상한 초과 증강 샘플: {over_cap}/{len(aug_lengths)}개")

    def to_xy(data_list):
        clipped = [(d[:MAX_FRAMES], l) for d, l in data_list]
        X = pad_sequences([d for d,_ in clipped], maxlen=MAX_FRAMES,
                          dtype='float32', padding='post', value=0.0)
        y = to_categorical(encoder.transform([l for _,l in clipped]),
                           num_classes=NUM_CLASSES)
        return X, y

    # ── STEP 4. 4-Fold Cross Validation ───────────────────────
    print("\n" + "=" * 55)
    print(f"STEP 4. {N_FOLDS}-Fold Cross Validation (피험자 기준)")
    print("=" * 55)

    fold_scores = []
    best_epochs = []
    fold_models = []

    for fold_idx, val_persons in enumerate(FOLDS):
        train_persons = [p for p in all_persons if p not in val_persons]

        val_persons_exist   = [p for p in val_persons   if p in person_data]
        train_persons_exist = [p for p in train_persons if p in person_data]

        print(f"\n  [Fold {fold_idx+1}/{N_FOLDS}]")
        print(f"    검증 피험자: {val_persons_exist}")
        print(f"    학습 피험자: {train_persons_exist}")

        train_raw = [item for p in train_persons_exist for item in person_data[p]]
        val_raw   = [item for p in val_persons_exist   for item in person_data[p]]

        if not train_raw or not val_raw:
            print(f"    ⚠️  Fold {fold_idx+1} 스킵: 데이터 없음")
            continue

        train_aug = augment(train_raw)
        X_tr,  y_tr  = to_xy(train_aug)
        X_val, y_val = to_xy(val_raw)

        print(f"    학습 샘플: {len(X_tr)}개 (증강 포함)  |  검증 샘플: {len(X_val)}개")

        model = build_model(NUM_CLASSES, MAX_FRAMES)
        es    = EarlyStopping(monitor='val_loss', patience=10,
                              restore_best_weights=True, verbose=0)
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-5, verbose=0)
        hist  = model.fit(X_tr, y_tr,
                          validation_data=(X_val, y_val),
                          epochs=80, batch_size=PARAMS['batch_size'],
                          callbacks=[es, rlrop], verbose=1)

        best_epoch = int(np.argmin(hist.history['val_loss'])) + 1
        best_epochs.append(best_epoch)

        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_scores.append(val_acc)
        fold_models.append(model)
        print(f"    ✅ Fold {fold_idx+1} Val 정확도: {val_acc:.4f} ({val_acc*100:.1f}%)  best epoch: {best_epoch}")

    if not fold_scores:
        print("❌ 유효한 Fold가 없습니다.")
        return

    mean_acc       = np.mean(fold_scores)
    optimal_epochs = int(np.mean(best_epochs))
    print(f"\n  {'='*45}")
    print(f"  {N_FOLDS}-Fold 평균 Val 정확도: {mean_acc:.4f} ({mean_acc*100:.1f}%)")
    for i, s in enumerate(fold_scores):
        print(f"    Fold {i+1}: {s:.4f}  (best epoch: {best_epochs[i]})")
    print(f"  Fold 평균 best epoch: {optimal_epochs}")
    print(f"  {'='*45}")

    # ── STEP 5. 전체 재학습 ───────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5. 전체 재학습 (train_val 전체)")
    print("=" * 55)

    all_trainval = [item for v in person_data.values() for item in v]
    all_aug      = augment(all_trainval)
    X_all, y_all = to_xy(all_aug)
    retrain_epochs = optimal_epochs * 2
    print(f"  전체 학습 샘플: {len(X_all)}개 (증강 포함)")
    print(f"  재학습 epoch 수: {retrain_epochs}")

    final_model = build_model(NUM_CLASSES, MAX_FRAMES)
    rlrop_final = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                    patience=5, min_lr=1e-5, verbose=1)
    final_model.fit(X_all, y_all,
                    epochs=retrain_epochs,
                    batch_size=PARAMS['batch_size'],
                    callbacks=[rlrop_final], verbose=1)

    # ── STEP 6. 최종 평가 ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6. 최종 평가 (test)")
    print("=" * 55)

    if not test_data:
        print("  ⚠️  test 데이터 없음 — 평가 스킵")
        test_acc = None
        final_tta_acc = None
        ensemble_acc = None
        ensemble_tta_acc = None
    else:
        X_test, y_test = to_xy(test_data)
        _, test_acc = final_model.evaluate(X_test, y_test, verbose=0)

        def accuracy_from_probs(probs):
            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(probs, axis=1)
            return float(np.mean(y_true == y_pred))

        def predict_tta(model, data_list):
            variants = [
                data_list,
                [(aug_speed_up(d), l) for d, l in data_list],
                [(aug_speed_down(d), l) for d, l in data_list],
            ]
            probs = []
            for variant in variants:
                X_variant, _ = to_xy(variant)
                probs.append(model.predict(X_variant, batch_size=PARAMS['batch_size'], verbose=0))
            return np.mean(probs, axis=0)

        final_tta_probs = predict_tta(final_model, test_data)
        final_tta_acc = accuracy_from_probs(final_tta_probs)

        ensemble_probs = np.mean(
            [model.predict(X_test, batch_size=PARAMS['batch_size'], verbose=0) for model in fold_models],
            axis=0
        )
        ensemble_acc = accuracy_from_probs(ensemble_probs)

        ensemble_tta_probs = np.mean(
            [predict_tta(model, test_data) for model in fold_models],
            axis=0
        )
        ensemble_tta_acc = accuracy_from_probs(ensemble_tta_probs)

        print(f"  ✅ {N_FOLDS}-Fold 평균 Val 정확도: {mean_acc:.4f} ({mean_acc*100:.1f}%)")
        print(f"  ✅ 최종 Test 정확도:         {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"  ✅ 최종 Test TTA 정확도:     {final_tta_acc:.4f} ({final_tta_acc*100:.1f}%)")
        print(f"  ✅ Fold Ensemble 정확도:     {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
        print(f"  ✅ Ensemble + TTA 정확도:    {ensemble_tta_acc:.4f} ({ensemble_tta_acc*100:.1f}%)")

    # ── STEP 7. 모델 저장 ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 7. 모델 저장")
    print("=" * 55)

    final_model.save(os.path.join(MODEL_DIR, 'asl_model_260.h5'))
    for i, fold_model in enumerate(fold_models, start=1):
        fold_model.save(os.path.join(MODEL_DIR, f'asl_fold{i}_260.h5'))
    joblib.dump(encoder.classes_,
                os.path.join(MODEL_DIR, 'asl_classes_260.pkl'))
    joblib.dump({'MAX_FRAMES': MAX_FRAMES, 'FEATURES': FEATURES, 'PARAMS': PARAMS},
                os.path.join(MODEL_DIR, 'asl_meta_260.pkl'))

    log_lines = [
        f"데이터 경로: {BASE_PATH}",
        f"학습/검증 split: {[os.path.basename(p) for p in TRAIN_VAL_PATHS]}",
        f"테스트 split: {os.path.basename(TEST_PATH)}",
        f"특징 차원: {FEATURES}",
        f"피험자 수: {len(all_persons)}명  {all_persons}",
        f"클래스 수: {NUM_CLASSES}",
        f"최대 프레임 수: {MAX_FRAMES}",
        "하이퍼파라미터:",
        *[f"  {k}: {v}" for k, v in PARAMS.items()],
        "",
        f"{N_FOLDS}-Fold 결과:",
        *[f"  Fold {i+1}: {s:.4f}" for i, s in enumerate(fold_scores)],
        f"  평균: {mean_acc:.4f}",
        "",
        f"최종 Test 정확도: {test_acc:.4f}" if test_acc else "최종 Test 정확도: 평가 없음",
        f"최종 Test TTA 정확도: {final_tta_acc:.4f}" if final_tta_acc else "최종 Test TTA 정확도: 평가 없음",
        f"Fold Ensemble 정확도: {ensemble_acc:.4f}" if ensemble_acc else "Fold Ensemble 정확도: 평가 없음",
        f"Ensemble + TTA 정확도: {ensemble_tta_acc:.4f}" if ensemble_tta_acc else "Ensemble + TTA 정확도: 평가 없음",
    ]
    with open(os.path.join(MODEL_DIR, 'result_log_260.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    print(f"  저장 위치: {MODEL_DIR}")
    print(f"    - asl_model_260.h5")
    print(f"    - asl_fold1_260.h5 ~ asl_fold{len(fold_models)}_260.h5")
    print(f"    - asl_classes_260.pkl")
    print(f"    - asl_meta_260.pkl")
    print(f"    - result_log_260.txt")
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
