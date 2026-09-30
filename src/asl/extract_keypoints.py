import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# ==========================================================
# 경로 설정
# ==========================================================
VIDEO_DIR   = os.environ.get("SIGNUS_ASL_VIDEO_DIR", "data/asl/videos")
SAVE_DIR    = os.environ.get("SIGNUS_ASL_FEATURE_DIR", "data/asl/features")
SPLIT_DIR   = os.environ.get("SIGNUS_ASL_SPLIT_DIR", "data/asl/splits")
CSV_FILES   = {
    'train': os.path.join(SPLIT_DIR, "train.csv"),
    'val':   os.path.join(SPLIT_DIR, "val.csv"),
    'test':  os.path.join(SPLIT_DIR, "test.csv"),
}

# ==========================================================
# 매핑 테이블 (확실한 142개 KSL→ASL)
# 학습에 사용할 ASL 단어만 필터링
# ==========================================================
TARGET_ASL_WORDS = {
    'ACCESS', 'AGAIN', 'ALRIGHT', 'ALWAYS', 'APPOINTMENT', 'ARRIVE',
    'BACK', 'BACKPACK1', 'BAD', 'BAG1', 'BASEBALL', 'BECOME', 'BEFORE',
    'BELT1', 'BICYCLE', 'BLANKET', 'BUS', 'CALL1', 'CAMERA', 'CANNOT',
    'CAR', 'CARDS', 'CAREFUL', 'CELLPHONE', 'CHANGE', 'CHECK', 'CHILD',
    'CLASS', 'CLOCK1', 'COFFEE', 'DAY', 'DIE', 'DIFFERENT', 'DISAPPEAR',
    'DONTNEED', 'DOOR', 'ELEVATOR1', 'END', 'EYEGLASSES', 'FACE',
    'FAINT', 'FAMILY', 'FAST', 'FOUR', 'FRIDAY', 'FRIEND', 'GETOFF',
    'GO', 'GOAL', 'GOOD', 'GRANDFATHER', 'GRANDMOTHER', 'HANDS', 'HAPPY',
    'HARD', 'HAVE', 'HEAD', 'HEALTH', 'HEARINGAID', 'HELLO', 'HELP',
    'HERE', 'HOSPITAL1', 'HURRY', 'IMPOSSIBLE', 'KEY', 'LAPTOP', 'LATE',
    'LEFT', 'LETTER1', 'LICENSE', 'LIFT1', 'LIGHT', 'LOOKFOR', 'LOOKAT',
    'LOST', 'MAN', 'ME', 'MEDICINE', 'MEET', 'MILK1', 'MINUTE',
    'MISSING', 'MONDAY', 'MONTH', 'MOTHER', 'NAME', 'NEED', 'NEXT',
    'NINE', 'NONE', 'NOSE', 'NOTYET', 'NOW', 'NUMBERS', 'OFF', 'ONE',
    'PHONE', 'PLACE', 'POLICEMAN1', 'POSSIBLE', 'RAINBOW', 'RING',
    'ROAD', 'SEVEN', 'SIX', 'SORRY', 'STAFF', 'STAIRS', 'STOP',
    'SUNDAY', 'TEENAGER', 'TEETH', 'TELL', 'THANKYOU', 'THERE',
    'THINGS', 'THREE', 'TICKET', 'TODAY', 'TOILET', 'TONGUE', 'TRAFFIC',
    'TRAIN', 'TUESDAY', 'TURNAROUND', 'TWINS1', 'TWO', 'UMBRELLA',
    'USE', 'WAIT', 'WALLET', 'WANT1', 'WARM', 'WEDNESDAY', 'WHAT1',
    'WOMAN1', 'WRONG', 'YOU', 'BECOME', 'BACK', 'BECOME', 'BAD',
    'BRIDGE1', 'CROSS1', 'BECOME',
}

# ==========================================================
# 얼굴 핵심 랜드마크 인덱스 (22개)
# 한국 수어 전처리와 동일하게 맞춤
# ==========================================================
LEFT_EYEBROW  = [70, 105, 107]
RIGHT_EYEBROW = [336, 334, 300]
LEFT_EYE      = [33, 133, 159, 145]
RIGHT_EYE     = [362, 263, 386, 374]
MOUTH         = [61, 291, 13, 14, 78, 308, 82, 312]
FACE_POINTS   = LEFT_EYEBROW + RIGHT_EYEBROW + LEFT_EYE + RIGHT_EYE + MOUTH  # 22개

# ==========================================================
# 좌표 추출 함수 (291차원)
# 한국 수어 전처리와 동일한 구조
# ==========================================================
def extract_keypoints(results):
    lh   = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()  if results.left_hand_landmarks  else np.zeros(63)
    rh   = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    pose = np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten()       if results.pose_landmarks       else np.zeros(99)
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x,
                          results.face_landmarks.landmark[i].y,
                          results.face_landmarks.landmark[i].z] for i in FACE_POINTS]).flatten()
    else:
        face = np.zeros(66)
    return np.concatenate([lh, rh, pose, face])  # 291차원

# ==========================================================
# CSV 로딩 - 매핑 단어만 필터링
# ==========================================================
def load_csv(csv_path, split_name):
    items = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row['Gloss']
            if gloss in TARGET_ASL_WORDS:
                items.append({
                    'video_file': row['Video file'],
                    'gloss': gloss,
                    'participant': row['Participant ID'],
                    'split': split_name
                })
    return items

# ==========================================================
# 메인
# ==========================================================
def main():
    mp_holistic = mp.solutions.holistic

    # 저장 폴더 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(SAVE_DIR, split), exist_ok=True)

    # CSV 로딩
    all_items = []
    for split, csv_path in CSV_FILES.items():
        items = load_csv(csv_path, split)
        all_items.extend(items)
        print(f"{split}: {len(items)}개 (매핑 단어만)")

    print(f"\n총 처리할 영상: {len(all_items)}개\n")

    processed = 0
    skipped   = 0
    not_found = 0
    no_keypoints = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for idx, item in enumerate(all_items):
            video_file  = item['video_file']
            gloss       = item['gloss']
            split       = item['split']
            participant = item['participant']

            # 저장 경로: SAVE_DIR/train/CARRY/P1_66221823911-CARRY.npy
            gloss_dir = os.path.join(SAVE_DIR, split, gloss)
            os.makedirs(gloss_dir, exist_ok=True)

            npy_name  = f"{participant}_{video_file.replace('.mp4', '.npy')}"
            save_path = os.path.join(gloss_dir, npy_name)

            # 영상 파일 경로
            video_path = os.path.join(VIDEO_DIR, video_file)
            if not os.path.exists(video_path):
                not_found += 1
                if not_found <= 5:
                    print(f"  ⚠️  영상 없음: {video_file}")
                continue

            # 영상 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                not_found += 1
                cap.release()
                continue

            # 프레임 전체 읽기 (ASL은 영상 전체가 1단어)
            sequence_data = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = holistic.process(image_rgb)
                keypoints = extract_keypoints(results)

                # 손 또는 포즈가 감지된 프레임만 저장
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    sequence_data.append(keypoints)

            cap.release()

            if len(sequence_data) > 0:
                np.save(save_path, np.array(sequence_data))
                processed += 1
                if processed % 10 == 0:
                    print(f"  ✅  [{idx+1}/{len(all_items)}] {gloss} → {npy_name} ({len(sequence_data)} 프레임)")
            else:
                no_keypoints += 1
                print(f"  ⚠️  좌표 없음: {video_file}")

    print(f"\n{'='*50}")
    print(f"✅ 완료!")
    print(f"   저장:       {processed}개")
    print(f"   스킵:       {skipped}개 (이미 처리됨)")
    print(f"   영상없음:   {not_found}개")
    print(f"   좌표없음:   {no_keypoints}개")
    print(f"   저장 위치:  {SAVE_DIR}")
    print(f"{'='*50}")
    print(f"\n저장 구조:")
    print(f"  {SAVE_DIR}/")
    print(f"    train/CARRY/P1_66221823911-CARRY.npy")
    print(f"    val/CARRY/P5_XXXXX-CARRY.npy")
    print(f"    test/CARRY/P42_XXXXX-CARRY.npy")

if __name__ == '__main__':
    main()
