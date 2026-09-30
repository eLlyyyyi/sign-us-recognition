import csv
import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_PATH = os.environ.get("SIGNUS_ASL_FEATURE_DIR", "data/asl/features")
TEST_PATH = os.path.join(BASE_PATH, "test")
MODEL_DIR = os.environ.get("SIGNUS_ASL_MODEL_DIR", "artifacts/asl_model")
OUT_DIR = os.path.join(MODEL_DIR, "error_analysis")

FEATURES = 260
MIN_FRAMES = 5
BATCH_SIZE = 32


def extract_person_id(file_name):
    base = os.path.splitext(file_name)[0]
    parts = base.split("_")
    for part in parts:
        if part.startswith("P") and part[1:].isdigit():
            return part
    return None


def load_test():
    data_list = []
    for word_folder in sorted(os.listdir(TEST_PATH)):
        word_path = os.path.join(TEST_PATH, word_folder)
        if not os.path.isdir(word_path):
            continue

        for file_name in sorted(os.listdir(word_path)):
            if not file_name.endswith(".npy"):
                continue

            path = os.path.join(word_path, file_name)
            data = np.load(path)
            if data.ndim != 2 or data.shape[1] != FEATURES or len(data) < MIN_FRAMES:
                continue

            data_list.append(
                {
                    "data": data.astype(np.float32),
                    "label": word_folder,
                    "person": extract_person_id(file_name) or "",
                    "file": file_name,
                    "path": path,
                }
            )
    return data_list


def aug_speed_up(d):
    n = max(1, int(len(d) * 0.75))
    return d[np.linspace(0, len(d) - 1, n).astype(int)]


def aug_speed_down(d):
    n = int(len(d) * 1.25)
    return d[np.linspace(0, len(d) - 1, n).astype(int)]


def to_x(samples, max_frames):
    clipped = [item["data"][:max_frames] for item in samples]
    return pad_sequences(
        clipped,
        maxlen=max_frames,
        dtype="float32",
        padding="post",
        value=0.0,
    )


def predict_tta(model, samples, max_frames):
    variants = [
        samples,
        [{**item, "data": aug_speed_up(item["data"])} for item in samples],
        [{**item, "data": aug_speed_down(item["data"])} for item in samples],
    ]

    probs = []
    for variant in variants:
        probs.append(
            model.predict(
                to_x(variant, max_frames),
                batch_size=BATCH_SIZE,
                verbose=0,
            )
        )
    return np.mean(probs, axis=0)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = np.array(joblib.load(os.path.join(MODEL_DIR, "asl_classes_260.pkl")))
    meta = joblib.load(os.path.join(MODEL_DIR, "asl_meta_260.pkl"))
    max_frames = int(meta["MAX_FRAMES"])

    samples = load_test()
    y_true = np.array([np.where(classes == item["label"])[0][0] for item in samples])

    model_paths = [
        os.path.join(MODEL_DIR, f"asl_fold{i}_260.h5")
        for i in range(1, 5)
    ]
    models = [tf.keras.models.load_model(path, compile=False) for path in model_paths]

    ensemble_tta_probs = np.mean(
        [predict_tta(model, samples, max_frames) for model in models],
        axis=0,
    )
    y_pred = np.argmax(ensemble_tta_probs, axis=1)
    top3 = np.argsort(ensemble_tta_probs, axis=1)[:, -3:][:, ::-1]

    correct = y_true == y_pred
    top3_correct = np.array([y_true[i] in top3[i] for i in range(len(y_true))])
    accuracy = float(np.mean(correct))
    top3_accuracy = float(np.mean(top3_correct))

    prediction_rows = []
    for i, item in enumerate(samples):
        prediction_rows.append(
            {
                "file": item["file"],
                "person": item["person"],
                "true_label": item["label"],
                "pred_label": classes[y_pred[i]],
                "correct": int(correct[i]),
                "top1_conf": float(ensemble_tta_probs[i, y_pred[i]]),
                "top2_label": classes[top3[i, 1]],
                "top2_conf": float(ensemble_tta_probs[i, top3[i, 1]]),
                "top3_label": classes[top3[i, 2]],
                "top3_conf": float(ensemble_tta_probs[i, top3[i, 2]]),
                "true_in_top3": int(top3_correct[i]),
                "path": item["path"],
            }
        )

    write_csv(
        os.path.join(OUT_DIR, "predictions_ensemble_tta.csv"),
        prediction_rows,
        [
            "file",
            "person",
            "true_label",
            "pred_label",
            "correct",
            "top1_conf",
            "top2_label",
            "top2_conf",
            "top3_label",
            "top3_conf",
            "true_in_top3",
            "path",
        ],
    )

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        confusion[true_idx, pred_idx] += 1

    with open(os.path.join(OUT_DIR, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label"] + list(classes))
        for i, label in enumerate(classes):
            writer.writerow([label] + confusion[i].tolist())

    class_rows = []
    for idx, label in enumerate(classes):
        support = int(np.sum(y_true == idx))
        class_correct = int(confusion[idx, idx])
        wrong_counts = confusion[idx].copy()
        wrong_counts[idx] = 0
        top_wrong_idx = int(np.argmax(wrong_counts))
        class_rows.append(
            {
                "label": label,
                "support": support,
                "correct": class_correct,
                "accuracy": class_correct / support if support else 0.0,
                "top_wrong_label": classes[top_wrong_idx] if wrong_counts[top_wrong_idx] else "",
                "top_wrong_count": int(wrong_counts[top_wrong_idx]),
            }
        )
    class_rows.sort(key=lambda row: (row["accuracy"], -row["support"], row["label"]))
    write_csv(
        os.path.join(OUT_DIR, "class_accuracy.csv"),
        class_rows,
        ["label", "support", "correct", "accuracy", "top_wrong_label", "top_wrong_count"],
    )

    pair_counts = {}
    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx == pred_idx:
            continue
        key = (classes[true_idx], classes[pred_idx])
        pair_counts[key] = pair_counts.get(key, 0) + 1

    top_confusion_rows = [
        {"true_label": true, "pred_label": pred, "count": count}
        for (true, pred), count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    write_csv(
        os.path.join(OUT_DIR, "top_confusions.csv"),
        top_confusion_rows,
        ["true_label", "pred_label", "count"],
    )

    person_rows = []
    persons = sorted({item["person"] for item in samples}, key=lambda p: int(p[1:]) if p[1:].isdigit() else 999)
    for person in persons:
        mask = np.array([item["person"] == person for item in samples])
        support = int(np.sum(mask))
        person_rows.append(
            {
                "person": person,
                "support": support,
                "correct": int(np.sum(correct[mask])),
                "accuracy": float(np.mean(correct[mask])) if support else 0.0,
                "top3_accuracy": float(np.mean(top3_correct[mask])) if support else 0.0,
            }
        )
    person_rows.sort(key=lambda row: row["accuracy"])
    write_csv(
        os.path.join(OUT_DIR, "person_accuracy.csv"),
        person_rows,
        ["person", "support", "correct", "accuracy", "top3_accuracy"],
    )

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ASL 260 Error Analysis - Ensemble + TTA\n")
        f.write(f"Samples: {len(samples)}\n")
        f.write(f"Classes: {len(classes)}\n")
        f.write(f"MAX_FRAMES: {max_frames}\n")
        f.write(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)\n")
        f.write(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy * 100:.1f}%)\n\n")

        f.write("Top 15 Confusions:\n")
        for row in top_confusion_rows[:15]:
            f.write(f"  {row['true_label']} -> {row['pred_label']}: {row['count']}\n")

        f.write("\nLowest 15 Class Accuracies:\n")
        for row in class_rows[:15]:
            f.write(
                f"  {row['label']}: {row['accuracy']:.3f} "
                f"({row['correct']}/{row['support']}), "
                f"top wrong={row['top_wrong_label']} ({row['top_wrong_count']})\n"
            )

        f.write("\nLowest Person Accuracies:\n")
        for row in person_rows:
            f.write(
                f"  {row['person']}: {row['accuracy']:.3f} "
                f"({row['correct']}/{row['support']}), "
                f"top3={row['top3_accuracy']:.3f}\n"
            )

    print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy * 100:.1f}%)")
    print(f"Saved error analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()
