import argparse
import json
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf


@dataclass
class TrainConfig:
    dataset_dir: str
    output_model: str
    output_labels: str
    seq_len: int = 30
    stride: int = 5
    img_size: int = 0
    use_processed: bool = True
    val_split: float = 0.2
    batch_size: int = 8
    epochs: int = 30
    seed: int = 42


def _list_frame_files(frame_dir: str) -> list[str]:
    if not os.path.isdir(frame_dir):
        return []
    files = [
        fname for fname in os.listdir(frame_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(files)


def _infer_img_size(dataset_dir: str, use_processed: bool) -> int | None:
    for word in sorted(os.listdir(dataset_dir)):
        word_dir = os.path.join(dataset_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for run in sorted(os.listdir(word_dir)):
            run_dir = os.path.join(word_dir, run)
            if not os.path.isdir(run_dir):
                continue
            meta_path = os.path.join(run_dir, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    img_size = int(meta.get("imgSize", 0))
                    if img_size > 0:
                        return img_size
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

            frame_dir = os.path.join(run_dir, "processed" if use_processed else "raw")
            frames = _list_frame_files(frame_dir)
            if not frames:
                continue
            sample = cv2.imread(os.path.join(frame_dir, frames[0]), cv2.IMREAD_GRAYSCALE)
            if sample is not None and sample.shape[0] == sample.shape[1]:
                return int(sample.shape[0])
    return None


def _load_frame(path: str, img_size: int) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return img[..., None]


def _collect_sequences_for_run(frame_dir: str, seq_len: int, stride: int, img_size: int) -> list[np.ndarray]:
    frames = _list_frame_files(frame_dir)
    if len(frames) < seq_len:
        return []

    sequences: list[np.ndarray] = []
    last_start = len(frames) - seq_len
    for start in range(0, last_start + 1, stride):
        seq_frames: list[np.ndarray] = []
        valid = True
        for i in range(start, start + seq_len):
            frame_path = os.path.join(frame_dir, frames[i])
            img = _load_frame(frame_path, img_size)
            if img is None:
                valid = False
                break
            seq_frames.append(img)
        if valid:
            sequences.append(np.stack(seq_frames, axis=0))
    return sequences


def load_dataset(config: TrainConfig) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.isdir(config.dataset_dir):
        raise RuntimeError(f"Dataset folder not found: {config.dataset_dir}")

    word_dirs = [
        d for d in os.listdir(config.dataset_dir)
        if os.path.isdir(os.path.join(config.dataset_dir, d))
    ]
    if not word_dirs:
        raise RuntimeError("No word folders found in dataset directory.")

    word_dirs.sort()
    labels = word_dirs

    img_size = config.img_size
    if img_size <= 0:
        inferred = _infer_img_size(config.dataset_dir, config.use_processed)
        img_size = inferred if inferred else 300

    all_sequences: list[np.ndarray] = []
    all_targets: list[int] = []

    for label_idx, word in enumerate(word_dirs):
        word_dir = os.path.join(config.dataset_dir, word)
        run_dirs = [
            d for d in os.listdir(word_dir)
            if os.path.isdir(os.path.join(word_dir, d))
        ]
        run_dirs.sort()

        for run in run_dirs:
            run_dir = os.path.join(word_dir, run)
            frame_dir = os.path.join(run_dir, "processed" if config.use_processed else "raw")
            sequences = _collect_sequences_for_run(
                frame_dir,
                config.seq_len,
                config.stride,
                img_size,
            )
            if not sequences:
                continue
            all_sequences.extend(sequences)
            all_targets.extend([label_idx] * len(sequences))

    if not all_sequences:
        raise RuntimeError("No sequences found. Check dataset paths and settings.")

    X = np.stack(all_sequences, axis=0)
    y = np.array(all_targets, dtype=np.int32)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y, labels, img_size


def build_model(seq_len: int, img_size: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, img_size, img_size, 1))

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")
    )(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")
    )(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(config: TrainConfig) -> None:
    X, y, labels, img_size = load_dataset(config)

    val_count = max(1, int(len(y) * config.val_split))
    X_train, X_val = X[:-val_count], X[-val_count:]
    y_train, y_val = y[:-val_count], y[-val_count:]

    model = build_model(config.seq_len, img_size, len(labels))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    preds = model.predict(X_val, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    num_classes = len(labels)

    per_class_f1 = []
    for cls in range(num_classes):
        tp = int(np.sum((y_val == cls) & (y_pred == cls)))
        fp = int(np.sum((y_val != cls) & (y_pred == cls)))
        fn = int(np.sum((y_val == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    verdict = "ok"
    if (train_acc - val_acc) > 0.15 and train_acc > 0.85:
        verdict = "likely overfitting"
    elif train_acc < 0.6 and val_acc < 0.6:
        verdict = "likely underfitting"

    os.makedirs(os.path.dirname(config.output_model), exist_ok=True)
    model.save(config.output_model)

    with open(config.output_labels, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Saved model to {config.output_model}")
    print(f"Saved labels to {config.output_labels}")
    print(f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Model verdict: {verdict}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train dynamic sign model from dataset frames.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root (default: dataset)")
    parser.add_argument("--out-model", default=os.path.join("Model", "dynamic_modelV2.h5"))
    parser.add_argument("--out-labels", default=os.path.join("Model", "dynamic_labelsV2.txt"))
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=0, help="0 = auto from meta/images")
    parser.add_argument("--raw", action="store_true", help="Use raw frames instead of processed")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return TrainConfig(
        dataset_dir=args.dataset,
        output_model=args.out_model,
        output_labels=args.out_labels,
        seq_len=args.seq_len,
        stride=args.stride,
        img_size=args.img_size,
        use_processed=not args.raw,
        val_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
