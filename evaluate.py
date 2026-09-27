"""
Evaluate the trained model on the 400 clips it NEVER trained on (ESC-50 fold 5).

It answers three questions:
  1. Which sounds does the model mix up?        -> confusion_matrix.png
  2. How sure is it when it's right vs wrong?   -> confidence_histogram.png
  3. Below what confidence should we say
     "Unknown sound" instead of guessing?       -> threshold table (printed + eval_report.json)

Run it with:
    python -m modal run evaluate.py

Results are saved in an "eval_results" folder next to this file.
"""

import json
from pathlib import Path

import modal

app = modal.App("audio-cnn-evaluate")

# Same image steps as train.py (so Modal reuses the cached layers),
# plus matplotlib for drawing the charts.
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
    .run_commands(
        [
            "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
            "cd /tmp && unzip esc50.zip",
            "mkdir -p /opt/esc50-data",
            "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
            "rm -rf /tmp/esc50.zip /tmp/ESC-50-master",
        ]
    )
    .pip_install("matplotlib")
    .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model")

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(13)]  # 0.20, 0.25, ... 0.80


# ---------------------------------------------------------------------------
# Pure analysis functions (only need numpy / matplotlib, no model)
# ---------------------------------------------------------------------------


def build_confusion_matrix(y_true, y_pred, num_classes):
    """cm[true][pred] = how many clips of class `true` were predicted as `pred`."""
    import numpy as np

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def top_confusions(cm, classes, k=10):
    """The k most common mistakes: (true class, predicted class, count)."""
    mistakes = []
    for t in range(len(classes)):
        for p in range(len(classes)):
            if t != p and cm[t, p] > 0:
                mistakes.append(
                    {"true": classes[t], "predicted": classes[p], "count": int(cm[t, p])}
                )
    mistakes.sort(key=lambda m: m["count"], reverse=True)
    return mistakes[:k]


def per_class_accuracy(cm, classes):
    """Accuracy for each class, worst first."""
    rows = []
    for i, name in enumerate(classes):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        rows.append(
            {
                "class": name,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )
    rows.sort(key=lambda r: r["accuracy"])
    return rows


def threshold_table(confidences, is_correct, thresholds=THRESHOLDS):
    """
    For each threshold t, pretend we answer "Unknown" whenever confidence < t.

    - answered:          % of clips we still give an answer for
    - accuracy_answered: accuracy on the clips we DID answer
    - wrong_caught:      % of wrong guesses that turned into "Unknown" (good)
    - right_lost:        % of right guesses that turned into "Unknown" (bad)
    """
    import numpy as np

    conf = np.asarray(confidences)
    ok = np.asarray(is_correct, dtype=bool)
    n_right, n_wrong = ok.sum(), (~ok).sum()

    table = []
    for t in thresholds:
        answered = conf >= t
        n_answered = answered.sum()
        table.append(
            {
                "threshold": t,
                "answered": float(n_answered / len(conf)),
                "accuracy_answered": float((ok & answered).sum() / n_answered) if n_answered else 0.0,
                "wrong_caught": float((~ok & ~answered).sum() / n_wrong) if n_wrong else 0.0,
                "right_lost": float((ok & ~answered).sum() / n_right) if n_right else 0.0,
            }
        )
    return table


def recommend_threshold(table, max_right_lost=0.05):
    """
    Highest threshold that still keeps at least 95% of the correct answers.
    Higher threshold = catches more mistakes, but we refuse to throw away
    more than 5% of the answers the model gets right.
    """
    ok_rows = [row for row in table if row["right_lost"] <= max_right_lost]
    best = max(ok_rows, key=lambda row: row["threshold"]) if ok_rows else table[0]
    return best


def plot_confusion_matrix(cm, classes):
    """Returns PNG bytes of a 50x50 confusion matrix."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [c.replace("_", " ") for c in classes]
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title("Confusion matrix on ESC-50 fold 5 (400 unseen clips)", fontsize=14)

    # Write the count in every non-empty cell. Mistakes (off the diagonal) in red.
    for t in range(cm.shape[0]):
        for p in range(cm.shape[1]):
            if cm[t, p] > 0:
                color = "white" if t == p else "crimson"
                ax.text(p, t, cm[t, p], ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def plot_confidence_histogram(confidences, is_correct, threshold):
    """Returns PNG bytes: confidence when right (green) vs when wrong (red)."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conf = np.asarray(confidences)
    ok = np.asarray(is_correct, dtype=bool)
    bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(conf[ok], bins=bins, alpha=0.7, color="seagreen", label=f"Correct ({ok.sum()})")
    ax.hist(conf[~ok], bins=bins, alpha=0.7, color="crimson", label=f"Wrong ({(~ok).sum()})")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Suggested threshold = {threshold:.2f}")
    ax.set_xlabel("Model's confidence in its top guess")
    ax.set_ylabel("Number of clips")
    ax.set_title("How sure is the model when it's right vs. wrong?")
    ax.legend()

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def analyse(y_true, y_pred, confidences, classes):
    """Runs every analysis and returns (report dict, {filename: png bytes})."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_correct = y_true == y_pred

    cm = build_confusion_matrix(y_true, y_pred, len(classes))
    table = threshold_table(confidences, is_correct)
    best = recommend_threshold(table)

    conf = np.asarray(confidences)
    report = {
        "num_clips": int(len(y_true)),
        "accuracy": float(is_correct.mean()),
        "mean_confidence_when_right": float(conf[is_correct].mean()) if is_correct.any() else 0.0,
        "mean_confidence_when_wrong": float(conf[~is_correct].mean()) if (~is_correct).any() else 0.0,
        "recommended_threshold": best["threshold"],
        "threshold_table": table,
        "top_confusions": top_confusions(cm, classes),
        "worst_classes": per_class_accuracy(cm, classes)[:10],
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }
    images = {
        "confusion_matrix.png": plot_confusion_matrix(cm, classes),
        "confidence_histogram.png": plot_confidence_histogram(confidences, is_correct, best["threshold"]),
    }
    return report, images


def print_report(report):
    print(f"\nAccuracy on {report['num_clips']} unseen clips: {report['accuracy']:.2%}")
    print(
        f"Average confidence when RIGHT: {report['mean_confidence_when_right']:.1%} | "
        f"when WRONG: {report['mean_confidence_when_wrong']:.1%}"
    )

    print("\nTop confusions (true -> predicted):")
    for m in report["top_confusions"]:
        print(f"  {m['true']:>18} -> {m['predicted']:<18} x{m['count']}")

    print("\nWorst classes:")
    for r in report["worst_classes"]:
        print(f"  {r['class']:>18}: {r['correct']}/{r['total']} ({r['accuracy']:.0%})")

    print("\nThreshold | answered | accuracy on answered | wrong caught | right lost")
    for row in report["threshold_table"]:
        print(
            f"   {row['threshold']:.2f}   |  {row['answered']:6.1%}  |       {row['accuracy_answered']:6.1%}"
            f"         |    {row['wrong_caught']:6.1%}    |   {row['right_lost']:5.1%}"
        )
    print(f"\nRecommended CONFIDENCE_THRESHOLD for main.py: {report['recommended_threshold']:.2f}")


# ---------------------------------------------------------------------------
# The part that runs the model (on Modal)
# ---------------------------------------------------------------------------


@app.function(image=image, cpu=4.0, volumes={"/models": model_volume}, timeout=60 * 20)
def evaluate():
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torchaudio.transforms as T

    from model import AudioCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("/models/best_model.pth", map_location=device)
    classes = checkpoint["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Must be IDENTICAL to val_transform in train.py, otherwise the model
    # sees differently-shaped "pictures" than it was trained on.
    transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025,
        ),
        T.AmplitudeToDB(),
    )

    esc50_dir = Path("/opt/esc50-data")
    metadata = pd.read_csv(esc50_dir / "meta" / "esc50.csv")
    metadata = metadata[metadata["fold"] == 5]

    y_true, y_pred, confidences = [], [], []
    batch_size = 32

    with torch.no_grad():
        for start in range(0, len(metadata), batch_size):
            rows = metadata.iloc[start : start + batch_size]
            specs = []
            for _, row in rows.iterrows():
                audio, _ = sf.read(esc50_dir / "audio" / row["filename"], dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
                specs.append(transform(waveform))  # (1, 128, time)
                y_true.append(class_to_idx[row["category"]])

            batch = torch.stack(specs).to(device)  # (B, 1, 128, time)
            probs = torch.softmax(model(batch), dim=1)
            conf, pred = probs.max(dim=1)
            y_pred.extend(pred.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

    report, images = analyse(np.array(y_true), np.array(y_pred), np.array(confidences), classes)
    print_report(report)

    files = {name: data for name, data in images.items()}
    files["eval_report.json"] = json.dumps(report, indent=2).encode("utf-8")
    return files


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).parent / "eval_results"
    out_dir.mkdir(exist_ok=True)

    files = evaluate.remote()
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        print(f"Saved {out_dir / name}")