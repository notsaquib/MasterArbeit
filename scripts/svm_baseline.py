import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm
import datetime

# Paths to feature data (update as needed)
DATASET_TAG = "CHARADES_MA52_MMAct_PKU"  # Change this depending on the datasets used
CACHE_DIR = "action_prediction_baseline/cached/features"

def load_split(split):
    path = os.path.join(CACHE_DIR, f"{split}_{DATASET_TAG}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_class_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-score')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1-score by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("📦 Loading datasets...")
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")

    print(train_data.keys())

    print("📊 Processing features...")
    X_train = np.stack(train_data["features"])
    y_train = np.array(train_data["encoded_labels"])

    X_val = np.stack(val_data["features"])
    y_val = np.array(val_data["encoded_labels"])

    X_test = np.stack(test_data["features"])
    y_test = np.array(test_data["encoded_labels"])

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("🧠 Training SVM model...")
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True))
    model.fit(X_train, y_train)

    print("✅ Evaluating model on Validation set")
    val_preds = model.predict(X_val)
    val_acc = round(accuracy_score(y_val, val_preds),4)
    print("Validation Accuracy:", val_acc)
    print(classification_report(y_val, val_preds))

    print("✅ Evaluating model on Test set")
    test_preds = model.predict(X_test)
    test_acc = round(accuracy_score(y_test, test_preds),4)
    print("Test Accuracy:", test_acc)
    print(classification_report(y_test, test_preds))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    print(cm)

    # Class names for plotting
    num_classes = len(np.unique(y_test))
    class_names = [f"Class {i}" for i in range(num_classes)]

    # Visualize
    plot_confusion_matrix(cm, class_names)
    plot_class_metrics(y_test, test_preds, class_names)

    save_time = datetime.datetime.now().strftime("%m%d_%H%M%S")

    # Save model
    os.makedirs("action_prediction_baseline/models", exist_ok=True)
    model_path = f"action_prediction_baseline/models/svm_{DATASET_TAG}_TestAcc_{test_acc}_ValAcc_{val_acc}_{save_time}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    main()
