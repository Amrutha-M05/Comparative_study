


# 🧠 Comparative Evaluation of Standard CNNs with Explainability on LAG Dataset

This project performs a **systematic comparison of popular CNN architectures** for **glaucoma detection** using the **LAG (Labelled Anterior Glaucoma) dataset**, with an added focus on **model explainability (XAI)**.

It is designed to be **modular, extensible, and research-friendly**, allowing easy integration of multiple models and explainability techniques.

## Dataset

The LAG dataset is not included in this repository due to size constraints.

Download it from:
[https://www.kaggle.com/datasets/toaharahmanratul/lag-dataset]

---

## 🚀 Supported Models

The framework supports plugging in and comparing the following architectures:

* VGG16
* ResNet50
* DenseNet121
* MobileNet
* EfficientNet-B0

---

## 🔍 Explainability (XAI)

Planned / Supported explainability methods:

* Grad-CAM
* Grad-CAM++
* Integrated Gradients

These help interpret **where the model is focusing** when making predictions.

---

## 📂 Project Structure

```text
glaucoma_cnn_lag/
│
├── LAG/
│   ├── test/
│   │   ├── glaucoma/
│   │   │   ├── image/
│   │   │   └── attention_map/
│   │   └── non_glaucoma/
│   │       ├── image/
│   │       └── attention_map/
│   │
│   ├── train/
│   │   ├── glaucoma/
│   │   │   ├── image/
│   │   │   └── attention_map/
│   │   └── non_glaucoma/
│   │       ├── image/
│   │       └── attention_map/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── xai/
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── metrics.py
│   ├── compare_models.py
│   └── main.py
│
├── generate_xai.py
├── generate_xai_missclassified.py
├── run_all_models.py
└── requirements.txt
```

---

## ✅ Quick Sanity Check

Make sure your environment is set up correctly:

```bash
python -c "import src; print('OK')"
```

---

## ⚙️ How to Run

### 🔹 Step 1: Test Pipeline

```bash
python -m src.main
```

---

### 🔹 Step 2: Full Experiment (All Models)

```bash
python run_all_models.py
```

---

### 🔹 Step 3: Compare Model Performance

```bash
pip install tabulate
python src/compare_models.py
```

---

### 🔹 Step 4: Generate Explainability Outputs

```bash
python generate_xai.py
```

For analyzing misclassified samples:

```bash
python generate_xai_missclassified.py
```

---

## 📊 Outputs

All results are stored in:

```text
outputs/
├── checkpoints/   # Saved model weights
├── logs/          # Training logs
├── predictions/   # Model outputs
└── xai/           # Heatmaps & explanations
```


---

## 📊 Model Performance Comparison

| Model           | Accuracy   | Sensitivity | Specificity | Precision  | F1 Score   | F2 Score   | ROC-AUC    | PR-AUC     | TN  | FP | FN | TP  |
| --------------- | ---------- | ----------- | ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | --- | -- | -- | --- |
| EfficientNet-B0 | 96.29%     | 94.23%      | 98.35%      | 98.28%     | 96.21%     | 95.01%     | 0.9935     | 0.9941     | 477 | 8  | 28 | 457 |
| DenseNet121     | **97.01%** | 94.85%      | **99.18%**  | **99.14%** | **96.94%** | 95.67%     | **0.9934** | **0.9951** | 481 | 4  | 25 | 460 |
| ResNet50        | 96.70%     | 94.85%      | 98.56%      | 98.50%     | 96.64%     | 95.55%     | 0.9909     | 0.9899     | 478 | 7  | 25 | 460 |
| VGG16           | 94.23%     | 95.88%      | 92.58%      | 92.81%     | 94.32%     | 95.25%     | 0.9879     | 0.9881     | 449 | 36 | 20 | 465 |
| MobileNetV2     | 95.67%     | **97.11%**  | 94.23%      | 94.39%     | 95.73%     | **96.56%** | 0.9857     | 0.9796     | 457 | 28 | 14 | 471 |

---

## 🏆 Key Insights

* **Best Overall Model:** DenseNet121 (highest accuracy & precision)
* **Best Sensitivity:** MobileNetV2 (detects glaucoma cases best)
* **Best Specificity:** DenseNet121 (fewest false positives)
* **Most Balanced Performance:** ResNet50 / EfficientNet-B0


---

## 🎯 How to Interpret Heatmaps (Grad-CAM)

### 🧠 What You’re Seeing

Using Grad-CAM:

* 🔴 Red / Yellow → Important regions (model attention)
* 🔵 Blue → Ignored regions

---

### ✅ Good Model Behavior

* Focus on **optic disc and cup**
* Clinically meaningful regions
* Indicates reliable learning

---

### ❌ Bad Model Behavior

* Focus on **background / irrelevant regions**
* Suggests dataset bias or overfitting
* Unreliable predictions

---

### 🔴 Error Analysis Insight

* **False Negatives**: Model fails to attend to optic cup → misses glaucoma
* **False Positives**: Model attends to noise/artifacts → incorrect detection

---

## 🧪 Experimental Goals

* Compare CNN architectures on medical imaging
* Evaluate trade-offs:

  * Accuracy vs Efficiency
  * Performance vs Interpretability
* Analyze model decision-making using XAI

---

## 🔧 Extending the Project

You can easily:

* Add new CNN models in `models.py`
* Plug in new XAI techniques
* Modify training strategies
* Add custom evaluation metrics

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```


---

## ⭐ Future Work

* Add Grad-CAM++ and Integrated Gradients support
* Hyperparameter tuning
* Cross-validation experiments
* Attention-based architectures (ViTs)
* Clinical validation pipeline

