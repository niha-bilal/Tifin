# 🧠 Intent Detection with Transformer Models

Identify user intents from short text inputs using traditional and transformer-based models (BERT, Pythia). This project combines data augmentation, model fine-tuning, and an interactive Gradio interface.

---

## 📁 Repository Structure



├── 1.py                         # Initial or test script  
├── app.py                       # Gradio app interface  
├── bert.py                      # BERT model training  
├── train.py                     # General training logic  
├── label_classes.npy            # Encoded intent classes  
├── prompt_training_examples.json # Augmented training examples  
├── README.md                    # This file  


---

## 🚀 Features

- Multi-class classification across 21 intent labels
- Fine-tuned BERT and Pythia-70M models
- Augmented dataset: 328 → 1004 examples
- Real-time Gradio interface
- Custom PyTorch training for flexibility

---

## 📦 Installation & Setup

1. Clone this repository:

git clone https://github.com/niha-bilal/Tifin.git
cd Tifin
2. Install dependencies:
pip install -r requirements.txt
3. Run the app:
python app.py

🧪 Model Accuracy
| Model        | Accuracy | Highlights                        |
| ------------ | -------- | --------------------------------- |
| Pythia-70M   | 92–93%   | Strong semantic performance       |
| BERT         | 90–92%   | Flexible, custom PyTorch training |
| SVM (TF-IDF) | 71–92%   | Good with augmentation            |
| Naive Bayes  | \~50%    | Lightweight but too shallow       |

🧰 How to Train
- Train BERT:python bert.py
- Train Pythia or other models:
- python train.py
Models are saved to:

./Tifin/trained_model/ (Pythia)

./intent_model/ (BERT)


🌐 Gradio Web Interface


To test predictions:


python app.py

You’ll get:

Running on local URL: http://127.0.0.1:7860

Running on public URL: https://xxxx.gradio.live


📈 Recommendations

- Use weighted loss or SMOTE to balance class distribution

- Add feedback/error correction in Gradio interface

- Test with optimized models like DistilBERT

- Apply active learning and Ray Tune for fine-tuning
