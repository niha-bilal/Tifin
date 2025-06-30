import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load Fine-tuned model
finetuned_model = AutoModelForSequenceClassification.from_pretrained("./Tifin/trained_model")
finetuned_tokenizer = AutoTokenizer.from_pretrained("./Tifin/trained_model")

# Load BERT base model
bert_model = AutoModelForSequenceClassification.from_pretrained("./intent_model")
bert_tokenizer = AutoTokenizer.from_pretrained("./intent_model")

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_model.to(device).eval()
bert_model.to(device).eval()

# Label mapping
id_to_label = {
    0: "100_NIGHT_TRIAL_OFFER",
    1: "ABOUT_SOF_MATTRESS",
    2: "CANCEL_ORDER",
    3: "CHECK_PINCODE",
    4: "COD",
    5: "COMPARISON",
    6: "DELAY_IN_DELIVERY",
    7: "DISTRIBUTORS",
    8: "EMI",
    9: "ERGO_FEATURES",
    10: "LEAD_GEN",
    11: "MATTRESS_COST",
    12: "OFFERS",
    13: "ORDER_STATUS",
    14: "ORTHO_FEATURES",
    15: "PILLOWS",
    16: "PRODUCT_VARIANTS",
    17: "RETURN_EXCHANGE",
    18: "SIZE_CUSTOMIZATION",
    19: "WARRANTY",
    20: "WHAT_SIZE_TO_ORDER"
}

def infer_intent(text):
    # Fine-tuned model prediction
    inputs_ft = finetuned_tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
    ).to(device)

    with torch.no_grad():
        outputs_ft = finetuned_model(**inputs_ft)
    pred_ft = torch.argmax(outputs_ft.logits, dim=-1).item()
    label_ft = id_to_label.get(pred_ft, "Unknown")

    # BERT base model prediction
    inputs_bert = bert_tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
    ).to(device)

    with torch.no_grad():
        outputs_bert = bert_model(**inputs_bert)
    pred_bert = torch.argmax(outputs_bert.logits, dim=-1).item()
    label_bert = id_to_label.get(pred_bert, "Unknown")

    return label_ft, label_bert

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 SOF Intent Classifier (Fine-tuned + BERT)")

    inp = gr.Textbox(label="Enter your sentence")
    btn = gr.Button("Predict")

    out1 = gr.Textbox(label="Fine-tuned Model Prediction")
    out2 = gr.Textbox(label="BERT Base Model Prediction")

    btn.click(fn=infer_intent, inputs=inp, outputs=[out1, out2])

demo.launch()
