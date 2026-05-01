"""
Rice Leaf Disease Classifier — Gradio App
Compatible with Python 3.13 + TensorFlow 2.21 + Gradio 5.x
"""

import os
import numpy as np
import gradio as gr
from PIL import Image

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]
MODEL_PATH  = "best_model.keras"

CLASS_INFO = {
    "Bacterial Leaf Blight": {
        "emoji": "🦠",
        "cause": "Xanthomonas oryzae pv. oryzae (bacterium)",
        "symptoms": "Water-soaked lesions turning brown and dry; wilting in severe cases.",
        "treatment": "Use resistant varieties, copper-based bactericides, improve drainage.",
    },
    "Brown Spot": {
        "emoji": "🟤",
        "cause": "Cochliobolus miyabeanus (fungus)",
        "symptoms": "Small oval brown spots with yellow halos on leaves.",
        "treatment": "Apply Mancozeb / Propiconazole fungicides; ensure proper nutrition.",
    },
    "Leaf Smut": {
        "emoji": "⚫",
        "cause": "Entyloma oryzae (fungus)",
        "symptoms": "Small reddish-brown spots turning black and powdery.",
        "treatment": "Use disease-free seeds; apply fungicides; practice crop rotation.",
    },
}

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading model…")
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    model = None
    print(f"❌ Model load failed: {e}")


# ── Predict function ──────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return {c: 0.0 for c in CLASS_NAMES}, "⚠️ Please upload an image."

    if model is None:
        return {c: 0.0 for c in CLASS_NAMES}, (
            "❌ Model not found.\n\nPlease upload `best_model.keras` to this Space."
        )

    # Preprocess
    img = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    probs      = model.predict(arr, verbose=0)[0]
    idx        = int(np.argmax(probs))
    label      = CLASS_NAMES[idx]
    confidence = float(probs[idx]) * 100
    info       = CLASS_INFO[label]

    detail = (
        f"## {info['emoji']} {label} — {confidence:.1f}% confidence\n\n"
        f"**🔬 Cause:** {info['cause']}\n\n"
        f"**🍃 Symptoms:** {info['symptoms']}\n\n"
        f"**💊 Treatment:** {info['treatment']}"
    )

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return prob_dict, detail


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🌾 Rice Leaf Disease Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🌾 Rice Leaf Disease Classifier
        Upload a rice leaf image to detect:
        **Bacterial Leaf Blight** · **Brown Spot** · **Leaf Smut**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Rice Leaf Image", type="numpy")
            submit_btn  = gr.Button("🔍 Classify Disease", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output  = gr.Label(label="Prediction Probabilities", num_top_classes=3)
            detail_output = gr.Markdown()

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, detail_output],
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, detail_output],
    )

    gr.Markdown(
        "---\n"
        "**Model:** Custom CNN · **Dataset:** 120 rice leaf images (224×224 RGB) · "
        "**Classes:** 3"
    )

if __name__ == "__main__":
    demo.launch()