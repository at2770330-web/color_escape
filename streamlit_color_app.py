import os
import base64
import streamlit as st
import pickle
import pandas as pd
import numpy as np


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def set_background(image_file, overlay_alpha=0.06, contrast=1.1, saturate=1.1):
    if not os.path.exists(image_file):
        return
    bin_str = get_base64(image_file)
    page_bg_img = f"""
    <style>
    /* Full-viewport fixed background layer using ::before to avoid clipping */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        background-image: linear-gradient(rgba(255,255,255,{overlay_alpha}), rgba(255,255,255,{overlay_alpha})), url("data:image/webp;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        filter: contrast({contrast}) saturate({saturate});
        opacity: 1;
    }}
    .stApp .main {{
        background-color: rgba(255,255,255,0.55);
        backdrop-filter: blur(10px);
        padding: 1rem 1.25rem;
        border-radius: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.6);
        position: relative;
    }}
    /* Responsive adjustments for 13-inch / narrow screens */
    @media (max-width: 1366px) {{
        .stApp::before {{
            background-position: center center;
            background-size: cover;
            filter: contrast({contrast - 0.05}) saturate({saturate - 0.05});
        }}
        .stApp .main {{
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }}
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Color Trading Predictor", layout="centered")

    st.sidebar.header("Predictor Background")
    default_image = "color_pink.webp"
    image_file = default_image
    if not os.path.exists(image_file):
        st.sidebar.warning(f"Background image not found: {image_file}")

    overlay_alpha = st.sidebar.slider("Overlay opacity", 0.0, 0.5, 0.06, 0.01)
    contrast = st.sidebar.slider("Contrast", 0.5, 1.5, 1.1, 0.01)
    saturate = st.sidebar.slider("Saturation", 0.5, 1.5, 1.1, 0.01)
    show_snow = st.sidebar.checkbox("Show snow animation", value=True)

    set_background(default_image, overlay_alpha=overlay_alpha, contrast=contrast, saturate=saturate)

    model_path = "color_model.pkl"
    model = None
    if os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, "rb"))
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    else:
        st.sidebar.warning(f"Model not found at {model_path}")

    if model is not None:
        acc = f"{getattr(model, 'oob_score_', None):.2%}" if hasattr(model, 'oob_score_') else "Unavailable"
        features = ", ".join(getattr(model, 'feature_names_in_', [])) if hasattr(model, 'feature_names_in_') else "Unknown"
        st.sidebar.write("Model accuracy:", acc)
        st.sidebar.write("Features:", features)

    st.title("🎨 Color Trading Predictor 🎨")

    period_str = st.text_input("Enter Period ID:", value="20260527100010983")
    try:
        period = int(period_str)
    except Exception:
        st.error("Period must be an integer (e.g. 20260527100010983)")
        period = None
    number = st.number_input("Enter Number:", min_value=0, max_value=9, value=5)
    size = st.selectbox("Select Size:", [0, 1], format_func=lambda x: "Big" if x == 1 else "Small")
    size_label = "Big" if size == 1 else "Small"

    st.write("Selected values:", f"Period={period}, Number={number}, Size={size_label}")

    label_map = {0: "RED", 1: "GREEN", 2: "BLUE", 3: "YELLOW"}

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded. Place color_model.pkl next to this script.")
        elif period is None:
            st.error("Invalid Period. Enter a numeric Period value.")
        elif period < 20260527100010983:
            st.error("Period must be >= 20260527100010983")
        else:
            X = pd.DataFrame([[period, number, size]], columns=['Period', 'Number', 'BigSmall'])
            try:
                pred = model.predict(X)[0]
                result = label_map.get(pred, f"Class {pred}")
                st.write("Prediction value:", result)
                if pred == 0:
                    st.error(f"Prediction: {result}")
                else:
                    st.success(f"Prediction: {result}")
            except Exception as e:
                st.error("Prediction failed")
                st.write(str(e))

    if show_snow:
        st.snow()


if __name__ == "__main__":
    main()
