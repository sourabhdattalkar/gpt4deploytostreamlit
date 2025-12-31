import streamlit as st
from huggingface_hub import InferenceClient
from openai import OpenAI
from PIL import Image
import os
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Multi-Model App",
    layout="centered"
)

st.title("🤖 Multi-Model AI App")
st.write("Text-to-Text (OpenAI) & Text-to-Image (HuggingFace)")

# -----------------------------
# Load API Keys
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")


# -----------------------------
# Initialize Clients
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

hf_client = InferenceClient(
    provider="nebius",
    api_key=HF_API_KEY
)

# -----------------------------
# UI Controls
# -----------------------------
model_type = st.selectbox(
    "Select AI Mode",
    ["Text to Text (OpenAI)", "Text to Image (HuggingFace)"]
)

user_prompt = st.text_area(
    "Enter your prompt",
    height=150,
    placeholder="Type your prompt here..."
)

generate_btn = st.button("🚀 Generate")

# -----------------------------
# Logic
# -----------------------------
if generate_btn and user_prompt.strip():

    # ========== TEXT → TEXT ==========
    if model_type == "Text to Text (OpenAI)":
        with st.spinner("Generating text..."):
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            output_text = response.choices[0].message.content

        st.subheader("📝 Generated Text")
        st.write(output_text)

    # ========== TEXT → IMAGE ==========
    elif model_type == "Text to Image (HuggingFace)":
        with st.spinner("Generating image..."):
            image = hf_client.text_to_image(
                user_prompt,
                model="black-forest-labs/FLUX.1-dev"
            )

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            image.save(filename)

        st.subheader("🖼 Generated Image")
        st.image(image, caption=user_prompt)
        st.success(f"Image saved as `{filename}`")

else:
    st.info("Enter a prompt and click Generate")
