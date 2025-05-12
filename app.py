import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer from local path
model_path = r"C:\Users\w\OneDrive\Desktop\Translation app\translation_model1"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Streamlit UI
st.title("Arabic to English Translator")
st.write("This app translates Arabic sentences to English using a fine-tuned model.")

# Input
arabic_text = st.text_area("Enter Arabic text", "")

if st.button("Translate"):
    if arabic_text.strip():
        try:
            # Tokenize input and move to device
            inputs = tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Generate translation
            with torch.no_grad():
                translated = model.generate(**inputs)
            
            # Decode output
            english_text = tokenizer.decode(translated[0], skip_special_tokens=True)

            # Show result
            st.success("Translation:")
            st.write(english_text)
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
    else:
        st.warning("Please enter some Arabic text.")