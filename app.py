from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import torch

# Check for GPU availability, default to CPU if not available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the page configuration
st.set_page_config(page_title="RadiantScriptor")

# Caching the model loading function to improve performance
@st.cache_data(allow_output_mutation=True)
def get_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

# Load model and tokenizer with a spinner
with st.spinner('Model is being loaded..'):
      tokenizer, model = get_model()

# Function to generate the report
def generate_report(labels):
    # Tokenize the input labels
    inputs = tokenizer(labels, return_tensors="pt").to(device)
    model.to(device)
    # Generate output using the model
    output = model.generate(**inputs, max_new_tokens=100, do_sample=True)
    # Decode the output sentences
    sentences = tokenizer.batch_decode(output[0])
    return sentences

# Streamlit interface for user interaction
st.title("Radiology Report Generator")
labels = st.text_input("Enter Finding Labels:")

# Button to generate the report
if st.button("Generate Report"):
    report = generate_report(labels)
    st.text_area("Generated Report:", value=report, height=300)
