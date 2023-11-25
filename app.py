from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Set the page configuration
st.set_page_config(page_title="RadiantScriptor")

# Caching the model loading function to improve performance
@st.cache(allow_output_mutation=True)
def get_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

# Load model and tokenizer with a spinner to indicate loading process
with st.spinner('Model is being loaded..'):
      tokenizer, model = get_model()

# Function to generate the radiology report
def generate_report(labels):
    # Tokenize the input labels
    inputs = tokenizer(labels, return_tensors="pt")
    # Generate output using the model
    output = model.generate(**inputs)
    # Decode the output sentences
    sentences = tokenizer.decode(output[0], skip_special_tokens=True)
    return sentences

# Streamlit interface for user interaction
st.title("Radiology Report Generator")

# User input for finding labels
labels = st.text_input("Enter Finding Labels:")

# Button to generate the report
if st.button("Generate Report"):
    # Generate the radiology report
    report = generate_report(labels)
    # Display the report
    st.text_area("Generated Report:", value=report, height=300)
