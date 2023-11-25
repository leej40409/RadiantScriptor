
import streamlit as st
import requests

# Function to call the Hugging Face model
def query_huggingface_model(prompt):
    API_TOKEN = "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"  # Replace with your Hugging Face API token
    API_URL = "https://api-inference.huggingface.co/models/MariamAde/Mistral_finetuned_Base2"  # Replace with your model's API URL

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Streamlit interface
def main():
    st.title("My Fine-tuned Model Demo")

    # User input
    user_input = st.text_area("Enter your text here", "")

    # Button to make the prediction
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            response = query_huggingface_model(user_input)
            if "error" in response:
                st.error(response["error"])
            else:
                st.success("Prediction Success")
                st.write(response)  # Modify this based on how your model's response is structured

if __name__ == "__main__":
    main()






























##############################################################################
# from transformers import pipeline
# import streamlit as st

# # Set the page configuration
# st.set_page_config(page_title="RadiantScriptor")

# # Initialize the pipeline for text generation
# # Note: The pipeline will automatically handle the device allocation (CPU/GPU)
# pipe = pipeline("text-generation", model="MariamAde/Mistral_finetuned_Base2")

# # Function to generate the report
# def generate_report(labels):
#     # Generate output using the pipeline
#     output = pipe(labels, max_new_tokens=100, do_sample=True)
#     # The output is a list of dictionaries, each containing the generated text
#     sentences = output[0]['generated_text']
#     return sentences

# # Streamlit interface for user interaction
# st.title("Radiology Report Generator")
# labels = st.text_input("Enter Finding Labels:")

# # Button to generate the report
# if st.button("Generate Report"):
#     report = generate_report(labels)
#     st.text_area("Generated Report:", value=report, height=300)




################################################################################
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import streamlit as st
# import torch

# # Check for GPU availability, default to CPU if not available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Set the page configuration
# st.set_page_config(page_title="RadiantScriptor")

# # Caching the model loading function to improve performance
# @st.experimental_singleton
# def get_model():
#     # Load tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained("MariamAde/Mistral_finetuned_Base2") #MariamAde/Mistral_finetuned_Base2
#     model = AutoModelForCausalLM.from_pretrained("MariamAde/Mistral_finetuned_Base2",low_cpu_mem_usage=True, device_map="cpu").to(device)
#     return tokenizer, model

# # Load model and tokenizer with a spinner
# with st.spinner('Model is being loaded..'):
#       tokenizer, model = get_model()

# # Function to generate the report
# def generate_report(labels):
#     # Tokenize the input labels
#     inputs = tokenizer(labels, return_tensors="pt").to(device)
#     # Generate output using the model
#     output = model.generate(**inputs, max_new_tokens=100, do_sample=True)
#     # Decode the output sentences
#     sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
#     return sentences

# # Streamlit interface for user interaction
# st.title("Radiology Report Generator")
# labels = st.text_input("Enter Finding Labels:")

# # Button to generate the report
# if st.button("Generate Report"):
#     report = generate_report(labels)
#     st.text_area("Generated Report:", value=report, height=300)

