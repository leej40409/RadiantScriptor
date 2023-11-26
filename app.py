
import streamlit as st
import requests

# Set the page configuration
st.set_page_config(page_title="RadiantScriptor")

# Function to call the Hugging Face model API
def query_huggingface_model(prompt):
    API_TOKEN = "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"  
    API_URL = "https://poxj7ux0l7kszkjs.us-east-1.aws.endpoints.huggingface.cloud"  
    
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


st.title("Radiology Report Generator")

# User input for uploading a text file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
user_prompt = ""

if uploaded_file is not None:
    # Read the contents of the uploaded text file
    user_prompt = uploaded_file.read().decode("utf-8")
    # Display the content to the user (optional)
    st.text_area("Uploaded Text:", value=user_prompt, height=150)

if st.button("Generate Report"):
    with st.spinner('Generating report...'):
        # Query the Hugging Face model API
        response = query_huggingface_model(user_prompt)
        if "error" in response:
            st.error(f"Error: {response['error']}")
        else:
            # Assuming the response is a JSON object containing the generated text
            report = response[0]['generated_text']  # Adjust based on the actual response structure
            # Display the report
            st.text_area("Generated Report:", value=report, height=300)

























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

