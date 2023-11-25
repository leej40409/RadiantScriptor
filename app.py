from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer, MistralForCausalLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model,AutoPeftModelForCausalLM
from transformers import MistralForCausalLM, LlamaTokenizer
import os,torch, wandb, platform, gradio, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
import fire
import streamlit as st

st.set_page_config(page_title= "RadiantScriptor ")
@st.cache_data #(allow_output_mutation=True)


def get_model():
    #device = "cuda" # the device to load the model onto
    tokenizer = AutoTokenizer.from_pretrained("J4Lee/Medalpaca_finetuned_test")
    model = AutoModelForCausalLM.from_pretrained("J4Lee/Medalpaca_finetuned_test", low_cpu_mem_usage=True, device_map="cpu")
    
    return tokenizer, model
with st.spinner('Model is being loaded..'):
      tokenizer, model=get_model()
    

def generate_report(labels): #,model,tokenizer):
    # Tokenize the input labels
    inputs = tokenizer(labels, return_tensors="pt") #.to(device)
    #model.to(device)
    # Generate output using the model
    output = model.generate(**inputs)
    # Decode the output sentences
    sentences = tokenizer.decode(output[0], skip_special_tokens=True)
    return sentences

#tokenizer, model = get_model()

# Streamlit interface
st.title("Radiology Report Generator")

# User input for finding labels
labels = st.text_input("Enter Finding Labels:")


if st.button("Generate Report"):

    # Generate the radiology report
    report = generate_report(labels) #,model,tokenizer)
    # Display the report
    st.text_area("Generated Report:", value=report, height=300)


    
    
    # option 1) Mistral Usage tip 
    
# @st.cache(allow_output_mutation=True)
# def get_model():
#     #device = "cuda" # the device to load the model onto
#     model = AutoModelForCausalLM.from_pretrained("MariamAde/Mistral_finetuned_v2", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")
#     tokenizer = AutoTokenizer.from_pretrained("MariamAde/Mistral_finetuned_v2")
#     return tokenizer, model
    
    
    # option 2) 
    
# @st.cache(allow_output_mutation=True)
# def get_model():
#     tokenizer = LlamaTokenizer.from_pretrained("J4Lee/Medalpaca_finetuned_test")
#     model = MistralForCausalLM.from_pretrained("J4Lee/Medalpaca_finetuned_test")
#     return tokenizer, model



    # option 3) 
    
# @st.cache(allow_output_mutation=True)
# def get_model():
#     base_model, new_model = "mistralai/Mistral-7B-v0.1" , "inferenceanalytics/radmistral_7b"

#     base_model_reload = AutoModelForCausalLM.from_pretrained(
#         base_model, low_cpu_mem_usage=True,
#         return_dict=True,torch_dtype=torch.bfloat16,
#         device_map= 'auto')
    
#     model = PeftModel.from_pretrained(base_model_reload, new_model)
#     model = merged_model.merge_and_unload()

#     # Reload tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     return tokenizer, model

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE
