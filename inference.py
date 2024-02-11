import streamlit as st
import torch
from transformers import BertForTokenClassification, BertTokenizerFast  # Import BertTokenizerFast

def load_model(model_name='linkbert.pth'):
    model_path = model_name
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to inference mode
    return model

def predict_and_annotate(model, tokenizer, text):
    # Tokenize the input text with special tokens
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    input_ids, attention_mask, offset_mapping = inputs["input_ids"], inputs["attention_mask"], inputs["offset_mapping"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    predictions = predictions.squeeze().tolist()
    offset_mapping = offset_mapping.squeeze().tolist()

    annotated_text = ""
    previous_end = 0
    for offset, prediction in zip(offset_mapping, predictions):
        start, end = offset
        if start == end:  # Skip special tokens
            continue
        if prediction == 1:  # Anchor text
            if start > previous_end:
                annotated_text += text[previous_end:start]
            annotated_text += f"<u>{text[start:end]}</u>"
        else:
            if start > previous_end:
                annotated_text += text[previous_end:start]
            annotated_text += text[start:end]
        previous_end = end
    annotated_text += text[previous_end:]  # Append remaining text

    return annotated_text

# Streamlit app setup
st.title("BERT Token Classification for Anchor Text Prediction")

# Load the model and tokenizer
model = load_model('linkbert.pth')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Use BertTokenizerFast

# User input text area
user_input = st.text_area("Paste the text you want to analyze:", "Type or paste text here.")

if st.button("Predict Anchor Texts"):
    if user_input:
        annotated_text = predict_and_annotate(model, tokenizer, user_input)
        st.markdown(annotated_text, unsafe_allow_html=True)
    else:
        st.write("Please paste some text into the text area.")
