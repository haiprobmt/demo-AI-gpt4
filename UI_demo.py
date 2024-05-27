import streamlit as st
from chat import (
    search, send_message_4o, 
    load_conversation, delete_conversation, 
    get_blob_url_with_sas, upload_to_blob_storage, upload_conversation_to_blob,
    get_json, remove_json, upload_string_to_blob, download_blob_to_string)
from openai import AzureOpenAI
import re
from index_doc import index_document
import index_doc_pdf
import credentials as cred  
import json

AZURE_OPENAI_SERVICE = cred.AZURE_OPENAI_SERVICE
api_version = "2023-12-01-preview"
endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=cred.openai_api_key
    # azure_ad_token_provider,
)

# Initialize conversation
conversation = []
conversation_final = []

st.set_page_config(page_title="Deeeplabs Demo Chatbot", layout="wide")

st.title('Sanderstrothmann Chatbot Demo')

col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    # Transcription_Mode = st.button('Transcription Mode', use_container_width=True)
    # if Transcription_Mode:
    # st.markdown("""
    # <a href="https://notebuddyview.z23.web.core.windows.net/" target="_blank">
    # <button style='margin: 10px; padding: 10px; background-color: #FFFFFF; color: black; border: curve; cursor: pointer;'>Transcription Mode</button>
    # </a>
    # """, unsafe_allow_html=True)
    
    # Define the options for the dropdown
    options = ['GPT 3.5', 'GPT 4', 'GPT 4-o']

    # Create the dropdown
    selected_option = st.selectbox('Select model', options)

    # Display the selected option
    if selected_option == 'GPT 3.5':
        model = "chat16k"
    elif selected_option == 'GPT 4':
        model = "chat4"
    else:
        model = "gpt-4o"

logo_url = get_blob_url_with_sas('dl-logo-hamburger.png', "image")
st.sidebar.image(logo_url, width=180)
# with col1:
# Sidebar for system prompt
    # st.sidebar.header("Settings")
st.sidebar.markdown("<h1 style='text-align: left;'>System prompt</h1>", unsafe_allow_html=True)

try:
    value = download_blob_to_string('test', 'system_prompt.txt')
except:
    value = ""
system_prompt = st.sidebar.text_area(label = "", value = value, height=200)
save_button = st.sidebar.button('Save')
if save_button:
    upload_string_to_blob('test', 'system_prompt.txt', system_prompt)
    st.sidebar.write('Your prompt has been saved!')

system_prompt_final = system_prompt

# st.sidebar.markdown("<br>"*4, unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: left;'>Upload File</h1>", unsafe_allow_html=True)
# Upload file
uploaded_file  = st.sidebar.file_uploader("Choose an excel file", type=["xlsx"], help="Upload an excel file to provide context")
# Check if a file was uploaded
if uploaded_file:
    # Save the file to Azure Blob Storage
    file_name = uploaded_file.name
    upload_to_blob_storage(uploaded_file)
    if "xlsx" in file_name:
    # # file_name = uploaded_file.name
        index_document(file_name)
    else:
        index_doc_pdf.run(uploaded_file)
    st.sidebar.write('File uploaded successfully!')

# Main layout
# Display conversation
st.header("Conversation")

delete_button = st.button('Clear chat')
if delete_button:
    st.session_state.messages = []
    try:
        st.session_state.messages = []
        delete_conversation("conversation_sander.json")
        delete_conversation("history_sander_json.json")
        st.write('Your chat has been deleted!')
    except:
        st.write('No chat to delete!')

# User input section
st.write(" ")
st.write(" ")

add_source = """\n\nIf the question is generic then just answer with a friendly tone. Return the outcome in json format following this format: {"response": 'The answer to the question', "images": 'the relevant image urls present in a list, if the question is generic, do not return anything'}."""
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# User-provided prompt
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        history = load_conversation("history_sander_json.json")['history']
        print(history)
    except:
        history = []

    summary_prompt_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question. Source names are not good search terms to include in the search query.

    Summary:
    {summary}

    Question:
    {question}

    Search query:
    """
        
    if len(history) > 0:
        completion = client.completions.create(
            model='davinci',
            prompt=summary_prompt_template.format(summary="\n".join(history), question=user_input),
            temperature=0.0,
            max_tokens=32,
            stop=["\n"])
        search_query = completion.choices[0].text
    else:
        search_query = user_input
    try:
        conversation = load_conversation("conversation_sander.json")
    except:
        conversation = [
                {
                    "role": "system",
                    "content": system_prompt.replace('   ', '') + add_source
                }
            ]
    query = search(search_query)
    conversation.append({"role": "user", "content": query})
    response = send_message_4o(conversation, model)
    try:
        response = json.loads(response)
        image_url_list = list(set(response['images']))
        response_final = response['response']
    except:
        response_final = response
        image_url_list = None

    conversation[-1]["content"] = user_input
    conversation_final.append({"role": "user", "content": user_input})
    conversation.append({"role": "assistant", "content": response_final})
    conversation_final.append({"role": "assistant", "content": response_final})
    upload_conversation_to_blob('conversation_sander.json', conversation)
    conversation_final.append({"role": "image", "content": image_url_list})
    upload_conversation_to_blob('conversation_image.json', conversation_final)
    history.append("user: " + user_input)
    history.append("assistant: " + response_final)
    history_json = {"history": history}
    upload_conversation_to_blob("conversation_sander.json", conversation)
    upload_conversation_to_blob("history_sander_json.json", history_json)

    st.session_state.messages.append({"role": "assistant", "content": {"response": response_final, "images": image_url_list}})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        elif message["role"] == "assistant":
            images = message["content"]["images"]
            col4, col5 = st.columns([3, 1])
            with col4:
                st.write(message["content"]["response"])
            with col5:
                if images is not None:
                    for img_path in images:
                        if "https" in img_path:
                            img_path_sas = img_path
                        else:
                            img_path_sas = get_blob_url_with_sas(img_path, "image")
                        # Display the image
                        st.image(img_path_sas, width=500)