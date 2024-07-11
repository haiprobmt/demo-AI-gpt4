import streamlit as st
from chat import (
    search_demo, send_message_4o, 
    load_conversation, delete_conversation, 
    get_blob_url_with_sas, upload_to_blob_storage, upload_conversation_to_blob, upload_string_to_blob, download_blob_to_string, list_and_stream_files_in_folder, get_object_id, list_folder, list_files)
from openai import AzureOpenAI
import re
import requests

AZURE_OPENAI_SERVICE = "cog-kguqugfu5p2ki"
api_version = "2023-12-01-preview"
endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key="4657af893faf48e5bd81208d9f87f271"
    # azure_ad_token_provider,
)

# Initialize conversation
conversation = []
conversation_final = []

st.set_page_config(page_title="Deeeplabs Demo Chatbot", layout="wide")

st.title('Deeeplabs Demo Chatbot')

col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    # Transcription_Mode = st.button('Transcription Mode', use_container_width=True)
    # if Transcription_Mode:
    st.markdown("""
    <a href="https://notebuddyview.z23.web.core.windows.net/" target="_blank">
    <button style='margin: 10px; padding: 10px; background-color: #FFFFFF; color: black; border: curve; cursor: pointer;'>Transcription Mode</button>
    </a>
    """, unsafe_allow_html=True)
    
    # Define the folders for the dropdown
    folders = ['GPT 4-o', 'GPT 4', 'GPT 3.5']

    # Create the dropdown
    selected_folder = st.selectbox('Select model', folders)

    # Display the selected folder
    if selected_folder == 'GPT 4-o':
        model = "gpt-4o"
    elif selected_folder == 'GPT 4':
        model = "chat4"
    else:
        model = "chat16k"

logo_url = get_blob_url_with_sas('dl-logo-hamburger.png', "image")
st.sidebar.image(logo_url, width=180)

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

with col1:
    # Initialize session state for login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.password = ""

    if 'display_name' not in st.session_state:
        st.session_state.display_name = ''

    if 'message' not in st.session_state:
        st.session_state.message = ''

    # Create a function to display the login form
    def display_login_form():
        with st.form(key='login_form'):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Enter")

            if login_button:
                if username and password:  # Simple check to ensure fields are filled
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.password = password
                    st.experimental_rerun()  # Rerun the app to update the UI
                    

        st.session_state.display_name = get_object_id(username, password)['display_name']
        st.session_state.message = get_object_id(username, password)['message']

    # Display the login button or the welcome message based on login status
    if st.session_state.logged_in:
        # st.write(f"Welcome {st.session_state.username}!")
        if st.session_state.display_name != "":
            st.write(f"Welcome {st.session_state.display_name}!")
        else:
            st.write(st.session_state.message)
            st.session_state.show_form = True

            if 'show_form' in st.session_state and st.session_state.show_form:
                display_login_form()
    else:
        if st.button("Login"):
            st.session_state.show_form = True

        if 'show_form' in st.session_state and st.session_state.show_form:
            display_login_form()


# with col1:
# Sidebar for system prompt
    # st.sidebar.header("Settings")

username = st.session_state.username
password = st.session_state.password
display_name = st.session_state.display_name
library_name = "Sharepoint RAG Testing"

# st.sidebar.markdown("<br>"*4, unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: left;'>Sharepoint Folder</h1>", unsafe_allow_html=True)
if 'select_folder' not in st.session_state:
    st.session_state.select_folder = False
    # Create a button in the sidebar
if st.sidebar.button("Select folder"):
    st.session_state.select_folder = True

# Display a selectbox in the sidebar if the button is clicked
if st.session_state.select_folder:
    folders = list_folder(username, password, library_name)
    if folders == []:
        st.sidebar.write('No folders available')
        st.session_state.selected_file = 'nothing.pdf'
    else:
        folders.insert(0, 'Select a folder')
        selected_folder = st.sidebar.selectbox("Select a folder", folders)
        if selected_folder != 'Select a folder':
            if 'select_file' not in st.session_state:
                st.session_state.select_file = False
                st.session_state.selected_file = ''

            if st.sidebar.button("Select file"):
                st.session_state.select_file = True

            if st.session_state.select_file:
                file_list = list_files(username, password, library_name, selected_folder)
                if file_list == []:
                    st.sidebar.write('No files available')
                    # st.session_state.selected_file = 'nothing.pdf'
                else:
                    file_list.insert(0, 'Select a file')
                    selected_file = st.sidebar.selectbox("Select a file", file_list)
                    if selected_file != 'Select a file':
                        options = ['Select an option', 'Index', 'Filter']
                        function = st.sidebar.selectbox("Index or Filter", options)
                        if function == 'Select an option':
                            pass
                        if function == 'Index':
                            if st.button('Clear chat') or st.chat_input() or save_button:
                                pass
                            else:
                                list_and_stream_files_in_folder(username, password, library_name, selected_folder, selected_file)
                                st.sidebar.write(f'The file {selected_file} is ready to chat.')
                        if function == 'Filter':
                            st.session_state.selected_file = selected_file
                    if selected_file == 'Select a file':
                        selected_file = 'Select a file'

def main():

    st.header("Conversation")

    if st.button('Clear chat'):
        delete_chat()

    if user_input :=st.chat_input():
        # user_input = st.chat_input()
        try:
            selected_file = st.session_state.selected_file
            print(selected_file)
            chat_process(username, system_prompt, model, user_input, file_name = selected_file)
        except:
            selected_file = 'Select a file'
            chat_process(username, system_prompt, model, user_input, file_name = selected_file)

def delete_chat():
    st.session_state.messages = []
    try:
        st.session_state.messages = []
        delete_conversation("conversation.json")
        delete_conversation("history_json.json")
        st.write('Your chat has been deleted!')
    except:
        st.write('No chat to delete!')

# # User input section
# st.write(" ")
# st.write(" ")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

def chat_process(display_name, system_prompt, model, user_input, file_name):
    add_source = "\n\nAnswer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.\
        Provide the relevant sourcepage in the end of the response. \
        Do not provide the irrelevant sourcepage. \
        Do not provide the sourcepage if the question is generic\
        Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. \
        Use square brackets to reference the source, for example [info1.txt]. Don't combine sources, list each source separately, for example [info1.txt][info2.pdf]."

    # User-provided prompt
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        history = load_conversation("history_json.json")['history']
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
        search = completion.choices[0].text
    else:
        search = user_input
    try:
        conversation = load_conversation("conversation.json")
    except:
        conversation = [
                {
                    "role": "system",
                    "content": system_prompt.replace('   ', '') + add_source + f"\nRemember, my name is {display_name}."
                }
            ]
    query = search_demo(search, file_name)['user_message']
    conversation.append({"role": "user", "content": query}) 
    response_0 = send_message_4o(conversation, model)
    
    pattern = r"\[([^\[\]]+\.pdf)\]"
    resources_final = list(set(re.findall(pattern, response_0)))

    resources_couple = [(key, value) for key, value in enumerate(resources_final, start=1)]
    response = response_0
    for key, value in resources_couple:
        response = response.replace(f"[{value}]", f"[{key}]")

    response_final = response
    conversation[-1]['content'] = user_input
    conversation.append({"role": "assistant", "content": response_final})

    history.append("user: " + user_input)
    history.append("assistant: " + response_final)
    history_json = {"history": history}
    upload_conversation_to_blob("conversation.json", conversation)
    upload_conversation_to_blob("history_json.json", history_json)
    st.session_state.messages.append({"role": "assistant", "content": {"response": response, "resources": resources_couple}})

    
if __name__ == "__main__":
    main()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            elif message["role"] == "assistant":
                st.write(message["content"]["response"].replace(")", "").replace("(", ""))
                resource_list = message["content"]["resources"]
                if len(resource_list) > 0:
                    st.write("References:")
                    for key, resource in resource_list:
                        resource_name = resource
                        reference_url = get_blob_url_with_sas(resource_name, "data-source")
                        st.write(f'{key}. [{resource_name}]({reference_url})')
