import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
import re
import json
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import uuid
from pypdf import PdfReader, PdfWriter
import html
import io
from azure.ai.formrecognizer import DocumentAnalysisClient
# from tenacity import retry, stop_after_attempt, wait_random_exponential
import tempfile
from office365.sharepoint.client_context import ClientContext
import os
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
# from office365.sharepoint.file import File
from office365.runtime.client_request_exception import ClientRequestException
import io
import PyPDF2
import index_doc_sharepoints
import requests

# Replace these with your own values, either in environment variables or directly here
AZURE_SEARCH_SERVICE = "search-sanderstrothmann"
AZURE_SEARCH_INDEX = "index-sanderstrothmann"
# AZURE_SEARCH_INDEX_1 = "vector-1715913242600"
AZURE_OPENAI_SERVICE = "cog-kguqugfu5p2ki"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = "chat16k"
AZURE_SEARCH_API_KEY = "i7F5uuUzXR8KCZ58o4r3aZAr9QG5dDp3erOLgz6kb9AzSeAabEHy"
AZURE_OPENAI_EMB_DEPLOYMENT = "embedding"

AZURE_CLIENT_ID = "c4642a73-05e3-4a68-8228-7d241ba8d6e6"
AZURE_CLIENT_SECRET = "I_F8Q~MhKD9fCfT9725j9mCad39G6bpwVpolAb.f"
AZURE_TENANT_ID = "667439c9-20b5-4283-bd7b-fb6b3099d221"
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")

storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=sasanderstrothmann;AccountKey=x4eeHxz6VMBqpmE+eLmA8ECKvA1EzTeUzOH2b9GkLiW7TVeo8DPrx1ckbcMM2QCj+u06a8vkxbI4+AStDI0lAQ==;EndpointSuffix=core.windows.net"
container = "data-source"

api_version = "2023-12-01-preview"
endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key="4657af893faf48e5bd81208d9f87f271"
    # azure_ad_token_provider,
)

AZURE_STORAGE_ACCOUNT = "sasanderstrothmann"
storagekey = "g2LVDKMxRjz09R2t/CQTz2JwAWZcsCge/dsMlOXb2mo2adikxqPNQpbDk8yeSlaoP7C+dvLxtEAV+AStYsDMWQ=="
formrecognizerservice = "pick-ai-doc-intel-version-2"
formrecognizerkey = "e739eef01ab34d46b16bb69e879a14b6"
verbose = True
novectors = True
remove = True
removeall = False
skipblobs = False
localpdfparser = True

def search(prompt, filter=None):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    # Set up clients for Cognitive Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=credential)   
    
    query_vector = client.embeddings.create(input=prompt,model= "embedding").data[0].embedding
    # filter = f"image eq '{image}'"
    r = search_client.search(prompt, 
                            filter=filter,
                            query_type=QueryType.SIMPLE, 
                            query_language="en-us", 
                            query_speller="lexicon", 
                            semantic_configuration_name="default", 
                            top=10,
                            vector=query_vector if query_vector else None, 
                            top_k=50 if query_vector else None,
                            vector_fields="embedding" if query_vector else None
                            )
    results = [doc['image'] + ": " + doc['content'].replace("\n", "").replace("\r", "") for doc in r if doc['image'] != None]
    content = "\n".join(results)
    user_message = prompt + "\n SOURCES:\n" + content
    
    # Regular expression pattern to match URLs
    url_pattern = r'https?://[^\s,]+(?:\.png|\.jpg|\.jpeg|\.gif)'
    # Find all URLs in the text
    image_urls = re.findall(url_pattern, content)
    if len(image_urls) > 0:
        image = image_urls[0]
    else:
        image = None
    return {"user_message": user_message, "image": image}

def search_demo(prompt, file_name):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    # Set up clients for Cognitive Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name="index-demo",
        credential=credential)   
    
    query_vector = client.embeddings.create(input=prompt,model= "embedding").data[0].embedding
    if file_name == 'Select a file':
        filter = None
    else:
        filter = f"sourcefile eq '{file_name}'"
    r = search_client.search(prompt, 
                            filter=filter,
                            query_type=QueryType.SIMPLE, 
                            query_language="en-us", 
                            query_speller="lexicon", 
                            semantic_configuration_name="default", 
                            top=5,
                            vector=query_vector if query_vector else None, 
                            top_k=50 if query_vector else None,
                            vector_fields="embedding" if query_vector else None
                            )
    results = [{"content": doc['sourcepage'] + ": " + doc['content'].replace("\n", "").replace("\r", ""), "source":doc['sourcepage']} for doc in r]
    content = [doc['content'].replace("\n", "").replace("\r", "") for doc in results]
    source = list(set([doc['source'] for doc in results]))
    content_final = "\n".join(content)
    user_message = prompt + "\n SOURCES:\n" + content_final
    return {"source": source, "user_message": user_message}

# def send_message(messages, model=AZURE_OPENAI_CHATGPT_DEPLOYMENT):
#     response = openai.ChatCompletion.create(
#         engine=model,
#         messages=messages,
#         temperature=0.5,
#         max_tokens=2024
#     )
#     response_final = response['choices'][0]['message']['content']
#     return response_final

def send_message_4o(messages, model):
    response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.5,
    max_tokens=2024
    ) 
    return response.choices[0].message.content

def upload_conversation_to_blob(blob_name, data):
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    # Convert dict to JSON
    if '.json' in blob_name:
        json_data = json.dumps(data)
    else:
        json_data = data
    # Get blob client
    blob_client = blob_service_client.get_blob_client("conversation", blob_name)

    # Upload the JSON data
    blob_client.upload_blob(json_data, overwrite=True)

def load_conversation(blob_name):
    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client("conversation")

    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)

    # Download the blob as a text string
    json_data = blob_client.download_blob().readall()

    # Convert the JSON string to a Python object
    json_object = json.loads(json_data)

    # Now you can work with the JSON object
    return json_object

def delete_conversation(blob_name):
    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client("conversation")

    # Get a reference to the blob
    blob_client = container_client.get_blob_client(blob_name)

    # Delete the blob
    blob_client.delete_blob()

def get_blob_url_with_sas(file_name, container):
    # Generate the SAS token for the file
    sas_token = generate_blob_sas(
        account_name="sasanderstrothmann",
        account_key="QtoEp5hl3aIWHdkTO1Q8I4R30M5lNnrKsSHjkuAL6BMKvf03Vh6BJfJ5RWEG7hlAGRRu3/pvK+Kx+AStgTMMQQ==",
        container_name=container,
        blob_name=file_name,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now() + timedelta(hours=1)  # Set the expiry time for the SAS token
    )

    # Construct the URL with SAS token
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container)
    blob_url = container_client.get_blob_client(file_name).url
    blob_url_with_sas = f"{blob_url}?{sas_token}"
    return blob_url_with_sas

def upload_to_blob_storage(file):
    # Define your Azure Blob Storage connection string
    connect_str = storage_connection_string

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Define your container name
    container_name = container

    # Create a ContainerClient object
    container_client = blob_service_client.get_container_client(container_name)

    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.getvalue())

    # Upload the file to Azure Blob Storage
    with open(temp_file.name, "rb") as data:
        blob_client = container_client.upload_blob(name=file.name, data=data, overwrite=True)

    # Delete the temporary file
    temp_file.close()

    # Return the URL of the uploaded file
    return blob_client.url 

def upload_string_to_blob(container_name, blob_name, data):
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data, overwrite = True)

def download_blob_to_string(container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall().decode("utf-8")

def list_and_stream_files_in_folder(username, password, library_name, folder_name, file_target=None):
    ctx = ClientContext('https://deeeplabs.sharepoint.com/sites/SharepointExperimentation-SharepointRAGTesting').with_credentials(UserCredential(username, password))
    try:
        folder_url = f"/sites/SharepointExperimentation-SharepointRAGTesting/Shared Documents/{library_name}/{folder_name}"
        root_folder = ctx.web.get_folder_by_server_relative_path(folder_url)
        files = root_folder.get_files(True).execute_query()

        print(f"Files in folder '{folder_name}':")
        for file in files:
                file_name = file.properties['Name']
                if file_name == file_target:
                    download_stream = io.BytesIO()
                    file.download(download_stream).execute_query()
                    download_stream.seek(0)
                    try:
                        index_doc_sharepoints.run(download_stream, filename_name=file_name)
                    except Exception as e:
                        print(f"Error reading PDF content from '{file_name}': {e}")
                else:
                    print(f"Skipping file '{file.properties['Name']}'")
    except ClientRequestException as e:
        print(f"Error accessing folder: {e}")

def get_object_id(username, password):

    # Set the Azure AD tenant ID and client ID
    tenant_id = '667439c9-20b5-4283-bd7b-fb6b3099d221'
    client_id = 'e30b46eb-fd4d-4966-8595-eeef2c48e82f'

    # Set the Microsoft Graph API endpoint
    graph_api_endpoint = 'https://graph.microsoft.com/v1.0'

    # Get the access token using the username and password
    token_url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
    token_data = {
        'grant_type': 'password',
        'client_id': client_id,
        'username': username,
        'password': password,
        'scope': 'https://graph.microsoft.com/.default'
    }
    token_response = requests.post(token_url, data=token_data).json()
    try:
        access_token = token_response['access_token']
        # Get the user's profile using the access token
        profile_url = f'{graph_api_endpoint}/me'
        profile_headers = {
            'Authorization': f'Bearer {access_token}'
        }
        profile_response = requests.get(profile_url, headers=profile_headers).json()

        # Get the Azure Object ID from the user's profile
        azure_object_id = profile_response['id']
        username_display = profile_response['displayName']
        error_message = ""
    except KeyError:
        azure_object_id = ""
        username_display = ""
        error_message = "Incorrect username or password, please try again."
    return {"azure_object_id": azure_object_id, "display_name": username_display, "message": error_message}

def list_folder(username, password, library_name):
    ctx = ClientContext('https://deeeplabs.sharepoint.com/sites/SharepointExperimentation-SharepointRAGTesting').with_credentials(UserCredential(username, password))

    try:
        folder_url = f"/sites/SharepointExperimentation-SharepointRAGTesting/Shared Documents/{library_name}"
        root_folder = ctx.web.get_folder_by_server_relative_path(folder_url)
        folder = root_folder.get_folders().get().execute_query()
        folder_list = [f.properties['Name'] for f in folder]
    except:
        print("Error accessing folder")
        folder_list = []
    return folder_list

def list_files(username, password, library_name, folder):
    ctx = ClientContext('https://deeeplabs.sharepoint.com/sites/SharepointExperimentation-SharepointRAGTesting').with_credentials(UserCredential(username, password))

    try:
        folder_url = f"/sites/SharepointExperimentation-SharepointRAGTesting/Shared Documents/{library_name}/{folder}"
        root_folder = ctx.web.get_folder_by_server_relative_path(folder_url)
        files = root_folder.get_files(True).execute_query()
        file_list = [file.properties['Name'] for file in files]
    except:
        print("Error accessing file")
        file_list = []
    return file_list
