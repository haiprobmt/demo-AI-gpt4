import os
import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
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
from openai import AzureOpenAI

# Replace these with your own values, either in environment variables or directly here
AZURE_SEARCH_SERVICE = "search-sanderstrothmann"
AZURE_SEARCH_INDEX = "index-sanderstrothmann"
# AZURE_SEARCH_INDEX_1 = "vector-1715913242600"
AZURE_OPENAI_SERVICE = "cog-kguqugfu5p2ki"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = "gpt-4o"
AZURE_SEARCH_API_KEY = "i7F5uuUzXR8KCZ58o4r3aZAr9QG5dDp3erOLgz6kb9AzSeAabEHy"
AZURE_OPENAI_EMB_DEPLOYMENT = "embedding"

AZURE_CLIENT_ID = "c4642a73-05e3-4a68-8228-7d241ba8d6e6"
AZURE_CLIENT_SECRET = "I_F8Q~MhKD9fCfT9725j9mCad39G6bpwVpolAb.f"
AZURE_TENANT_ID = "667439c9-20b5-4283-bd7b-fb6b3099d221"
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")

storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=sasanderstrothmann;AccountKey=x4eeHxz6VMBqpmE+eLmA8ECKvA1EzTeUzOH2b9GkLiW7TVeo8DPrx1ckbcMM2QCj+u06a8vkxbI4+AStDI0lAQ==;EndpointSuffix=core.windows.net"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
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
    
    query_vector = client.Embedding.create(engine=AZURE_OPENAI_EMB_DEPLOYMENT, input=prompt)["data"][0]["embedding"]
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

def search_demo(prompt, filter=None):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    # Set up clients for Cognitive Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name="index-demo",
        credential=credential)   
    
    query_vector = client.embeddings.create(input=prompt,model= "embedding").data[0].embedding
    # filter = f"image eq '{image}'"
    r = search_client.search(prompt, 
                            filter=filter,
                            query_type=QueryType.SIMPLE, 
                            query_language="en-us", 
                            query_speller="lexicon", 
                            semantic_configuration_name="default", 
                            top=3,
                            vector=query_vector if query_vector else None, 
                            top_k=50 if query_vector else None,
                            vector_fields="embedding" if query_vector else None
                            )
    results = [doc['sourcepage'] + ": " + doc['content'].replace("\n", "").replace("\r", "") for doc in r if doc['sourcepage'] != None]
    content = "\n".join(results)
    user_message = prompt + "\n SOURCES:\n" + content
    return user_message

def send_message(messages, model=AZURE_OPENAI_CHATGPT_DEPLOYMENT):
    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    response_final = response['choices'][0]['message']['content']
    return response_final
    
def send_message_4o(messages):
    response = client.chat.completions.create(
    model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
    messages=messages) 
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
