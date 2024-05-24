from io import StringIO
import pandas as pd
from azure.storage.blob import BlobClient, BlobServiceClient
import os
import uuid
import json
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import credentials as cred

AZURE_OPENAI_SERVICE = cred.AZURE_OPENAI_SERVICE
api_version = "2023-12-01-preview"
endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=cred.openai_api_key
    # azure_ad_token_provider,
)

container = cred.sa_container_name
connection_string = cred.sa_connection_string


def load_excel_from_blob(blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    blob = blob_client.download_blob().content_as_bytes()
    df = pd.read_excel(blob)
    return df

def compute_embedding_text_3_large(text):
    response = client.embeddings.create(
        input=text,
        model= "embedding"
    )
    return response.data[0].embedding

def upload_doc(docs):
    # Replace with your search service endpoint and admin key
    AZURE_SEARCH_SERVICE = AZURE_SEARCH_SERVICE
    AZURE_SEARCH_INDEX = AZURE_SEARCH_INDEX
    service_endpoint =f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"
    admin_key = cred.SEARCH_API_KEY

    # Replace with your index name
    index_name = AZURE_SEARCH_INDEX
    credential = AzureKeyCredential(admin_key)

    # print(doc["product_name", "wine_year"])
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
    search_client.upload_documents(docs)
    print(f"The documents have been uploaded to the index")

def index_document(file_name):
    df = load_excel_from_blob(file_name)
    dictionary_data = df.to_dict(orient="records")
    #Convert to long String
    docs=[]
    for dictionary in dictionary_data:
        string = ""
        for key in dictionary:
            string += f"{key}:{dictionary[key]} \n "
        docs.append({
            "id": str(uuid.uuid4()),
            "source": dictionary["Source"] if str(dictionary["Source"]) != "nan" else None,
            "image": dictionary["Picture Product"] if str(dictionary["Picture Product"]) != "nan" else None,
            "content": string,
            "embedding":compute_embedding_text_3_large(string),
            "item": dictionary["Item Name"] if str(dictionary["Item Name"]) != "nan" else None,
            "sourcefile": file_name
            })  
    upload_doc(docs)
