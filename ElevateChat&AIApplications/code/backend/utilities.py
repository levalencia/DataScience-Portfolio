import os
import time
import requests

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField
)

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_KNOWLEDGE_STORE_CONNECTION_STRING = os.getenv("AZURE_KNOWLEDGE_STORE_STORAGE_CONNECTION_STRING")


def get_index_client(config) -> SearchIndexClient:
    """Returns a SearchIndexClient object for the specified Azure Search service."""
    return SearchIndexClient(config['AZURE_SEARCH_SERVICE_ENDPOINT'], AzureKeyCredential(config['AZURE_SEARCH_ADMIN_KEY']))


def get_indexer_client(config) -> SearchIndexerClient:
    """Returns a SearchIndexerClient object for the specified Azure Search service."""
    return SearchIndexerClient(config['AZURE_SEARCH_SERVICE_ENDPOINT'], AzureKeyCredential(config['AZURE_SEARCH_ADMIN_KEY']))


def get_index_name(index_prefix):
    """Returns the name of an Azure Search index given a prefix."""
    return f"{index_prefix}-index"


def get_datasource_name(index_prefix):
    """Returns the name of an Azure Search datasource given a prefix."""
    return f"{index_prefix}-datasource"


def get_skillset_name(index_prefix):
    """Returns the name of an Azure Search skillset given a prefix."""
    return f"{index_prefix}-skillset"


def get_indexer_name(index_prefix):
    """Returns the name of an Azure Search indexer given a prefix."""
    return f"{index_prefix}-indexer"


def get_chunk_index_blob_container_name(index_prefix):
    """Returns the name of an Azure Blob Storage container for chunk indexing given a prefix."""
    return f"{index_prefix}ChunkIndex".replace('-', '').lower()


def get_knowledge_store_connection_string(config):
    """Returns the connection string for an Azure Knowledge Store."""
    return config['AZURE_SEARCH_KNOWLEDGE_STORE_CONNECTION_STRING']


def create_index(index_name, fields, vector_search, semantic_title_field_name, semantic_content_field_names, config):
    """Creates an Azure Search index with the specified fields and semantic settings."""
    semantic_settings = SemanticSettings(
        configurations=[SemanticConfiguration(
            name='default',
            prioritized_fields=PrioritizedFields(
                title_field=SemanticField(field_name=semantic_title_field_name), prioritized_content_fields=[SemanticField(field_name=field_name) for field_name in semantic_content_field_names]))])
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_settings=semantic_settings)
    index_client = get_index_client(config)
    return index_client.create_index(index)


def create_blob_datasource(datasource_name, storage_connection_string, container_name, config):
    """Creates an Azure Search datasource for Azure Blob Storage with the specified connection string and container name."""
    # This example utilizes a REST request as the python SDK doesn't support the blob soft delete policy yet
    api_version = '2023-07-01-Preview'
    headers = {
        'Content-Type': 'application/json',
        'api-key': f'{config["AZURE_SEARCH_ADMIN_KEY"]}'
    }
    data_source = {
        "name": datasource_name,
        "type": "azureblob",
        "credentials": {"connectionString": storage_connection_string},
        "container": {"name": container_name},
        "dataDeletionDetectionPolicy": {"@odata.type": "#Microsoft.Azure.Search.NativeBlobSoftDeleteDeletionDetectionPolicy"}
    }

    url = '{}/datasources/{}?api-version={}'.format(config['AZURE_SEARCH_SERVICE_ENDPOINT'], datasource_name, api_version)
    requests.put(url, json=data_source, headers=headers)

    ds_client = get_indexer_client(config)
    return ds_client.get_data_source_connection(datasource_name)


def wait_for_indexer_completion(indexer_name, config):
    """Waits for an Azure Search indexer to complete indexing."""
    indexer_client = get_indexer_client(config)
    # poll status and wait until indexer is complete
    status = f"Indexer {indexer_name} not started yet"
    while (indexer_client.get_indexer_status(indexer_name).last_result is None) or ((status := indexer_client.get_indexer_status(indexer_name).last_result.status) != "success"):
        print(f"Indexing status:{status}")

        # It's possible that the indexer may reach a state of transient failure, especially when generating embeddings
        # via Open AI. For the purposes of the demo, we'll just break out of the loop and continue with the rest of the steps.
        if (status == "transientFailure"):
            print(f"Indexer {indexer_name} failed before fully indexing documents")
            break
        time.sleep(5)
