import time

from azure.search.documents.indexes.models import (
    SimpleField,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndexer,
    IndexingParameters,
    VectorSearch,
    VectorSearchAlgorithmConfiguration
)

from utilities import (
    get_index_name,
    create_index,
    get_datasource_name,
    create_blob_datasource,
    get_indexer_name,
    get_indexer_client,
    get_knowledge_store_connection_string,
    get_chunk_index_blob_container_name,
    wait_for_indexer_completion,
    get_index_client
)


class ChunkIndexManager():

    def _create_chunk_index(self, index_prefix, config):
        """
        Creates a chunk index in Azure Search with the given index_prefix and config.

        Args:
            index_prefix (str): The prefix to use for the index name.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.

        Returns:
            SearchIndex: The created index.
        """
        name = get_index_name(f"{index_prefix}-chunk")
        vector_search = VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(
                    name="my-vector-config",
                    kind="hnsw",
                    hnsw_parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 1000,
                        "metric": "cosine"
                    }
                )
            ]
        )
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String,  filterable=True, sortable=True, key=True),
            SimpleField(name="source_document_id", type=SearchFieldDataType.String),
            SimpleField(name="source_document_filepath", type=SearchFieldDataType.String),
            SimpleField(name="source_field_name", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="index", type=SearchFieldDataType.Int64),
            SimpleField(name="offset", type=SearchFieldDataType.Int64),
            SimpleField(name="length", type=SearchFieldDataType.Int64),
            SimpleField(name="hash", type=SearchFieldDataType.String),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SearchField(name="embedding",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=1536,
                        vector_search_configuration="my-vector-config")
        ]
        index = create_index(name, fields, vector_search=vector_search, semantic_title_field_name="title", semantic_content_field_names=["text"], config=config)
        return index

    def _create_chunk_datasource(self, index_prefix, storage_connection_string, container_name, config):
        """
        Creates a blob data source for the chunk index with the given index_prefix, storage_connection_string, container_name, and config.

        Args:
            index_prefix (str): The prefix to use for the data source name.
            storage_connection_string (str): The connection string for the Azure Storage account.
            container_name (str): The name of the blob container.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.

        Returns:
            SearchIndexerDataSource: The created data source.
        """
        name = get_datasource_name(f"{index_prefix}-chunk")
        return create_blob_datasource(name, storage_connection_string, container_name, config=config)

    def _create_chunk_indexer(self, index_prefix, data_source_name, index_name, config):
        """
        Creates an indexer for the chunk index with the given index_prefix, data_source_name, index_name, and config.

        Args:
            index_prefix (str): The prefix to use for the indexer name.
            data_source_name (str): The name of the data source.
            index_name (str): The name of the index.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.

        Returns:
            SearchIndexer: The created indexer.
        """
        name = get_indexer_name(f"{index_prefix}-chunk")
        parameters = IndexingParameters(configuration={"parsing_mode": "json"})
        indexer = SearchIndexer(
            name=name,
            data_source_name=data_source_name,
            target_index_name=index_name,
            parameters=parameters
        )
        indexer_client = get_indexer_client(config)
        return indexer_client.create_indexer(indexer)

    def create_chunk_index_resources(self, index_prefix, config) -> dict:
        """
        Creates the resources for the chunk index with the given index_prefix and config.

        Args:
            index_prefix (str): The prefix to use for the index, data source, and indexer names.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.

        Returns:
            dict: A dictionary containing information about the created resources.
        """
        chunk_index_storage_connection_string = get_knowledge_store_connection_string(config)
        chunk_index_blob_container_name = get_chunk_index_blob_container_name(index_prefix)
        index_name = self._create_chunk_index(index_prefix, config).name
        data_source_name = self._create_chunk_datasource(index_prefix, chunk_index_storage_connection_string, chunk_index_blob_container_name, config=config).name
        time.sleep(5)
        indexer_name = self._create_chunk_indexer(index_prefix, data_source_name, index_name, config=config).name
        wait_for_indexer_completion(indexer_name, config=config)
        return {"index_name": index_name, "data_source_name": data_source_name, "indexer_name": indexer_name}

    def delete_chunk_index_resources(self, index_prefix, config):
        """
        Deletes the resources for the chunk index with the given index_prefix and config.

        Args:
            index_prefix (str): The prefix used for the index, data source, and indexer names.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.
        """
        index_client = get_index_client(config)
        indexer_client = get_indexer_client(config)

        index_client.delete_index(index=f"{index_prefix}-chunk-index")
        indexer_client.delete_indexer(indexer=f"{index_prefix}-chunk-indexer")
        indexer_client.delete_data_source_connection(data_source_connection=f"{index_prefix}-chunk-datasource")
