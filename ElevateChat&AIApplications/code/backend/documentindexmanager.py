import time

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SearchIndexer,
    IndexingParameters,
    FieldMapping,
    FieldMappingFunction,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchIndexerSkillset,
    SearchIndexerKnowledgeStore,
    SearchIndexerKnowledgeStoreProjection,
    SearchIndexerKnowledgeStoreFileProjectionSelector,
    WebApiSkill,
    OcrSkill,
    ImageAnalysisSkill,
    MergeSkill,
    CognitiveServicesAccountKey
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
    get_index_client,
    get_skillset_name
)


class DocumentIndexManager():
    def _create_document_index(self, index_prefix, config):
        """
        Creates a document index in Azure Search with the given index_prefix and config.

        Args:
            index_prefix (str): The prefix to use for the index name.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.

        Returns:
            Index: The created document index.
        """
        # Get the name for the index
        name = get_index_name(index_prefix)
        # Define the fields for the index
        fields = [
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True, sortable=True, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="filesize", type=SearchFieldDataType.Int64),
            SimpleField(name="filepath", type=SearchFieldDataType.String),
            SearchableField(name="metadata_storage_name", type=SearchFieldDataType.String, filterable=True, retrievable=True),
            SimpleField(name="metadata_storage_path", type=SearchFieldDataType.String, retrievable=True),
            SearchableField(name="merged_content", type=SearchFieldDataType.String, retrievable=True),
            SimpleField(name="text", type="Collection(Edm.String)", retrievable=True, searchable=True),
            SimpleField(name="layoutText", type="Collection(Edm.String)", retrievable=True, searchable=True)
        ]
        # Create the index using the custom utility function
        return create_index(name, fields, config=config, vector_search=None, semantic_title_field_name="filepath", semantic_content_field_names=["content"])

    def _create_document_datasource(self, index_prefix, storage_connection_string, container_name, config):
        """
        Creates a blob datasource in Azure Search with the given index_prefix, storage_connection_string, container_name, and config.

        Args:
            index_prefix (str): The prefix to use for the datasource name.
            storage_connection_string (str): The connection string for the storage account.
            container_name (str): The name of the container to index.
            config (SearchServiceClientConfiguration): The configuration for the Azure Search service.

        Returns:
            DataSource: The created blob datasource.
        """
        # Get the name for the datasource
        name = get_datasource_name(index_prefix)
        # Create the datasource using the custom utility function
        return create_blob_datasource(name, storage_connection_string, container_name, config)

    def _create_document_skillset(self, index_prefix, config, content_field_name="content"):
        """
        Creates a skillset for a document using Azure Search.

        Args:
            index_prefix (str): The prefix for the index.
            config (dict): The configuration dictionary.
            content_field_name (str, optional): The name of the content field. Defaults to "content".

        Returns:
            Skillset: The created skillset.
        """

        # Get the endpoint for the embedding skill from the configuration dictionary
        embedding_skill_endpoint = config['AZURE_SEARCH_EMBEDDING_SKILL_ENDPOINT']

        # Get the name of the skillset
        name = get_skillset_name(index_prefix)

        # Get the name of the chunk index blob container
        chunk_index_blob_container_name = get_chunk_index_blob_container_name(index_prefix)

        # Define the content context
        content_context = f"/document/{content_field_name}"

        # Define the embedding skill
        embedding_skill = WebApiSkill(
            name="chunking-embedding-skill",
            uri=embedding_skill_endpoint,
            timeout="PT3M",
            batch_size=1,
            degree_of_parallelism=1,
            context=content_context,
            inputs=[
                InputFieldMappingEntry(name="document_id", source="/document/document_id"),
                InputFieldMappingEntry(name="text", source=content_context),
                InputFieldMappingEntry(name="filepath", source="/document/filepath"),
                InputFieldMappingEntry(name="fieldname", source=f"='{content_field_name}'")
            ],
            outputs=[OutputFieldMappingEntry(name="chunks", target_name="chunks")]
        )

        # Define the OCR skill
        ocr_skill = OcrSkill(
            name="ocr-skill",
            context=content_context,
            inputs=[InputFieldMappingEntry(name="image", source="/document/normalized_images/*")],
            outputs=[
                OutputFieldMappingEntry(name="text", target_name="text"),
                OutputFieldMappingEntry(name="layoutText", target_name="layoutText")
            ]
        )

        # Define the merge skill
        merge_skill = MergeSkill(
            name="merge-skill",
            context="/document",
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/content"),
                InputFieldMappingEntry(name="itemsToInsert", source="/document/normalized_images/*/text"),  # Example field
                InputFieldMappingEntry(name="offsets", source="/document/normalized_images/*/contentOffset")  # Example field
            ],
            outputs=[
                OutputFieldMappingEntry(name="mergedText", target_name="merged_text")
            ]
        )

        # Define the ImageAnalysisSkill
        image_analysis_skill = ImageAnalysisSkill(
            name="image-analysis-skill",
            context=content_context,
            inputs=[InputFieldMappingEntry(name="image", source="/document/normalized_images/*")],  # Add inputs parameter
            visual_features=["tags", "description"],
            outputs=[
                OutputFieldMappingEntry(name="categories", target_name="categories"),
                OutputFieldMappingEntry(name="tags", target_name="tags"),
                OutputFieldMappingEntry(name="description", target_name="description"),
                OutputFieldMappingEntry(name="faces", target_name="faces")
            ]
        )

        # Define the knowledge store
        knowledge_store = SearchIndexerKnowledgeStore(
            storage_connection_string=get_knowledge_store_connection_string(config),
            projections=[
                SearchIndexerKnowledgeStoreProjection(
                    objects=[SearchIndexerKnowledgeStoreFileProjectionSelector(
                        storage_container=chunk_index_blob_container_name,
                        generated_key_name="id",
                        source_context=f"{content_context}/chunks/*",
                        inputs=[
                            InputFieldMappingEntry(name="source_document_id", source="/document/document_id"),
                            InputFieldMappingEntry(name="source_document_filepath", source="/document/filepath"),
                            InputFieldMappingEntry(name="source_field_name", source=f"{content_context}/chunks/*/embedding_metadata/fieldname"),
                            InputFieldMappingEntry(name="title", source=f"{content_context}/chunks/*/title"),
                            InputFieldMappingEntry(name="text", source=f"{content_context}/chunks/*/content"),
                            InputFieldMappingEntry(name="embedding", source=f"{content_context}/chunks/*/embedding_metadata/embedding"),
                            InputFieldMappingEntry(name="index", source=f"{content_context}/chunks/*/embedding_metadata/index"),
                            InputFieldMappingEntry(name="offset", source=f"{content_context}/chunks/*/embedding_metadata/offset"),
                            InputFieldMappingEntry(name="length", source=f"{content_context}/chunks/*/embedding_metadata/length")
                        ]
                    )]
                ),
                SearchIndexerKnowledgeStoreProjection(
                    files=[SearchIndexerKnowledgeStoreFileProjectionSelector(
                        storage_container=f"{chunk_index_blob_container_name}images",
                        generated_key_name="imagepath",
                        source="/document/normalized_images/*",
                        inputs=[]
                    )]
                )
            ]
        )

        # Define the cognitive services account
        cognitiveservicesaccount = CognitiveServicesAccountKey(description="Cognitive Services Account", key=config['AZURE_SEARCH_COGNITIVE_SERVICES_KEY'])

        # Define the skillset
        skillset = SearchIndexerSkillset(
            name=name,
            skills=[embedding_skill], #here more skills can be added
            description=name,
            knowledge_store=knowledge_store,
            cognitive_services_account=cognitiveservicesaccount
        )

        # Create the skillset using the indexer client
        client = get_indexer_client(config)
        return client.create_skillset(skillset)

    def _create_document_indexer(self, index_prefix, data_source_name, index_name, skillset_name, config, content_field_name="content", generate_page_images=True):
        """
        Creates an indexer in Azure Search with the given index_prefix, data_source_name, index_name, skillset_name, config, content_field_name, and generate_page_images.

        Args:
            index_prefix (str): The prefix to use for the indexer name.
            data_source_name (str): The name of the data source to use for the indexer.
            index_name (str): The name of the index to use for the indexer.
            skillset_name (str): The name of the skillset to use for the indexer.
            config (dict): The configuration for the Azure Search service.
            content_field_name (str): The name of the content field to use for the indexer. Defaults to "content".
            generate_page_images (bool): Whether to generate normalized images for each page of the document. Defaults to True.

        Returns:
            Indexer: The created indexer.
        """
        # Get the name for the indexer
        name = get_indexer_name(index_prefix)

        # Define the indexer configuration based on the generate_page_images parameter
        indexer_config = {"dataToExtract": "contentAndMetadata", "imageAction": "generateNormalizedImagePerPage"} if generate_page_images else {"dataToExtract": "contentAndMetadata"}

        # Define the indexing parameters
        parameters = IndexingParameters(max_failed_items=-1, configuration=indexer_config)

        # Define the field mappings for the indexer
        field_mappings = [
            FieldMapping(source_field_name="metadata_storage_path", target_field_name="document_id", mapping_function=FieldMappingFunction(name="base64Encode", parameters=None)),
            FieldMapping(source_field_name="metadata_storage_name", target_field_name="filepath"),
            FieldMapping(source_field_name="metadata_storage_size", target_field_name="filesize")
        ]

        # Define the output field mappings for the indexer
        output_field_mappings = []

        # Create the indexer using the custom utility function
        indexer = SearchIndexer(
            name=name,
            data_source_name=data_source_name,
            target_index_name=index_name,
            skillset_name=skillset_name,
            field_mappings=field_mappings,
            output_field_mappings=output_field_mappings,
            parameters=parameters
        )
        indexer_client = get_indexer_client(config)
        return indexer_client.create_indexer(indexer)

    def create_document_index_resources(self, index_prefix, customer_storage_connection_string, customer_container_name, config) -> dict:
        """
        Creates the necessary resources for a document index in Azure Search with the given index_prefix, customer_storage_connection_string, customer_container_name, and config.

        Args:
            index_prefix (str): The prefix to use for the index, data source, indexer, and skillset names.
            customer_storage_connection_string (str): The connection string for the customer's storage account.
            customer_container_name (str): The name of the container in the customer's storage account.
            config (dict): The configuration for the Azure Search service.

        Returns:
            dict: A dictionary containing the names of the created index, data source, indexer, and skillset.
        """
        # Create the index, data source, skillset, and indexer using the custom utility functions
        index_name = self._create_document_index(index_prefix, config).name
        data_source_name = self._create_document_datasource(index_prefix, customer_storage_connection_string, customer_container_name, config).name
        skillset_name = self._create_document_skillset(index_prefix, config).name
        time.sleep(5)
        indexer_name = self._create_document_indexer(index_prefix, data_source_name, index_name, skillset_name, config=config).name
        wait_for_indexer_completion(indexer_name, config=config)

        # Return a dictionary containing the names of the created index, data source, indexer, and skillset
        return {"index_name": index_name, "data_source_name": data_source_name, "skillset_name": skillset_name, "indexer_name": indexer_name}

    def delete_document_index_resources(self, index_prefix, config):
        """
        Deletes the resources for a document index in Azure Search with the given index_prefix and config.

        Args:
            index_prefix (str): The prefix used for the index, data source, indexer, and skillset names.
            config (dict): The configuration for the Azure Search service.
        """
        # Get the index and indexer clients using the custom utility functions
        index_client = get_index_client(config)
        indexer_client = get_indexer_client(config)

        # Delete the index, indexer, data source, and skillset using the corresponding client methods
        index_client.delete_index(index=get_index_name(index_prefix))
        indexer_client.delete_indexer(indexer=get_indexer_name(index_prefix))
        indexer_client.delete_data_source_connection(data_source_connection=get_datasource_name(index_prefix))
        indexer_client.delete_skillset(skillset=get_skillset_name(index_prefix))

        # Delete the knowledge store tables and blobs
        knowledge_store_connection_string = get_knowledge_store_connection_string()

        # Delete the container directly from storage
        try:
            blob_service = BlobServiceClient.from_connection_string(knowledge_store_connection_string)
            blob_service.delete_container(get_chunk_index_blob_container_name(index_prefix))
        except ResourceNotFoundError:
            # Handle resource not found error
            pass
