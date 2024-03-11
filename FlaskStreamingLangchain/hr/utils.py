import threading
from langchain import callbacks
from langsmith import Client
from langchain_openai import AzureChatOpenAI
from config.config import (OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION, 
                            OPENAI_DEPLOYMENT_NAME, OPENAI_API_TYPE, OPENAI_MODEL_NAME,
                            ANSWER_PROMPT, SEARCH_SERVICE_ADMIN_KEY, SEARCH_SERVICE_ENPOINT,
                            AZURE_SEARCH_INDEX_NAME, 
                            OPENAI_EMBEDDING_DEPLOYMENT_NAME, DOCUMENT_PROMPT)
from handlers.ChainStreamHandler import ChainStreamHandler
from utils.ThreadedGenerator import ThreadedGenerator
from langchain_core.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate,
                                         SystemMessagePromptTemplate)
from langchain.vectorstores import AzureSearch 
from langchain_openai import AzureOpenAIEmbeddings
from retrievers.YourRetriever import YourRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

def setup_embeddings():
    """
    Set up embeddings.

    Returns:
        AzureOpenAIEmbeddings: Instance of AzureOpenAIEmbeddings.

    """
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,    
        azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version="2023-05-15",
    )
    
def setup_vector_store(embeddings):
    """
    Set up the vector store.

    Args:
        embeddings (AzureOpenAIEmbeddings): Instance of AzureOpenAIEmbeddings.

    Returns:
        AzureSearch: Instance of AzureSearch.

    """
    return AzureSearch(
        azure_search_endpoint=SEARCH_SERVICE_ENPOINT,
        azure_search_key=SEARCH_SERVICE_ADMIN_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query
    )

def setup_memory():
    """
    Set up memory.

    Returns:
        ConversationBufferMemory: Instance of ConversationBufferMemory.

    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
def llm_thread_hrgpt(g, prompt):
    """
    A function to run the language model in a separate thread.

    Args:
        g (ThreadedGenerator): ThreadedGenerator instance for token generation.
        prompt (str): Prompt for the language model.

    """
    try:
        # Initialize AzureChatOpenAI instance for interaction with OpenAI model
        chat = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=OPENAI_API_VERSION,
            deployment_name=OPENAI_DEPLOYMENT_NAME,
            openai_api_key=OPENAI_API_KEY,
            openai_api_type=OPENAI_API_TYPE,
            model_name=OPENAI_MODEL_NAME,
            streaming=True,
            callbacks=[ChainStreamHandler(g)],  # Set ChainStreamHandler as callback
            temperature=0)
        
        # Define system and human message prompts
        messages = [
            SystemMessagePromptTemplate.from_template(ANSWER_PROMPT),
            HumanMessagePromptTemplate.from_template("{question} Please answer in html format"),
        ]
        
        # Set up embeddings, vector store, chat prompt, retriever, memory, and chain
        embeddings = setup_embeddings()
        vector_store = setup_vector_store(embeddings)
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        retriever = YourRetriever(vectorstore=vector_store)
        memory = setup_memory()
        chain = ConversationalRetrievalChain.from_llm(chat, 
            retriever=retriever, 
            memory=memory, 
            verbose=False, 
            combine_docs_chain_kwargs={
                "prompt": chat_prompt, 
                "document_prompt": PromptTemplate(
                    template=DOCUMENT_PROMPT,
                    input_variables=["page_content", "source"]
                )
            }
        )
        client = Client()  # noqa: F841
        with callbacks.collect_runs() as cb:    # noqa: F841
            # Run the chain with the given prompt
            chain.run(prompt)
    finally:
        # Close the generator after use
        g.close()

def hr_gpt_chain(prompt):
    """
    Start a new thread to generate tokens.

    Args:
        prompt (str): Prompt for the language model.

    Returns:
        ThreadedGenerator: ThreadedGenerator instance for token generation.

    """
    g = ThreadedGenerator()  # Initialize ThreadedGenerator
    # Start a new thread to run llm_thread_hrgpt function
    threading.Thread(target=llm_thread_hrgpt, args=(g, prompt)).start()
    return g  # Return the ThreadedGenerator instance
