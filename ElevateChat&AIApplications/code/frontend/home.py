import streamlit as st
import base64
import requests
import json

from PIL import Image
from io import BytesIO

st.set_page_config(page_title='💬 Liantis POC')


def add_company_logo_and_welcome_text():
    """
    This function displays the company logo in the center of the page.
    """
    customer_logo = Image.open('yourlogo.png')

    # Center the logo using CSS style
    st.write(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/png;base64,{image_to_base64(customer_logo)}" width="300">'
        '</div>',
        unsafe_allow_html=True
    )

    # Define company name to be used for welcome title --> should be a parameter
    company_name = 'YourCompanyName'
    # Page title and subtitle
    st.markdown("<h2 style='text-align: center; color: black;'>Welcome to the {} chatbot</h2>".format(company_name), unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>Your AI-powered copilot, check some samples</h5> ", unsafe_allow_html=True)


def image_to_base64(image):
    """
    Convert an image to its base64 representation.
    """

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



def askdocuments(
    url,
    AZURE_SEARCH_ADMIN_KEY,
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_SERVICE_NAME,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
    AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    OPENAI_API_TYPE,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_MODEL_NAME,
    NUMBER_OF_CHUNKS_TO_RETURN,
        question):

    try:
        # Prepare the request data
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'AZURE_SEARCH_ADMIN_KEY': AZURE_SEARCH_ADMIN_KEY,
            'AZURE_SEARCH_SERVICE_ENDPOINT': AZURE_SEARCH_SERVICE_ENDPOINT,
            'AZURE_SEARCH_SERVICE_NAME': AZURE_SEARCH_SERVICE_NAME,
            'AZURE_SEARCH_INDEX_NAME': AZURE_SEARCH_INDEX_NAME,
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME': AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
            'AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL': AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL,
            'AZURE_OPENAI_API_VERSION': AZURE_OPENAI_API_VERSION,
            'AZURE_OPENAI_ENDPOINT': AZURE_OPENAI_ENDPOINT,
            'AZURE_OPENAI_API_KEY': AZURE_OPENAI_API_KEY,
            'OPENAI_API_TYPE': OPENAI_API_TYPE,
            'OPENAI_DEPLOYMENT_NAME': OPENAI_DEPLOYMENT_NAME,
            'OPENAI_MODEL_NAME': OPENAI_MODEL_NAME,
            'NUMBER_OF_CHUNKS_TO_RETURN': NUMBER_OF_CHUNKS_TO_RETURN,
            'question': question
        }

        # Make the API call and extract the documents from the response
        response = requests.get(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        answer = json.loads(response.content)
        return answer

    except requests.exceptions.RequestException as e:
        # Handle any requests-related errors (e.g., network issues, invalid URL)
        raise ValueError(f"Error with the API request: {e}")

    except json.JSONDecodeError as e:
        # Handle any JSON decoding errors (e.g., invalid JSON format)
        raise ValueError(f"Error decoding API response as JSON: {e}")


def main():
    add_company_logo_and_welcome_text()

    st.markdown('#')



    user_choice = st.radio("Select an index:", ('<yourownsetting>', '<yourownsetting>'))

    with st.sidebar:
        st.subheader("Settings for Ask your documents")
        st.write("Enter your Azure Search API key, endpoint, and index name.")
        url = st.text_input("Enter your REST API URL", value="http://localhost:7071/api/AskYourDocuments")

        AZURE_SEARCH_ADMIN_KEY = st.text_input("Enter your AZURE_SEARCH_ADMIN_KEY", type="password", value="<yourownsetting>")
        AZURE_SEARCH_SERVICE_ENDPOINT = st.text_input("Enter your AZURE_SEARCH_SERVICE_ENDPOINT", value="<yourownsetting>t")
        AZURE_SEARCH_SERVICE_NAME = st.text_input("Enter your AZURE_SEARCH_SERVICE_NAME", value="<yourownsetting>")
        AZURE_SEARCH_INDEX_NAME = st.text_input("Enter your AZURE_SEARCH_INDEX_NAME", value="<yourownsetting>")
        AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = st.text_input("Enter your AZURE_SEARCH_INDEX_NAME", value="text-embedding-ada-002")
        AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL = st.text_input("Enter your AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL", value="text-embedding-ada-002")
        AZURE_OPENAI_API_VERSION = st.text_input("Enter your AZURE_OPENAI_API_VERSION", value="2023-03-15-preview")
        AZURE_OPENAI_ENDPOINT = st.text_input("Enter your AZURE_OPENAI_ENDPOINT", value="<yourownsetting>")
        AZURE_OPENAI_API_KEY = st.text_input("Enter your AZURE_OPENAI_API_KEY", type="password", value="<yourownsetting>")
        OPENAI_API_TYPE = st.text_input("Enter your OPENAI_API_TYPE", value="azure")
        OPENAI_DEPLOYMENT_NAME = st.text_input("Enter your OPENAI_DEPLOYMENT_NAME", value="chat")
        OPENAI_MODEL_NAME = st.text_input("Enter your OPENAI_MODEL_NAME", value="gpt-35-turbo")
        NUMBER_OF_CHUNKS_TO_RETURN = st.text_input("Enter your NUMBER_OF_CHUNKS_TO_RETURN", value="3")

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = askdocuments(url=url,
                                        AZURE_SEARCH_ADMIN_KEY=AZURE_SEARCH_ADMIN_KEY,
                                        AZURE_SEARCH_SERVICE_ENDPOINT=AZURE_SEARCH_SERVICE_ENDPOINT,
                                        AZURE_SEARCH_SERVICE_NAME=AZURE_SEARCH_SERVICE_NAME,
                                        AZURE_SEARCH_INDEX_NAME=AZURE_SEARCH_INDEX_NAME,
                                        AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
                                        AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL,
                                        AZURE_OPENAI_API_VERSION=AZURE_OPENAI_API_VERSION,
                                        AZURE_OPENAI_ENDPOINT=AZURE_OPENAI_ENDPOINT,
                                        AZURE_OPENAI_API_KEY=AZURE_OPENAI_API_KEY,
                                        OPENAI_API_TYPE=OPENAI_API_TYPE,
                                        OPENAI_DEPLOYMENT_NAME=OPENAI_DEPLOYMENT_NAME,
                                        OPENAI_MODEL_NAME=OPENAI_MODEL_NAME,
                                        NUMBER_OF_CHUNKS_TO_RETURN=NUMBER_OF_CHUNKS_TO_RETURN,
                                        question=prompt
                                        )

                st.write(response['result'])

                # Display each document title with a clickable link
                for idx, doc in enumerate(response['source_documents']):
                    doc_title = doc["title"]
                    doc_page_content = doc.get("page_content", "Content not available")

                    # Display the title as a clickable link
                    button_key = f"doc_button_{idx}"
                    if st.button(doc_title, key=button_key, help=doc_page_content):
                        pass  # You can add additional code here if needed

        message = {"role": "assistant", "content": response['result']}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
