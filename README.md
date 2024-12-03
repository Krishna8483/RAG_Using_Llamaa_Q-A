RAG Using Llama Model for Q&A with PDF Documents

This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) system using the Llama 2 13B LLM (quantized version) with LangChain. The system is designed to efficiently answer queries from PDF documents by retrieving relevant context from the PDF and generating accurate responses.
Features

    Retrieval-Augmented Generation (RAG): Combines information retrieval with generative language models to provide precise and context-aware answers.
    Integration with LangChain: Enables seamless chaining of LLM tasks such as retrieval, summarization, and question answering.
    PDF Support: Parses content from PDF documents for query-answering purposes.
    Efficient Retrieval: Uses vector embeddings to retrieve relevant sections from large documents.
    Scalable: Works effectively with extensive document repositories.

Workflow

    PDF Parsing: The uploaded PDF is converted into text.
    Text Embedding: The content is split into chunks and converted into vector embeddings.
    Retrieval: Based on the user's query, relevant text chunks are retrieved using a similarity search.
    Generation: The Llama 2 model processes the retrieved context and generates a response.

Technologies Used

    Llama 2 (13B, Quantized): OpenAI's large language model for natural language generation.
    LangChain: Framework for building applications with LLMs.
    Vector Database: For storing and retrieving text embeddings (e.g., Pinecone or FAISS).
    PyPDF2 / pdfplumber: For extracting text from PDF files.
    Python Libraries: transformers, sentence-transformers, langchain, openai.

Prerequisites

    Python 3.8 or later.
    Install required libraries:

    pip install langchain llama-index pypdf2 transformers pinecone-client sentence-transformers

    Access to the Llama 2 model (use Hugging Face or other providers).
    Vector database setup (e.g., Pinecone or FAISS).


Configure API Keys:

    Set up your  Hugging Face API key for accessing the Llama 2 model.
    Configure your  FAISS vector database.

Run the Application:

    Start the application by executing:

        python app.py

    Upload PDF:
        Upload a PDF document via the interface.
        Enter a query to retrieve context-aware answers.

Example Queries

    "What is the main idea of the first section of the document?"
    "Summarize the findings in the PDF."
    "What are the key takeaways from the research study?"

Future Enhancements

    Support for additional document types (e.g., Word, Excel).
    Enhanced multi-PDF support for querying across multiple files.
    Deployment as a web app using Streamlit or FastAPI.

