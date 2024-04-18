# -*- coding: utf-8 -*-
"""MindCase_AdityaSharma_Assignmnent

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eNbqtm4AGZcYCrv8gwGCjTf0zvQiRvs_
"""

!pip install -q pypdf
!pip install torch
!pip install -q transformers
!pip -q install sentence-transformers
!pip install -q llama-index

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install  llama-cpp-python --no-cache-dir

!pip install llama-index-embeddings-huggingface
!pip install llama-index-llms-llama-cpp

!pip install cuda

import torch
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
# Initializing LlamaCPP with a pretrained model.
llm = LlamaCPP(
    # I provide the option to specify the URL of a GGML model for automatic download.
    # The model_url parameter can be uncommented and filled accordingly.
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',

    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,

    # I limit the context window to allow for some flexibility.
    context_window=4096,

    # Additional arguments to be passed to the __call__() method.
    generate_kwargs={},

    # Additional arguments to be passed to the __init__() method.
    # Setting n_gpu_layers to -1 enables GPU usage.
    model_kwargs={"n_gpu_layers": -1},

    # Transforming inputs into Llama2 format.
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

# Loading data from the specified input file.
documents = SimpleDirectoryReader(
    input_files = ["/content/Mindcase Data.pdf"]
).load_data()

# Combining text from all loaded documents into a single Document object.
documents = Document(text = "\n\n".join([doc.text for doc in documents]))

import os
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, ServiceContext, load_index_from_storage
from llama_index.core import StorageContext
def get_build_index(documents,llm,embed_model="local:BAAI/bge-small-en-v1.5",sentence_window_size=3,save_dir="./vector_store/index"):
  # Configuring the node parser for extracting sentence windows.
  node_parser = SentenceWindowNodeParser(
      window_size = sentence_window_size,
      window_metadata_key = "window",
      original_text_metadata_key = "original_text"
  )

  # Setting up the service context with necessary components.
  sentence_context = ServiceContext.from_defaults(
      llm = llm,
      embed_model= embed_model,
      node_parser = node_parser,
  )

  if not os.path.exists(save_dir):
        # If the save directory doesn't exist, create and load the index.
        index = VectorStoreIndex.from_documents(
            [documents], service_context=sentence_context
        )
        index.storage_context.persist(persist_dir=save_dir)
  else:
      # If the save directory exists, load the existing index.
      index = load_index_from_storage(
          StorageContext.from_defaults(persist_dir=save_dir),
          service_context=sentence_context,
      )

  return index

# Obtain the vector index.
vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3, save_dir="./vector_store/index")

from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
  # Configuring postprocessors for metadata replacement and sentence reranking.
  postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
  rerank = SentenceTransformerRerank(
      top_n=rerank_top_n, model="BAAI/bge-reranker-base"
  )
  # Creating a query engine from the sentence index with specified parameters.
  engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
  )

  return engine

query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=6, rerank_top_n=2)

"""**Example of queries that should be answered:**

- Explain the theme of the movie?
- Who are the characters?
- How many male and female characters are in the movie?
- Does the script pass the Bechdel test?
- What is the role of Deckard in the movie?
"""

import sys

# Define a function to read input with timeout
def input_with_timeout(prompt, timeout):
    print(prompt, end='', flush=True)
    try:
        response = input()
        return response
    except KeyboardInterrupt:
        sys.exit(0)

# Define a flag to indicate whether the loop should continue running.
running = True

while running:
    try:
        # Input a query from the user.
        query = input("Enter your query (Press 'Esc' to exit): ")

        # Check if the input is 'Esc' to exit.
        if query.lower() == 'esc':
            print("Exiting the loop...")
            running = False
            continue

        # Query the engine with the input query.
        response = query_engine.query(query)

        # Print the response.
        print(response)
        print("\n")

    except KeyboardInterrupt:
        # If the user presses Ctrl+C, exit the loop.
        print("\nExiting the loop...")
        running = False

"""# Techniques Used

In the work above, I utilized several techniques and libraries to perform various tasks efficiently. Here's an overview of the key techniques used:

1. **LLAMA Index Integration:** The code integrates with the LLAMA Index library to handle document indexing and querying tasks. LLAMA Index provides efficient methods for processing and querying large collections of text data.

2. **PyTorch:** The PyTorch library is employed for handling tensor computations and machine learning tasks. It is used in conjunction with LLAMA Index for certain operations such as embedding generation and similarity calculations.

3. **Sentence Embeddings:** The code leverages pre-trained sentence embeddings to represent textual data in a high-dimensional vector space. These embeddings capture semantic information from text, enabling similarity comparisons and retrieval of relevant documents.

4. **Query Engine:** A query engine is constructed to facilitate user queries on the indexed documents. The engine utilizes similarity calculations and post-processing techniques to retrieve the most relevant results based on the user's input.

5. **Metadata Processing:** Metadata replacement and sentence reranking are applied as post-processing steps to enhance the quality of search results. These techniques involve modifying metadata associated with document nodes and reordering search results based on additional criteria.

By incorporating these techniques, the code demonstrates effective methods for document indexing, retrieval, and semantic search, making it suitable for various information retrieval tasks.

"""
