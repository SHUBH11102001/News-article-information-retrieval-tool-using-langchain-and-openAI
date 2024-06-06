import os
import streamlit as st
from langchain import OpenAI

import pickle
import logging

from dotenv import load_dotenv
import time 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS


# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading the environment variable( our OpenAI API key)
load_dotenv()

# Set up the Streamlit app
st.title("News Research and Analysis Tool")
st.sidebar.title(" Add news Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading Started")
        data = loader.load()

        if not data:
            st.error("No data loaded from URLs. Please check the URLs and try again.")
        else:
            logging.info("Data loaded successfully.")

            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=500  # Adjust chunk size as needed
            )
            main_placeholder.text("Text Splitting Started...")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.error("No documents were created. Please check the document splitting configuration.")
            else:
                logging.info(f"Split {len(docs)} documents successfully.")

                # Display document details
                for i, doc in enumerate(docs):
                    st.write(f"Document {i + 1}:")
                    st.write(doc)

                # Create embeddings and save to FAISS index
                embeddings = OpenAIEmbeddings()
                emb_list = []
                for doc in docs:
                    emb = embeddings.embed_document(doc)
                    emb_list.append(emb)
                    logging.info(f"Generated embedding for document {len(emb_list)}.")

                if not emb_list or len(emb_list[0]) == 0:
                    st.error("No embeddings were generated. Please check the embedding generation process.")
                else:
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)
                    main_placeholder.text("Building Embedding Vector Started")
                    time.sleep(2)

                    # Save the FAISS index to a pickle file
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore_openai, f)
                    logging.info("FAISS index saved successfully.")

    except Exception as e:
        logging.error(f"Error processing URLs: {e}")
        st.error(f"Error processing URLs: {e}")

# Query section
query = main_placeholder.text_input("Question: ")
if query:
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                if 'answer' in result:
                    st.header("Answer")
                    st.write(result["answer"])
                else:
                    st.error("No answer found.")

                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources:")
                    for source in sources:
                        st.write(source)

                # Provide option to download results
                if st.button("Export Results"):
                    with open("query_results.txt", "w") as f:
                        f.write(f"Question: {query}\n")
                        f.write(f"Answer: {result.get('answer', 'No answer found')}\n")
                        f.write("Sources:\n")
                        for source in sources:
                            f.write(f"{source}\n")
                    st.success("Results exported successfully.")

    except Exception as e:
        logging.error(f"Error processing the query: {e}")
        st.error(f"Error processing the query: {e}")


