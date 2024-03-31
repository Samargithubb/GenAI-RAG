import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import HumanMessage
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import sys
import warnings
warnings.filterwarnings('ignore')


load_dotenv()
vectordb_path = "Vector_db"
resumes_path = "docs"
huggingface_api = os.getenv("hugging_api")


class GenerateResponse:
    
    def split_docs(self, documents, chunk_size=4000, chunk_overlap=100):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(documents)
            return docs
        except Exception as e:
            raise RuntimeError(f"An error occurred in split doc function: {str(e)}")

    def process_rag_system(self, job_description):
        try:
            self.directory = resumes_path
            self.persist_directory = vectordb_path

            if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
                if not os.path.exists(self.persist_directory):
                    os.makedirs(self.persist_directory)

                self.loader = DirectoryLoader(self.directory, show_progress=True)
                self.documents = self.loader.load()
                self.docs = self.split_docs(self.documents)

                self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

                self.vectordb = Chroma.from_documents(documents=self.docs,
                                                  embedding=self.embeddings,
                                                  persist_directory=self.persist_directory)
                self.vectordb.persist()
            else:
                self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vectordb = Chroma(persist_directory=self.persist_directory,
                                       embedding_function=self.embeddings)

            self.vectorstore_retriever = self.vectordb.as_retriever(
                search_kwargs={
                    "k": 1
                }
            )

            self.turbo_llm = HuggingFaceHub(repo_id="google/flan-t5-large",  model_kwargs={"temperature":0.5, "max_length":512},
                                                huggingfacehub_api_token=huggingface_api)
            self.qa_chain = RetrievalQA.from_chain_type(llm=self.turbo_llm,
                                                            chain_type="stuff",
                                                            retriever=self.vectorstore_retriever,
                                                            return_source_documents=True)
            warning = "Please refrain from speculating if you're unsure. Simply state that you don't know. Answers should be concise, within 200 words."
            question = warning + "you are a AI bot. Your Job is to find the Job based on the Given requirement"
            query = question + " Requirement: " + job_description
            llm_response = self.qa_chain(query)
            llm_result = llm_response

            return llm_result

        except Exception as e:
            raise RuntimeError(f"An error occurred while processing answers: {str(e)}")


def main():
    response_generator = GenerateResponse()
    st.image("leadspace.jpeg")
    st.title("Find Job with RAG (AI)")
    job_description = st.text_area("Describe the Job You want:")
    
    if st.button("Generate"):
        llm_result = response_generator.process_rag_system(job_description)
        st.write('LLM Result: ',llm_result['result'])
        for source in llm_result["source_documents"]:
            st.write(source.page_content)


if __name__ == "__main__":
    main()
