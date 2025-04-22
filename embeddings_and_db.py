import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentManager:
    def __init__(self, data_dir="D:\\Ace Advisory"):
        self.data_dir = data_dir

    def load_documents(self):
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    loader = PyPDFLoader(filepath)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename} with PyPDFLoader: {e}. Trying UnstructuredPDFLoader...")
                    try:
                        loader = UnstructuredPDFLoader(filepath)
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading {filename} with UnstructuredPDFLoader: {e}. Skipping file.")
        return documents

    def split_documents(self, documents, chunk_size=1300, chunk_overlap=300):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)


class EmbeddingManager:
    def __init__(self, embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def embed_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)


class VectorDatabase:
    def __init__(self, persist_directory="chroma_legal_embeddings", embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_database(self, documents, embeddings):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        db = Chroma.from_embeddings(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        db.persist()
        print(f"Vector database created and persisted at {self.persist_directory}")
        return db

    def load_database(self):
        try:
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            print("Loaded existing vector database.")
            return db
        except Exception as e:
            print(f"Error loading database: {e}")
            return None


if __name__ == "__main__":
    # Step 1: Load and split documents
    doc_manager = DocumentManager(data_dir="D:\\Ace Advisory")
    raw_documents = doc_manager.load_documents()
    print(f"Loaded {len(raw_documents)} raw documents")

    split_documents = doc_manager.split_documents(raw_documents)
    print(f"Split into {len(split_documents)} chunks")

    # Step 2: Create embeddings once
    embed_manager = EmbeddingManager()
    embeddings = embed_manager.embed_documents(split_documents)
    print("Created embeddings for all document chunks")

    # Step 3: Create and persist Chroma vector database
    vector_db = VectorDatabase(persist_directory="chroma_legal_embeddings")
    vector_db.create_database(split_documents, embeddings)

    print("Setup complete. You can now load and query the vector database without recomputing embeddings.")