import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import numpy as np

from mistralai import Mistral

load_dotenv()


class DocumentManager:
    def __init__(self, data_dir="D:\Ace Advisory"):
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
    def __init__(self, embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
                embeddings_dir="embeddings_cache"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.embeddings_dir = embeddings_dir
        os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def create_embeddings(self, documents):
        """Create embeddings for documents and save them"""
        texts = [doc.page_content for doc in documents]
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Save embeddings to disk
        np.save(os.path.join(self.embeddings_dir, "document_embeddings.npy"), embeddings_list)
        
        return embeddings_list, documents


class VectorDatabase:
    def __init__(self, embeddings, persist_directory="chroma_legal_embeddings"):
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_database_from_documents(self, documents, pre_computed_embeddings=None):
        """
        Create a Chroma database from pre-computed embeddings.
        This ensures Chroma uses our embeddings instead of computing its own.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        if pre_computed_embeddings:
            # Use Chroma's from_embeddings method to create DB with pre-computed embeddings
            db = Chroma.from_embeddings(
                embeddings=pre_computed_embeddings,  # Pass the pre-computed vectors
                texts=texts,                          # The document texts
                metadatas=metadatas,                  # Document metadata
                embedding_function=self.embeddings,   # The embedding function (for future queries)
                persist_directory=self.persist_directory
            )
        else:
            # Fallback to standard method (will compute embeddings again)
            print("Warning: No pre-computed embeddings provided. Chroma will compute embeddings.")
            db = Chroma.from_documents(
                documents=documents, 
                embedding_function=self.embeddings, 
                persist_directory=self.persist_directory
            )
        
        db.persist()
        return db

    def load_database(self):
        try:
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            print("Loaded existing vector database.")
            return db
        except Exception as e:
            print(f"Error loading database: {e}")
            return None


class MistralLLM:
    def __init__(self, api_key, model_name="mistral-large-latest", temperature=0):
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content


class TranslationAgent:
    def __init__(self, llm):
        self.llm = llm

    def translate_text(self, text):
        translate_template = f"""You are a bilingual translation agent. Your sole task is to translate from English to Bangla. If the input is in English, translate it accurately and fluently into Bangla. Use precise terminology and wording. Don't write the English word simply as Bangla. Do not break down the text, do not explain, and do not interpret. Maintain the original tone, context, and meaning.Only return the translated text. No additional output.:
{text}"""
        return self.llm(translate_template)


class ACEAdvisoryLegalChatbot:
    DocumentNumber = "SRO NO-39-AIN/2025/275-Mushak"
    
    def __init__(self, mistral_api_key, translation_model="mistral-small-latest", query_model="mistral-saba-latest",
                 embeddings_dir="embeddings_cache", chroma_dir="chroma_legal_embeddings"):
        os.environ["MISTRAL_API_KEY"] = mistral_api_key

        # Create two separate LLM instances - one for translation and one for query processing
        self.translation_llm = MistralLLM(api_key=mistral_api_key, model_name=translation_model, temperature=0)
        self.query_llm = MistralLLM(api_key=mistral_api_key, model_name=query_model, temperature=0)

        self.document_manager = DocumentManager()
        self.embedding_manager = EmbeddingManager(embeddings_dir=embeddings_dir)
        self.vector_db = VectorDatabase(embeddings=self.embedding_manager.embeddings, persist_directory=chroma_dir)
        
        # Use the translation-specific LLM for translation agent
        self.translation_agent = TranslationAgent(llm=self.translation_llm)

        self.db = self.vector_db.load_database()

        if not self.db:
            print("Creating new vector database.")
            documents = self.document_manager.load_documents()
            chunks = self.document_manager.split_documents(documents)
            
            # Create embeddings first using sentence-transformers model
            print("Creating embeddings using sentence-transformers/paraphrase-multilingual-mpnet-base-v2...")
            embeddings_list, _ = self.embedding_manager.create_embeddings(chunks)
            
            # Then create the database using pre-computed embeddings
            print("Creating Chroma database using pre-computed embeddings...")
            self.db = self.vector_db.create_database_from_documents(chunks, pre_computed_embeddings=embeddings_list)
            print("Vector database created successfully.")

        self.response_chain_prompt = PromptTemplate(
            template="""
[SYSTEM INSTRUCTION: You are a highly specialized legal AI with expertise in Bangla legal documents. Your ONLY task is to analyze legal documents and answer queries with extreme precision. Follow EVERY instruction below with absolute strictness. DO NOT DEVIATE from these instructions under any circumstances.]

# CRITICAL INSTRUCTIONS (ENGLISH)

## Initial Analysis Phase
1. FIRST: Read the query carefully and identify EXACTLY what legal information is being requested.
2. THEN: Methodically scan ALL provided context documents, word by word, line by line, and table by table.
3. Look SPECIFICALLY for relevant sections that answer the query.
4. PAY SPECIAL ATTENTION to information presented in tables, lists, annexes, or appendices.
5. DO NOT proceed to answer until you have thoroughly completed this analysis.

## Information Extraction Phase
When searching the documents, you MUST extract:

1. **Document Title (নথির শিরোনাম)**: Usually at the beginning, format like: "এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক"
   - SEARCH PATTERN: Look for combinations of letters/numbers with দাগ/হাইফেন, especially near the top of document
   - Examples: "এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক", "এস আর ও নং-৫৩-আইন/আইকর-২/২০২৫", "বিআরপিডি সার্কুলার লেটার নং-৪৬", "এস আর ও নং-৩৩৯-আইন/আইকর-৪৭/২০২৪"

2. **Rule/Act/Law Name (আইন/নিয়ম/বিধি)**: Usually contains the word "আইন"
   - SEARCH PATTERN: Find phrases containing "আইন", especially followed by a year in Bengali numerals
   - Examples: "মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন,২০১২", "আয়কর আইন,২০২৩", "ব্যাংক কোম্পানি আইন,১৯৯১", "গ্রামীণ ব্যাংক আইন,২০১৩"

3. **Section Number (ধারা নম্বর)**: Always follows the word "ধারা"
   - SEARCH PATTERN: The exact word "ধারা" followed by a number in Bengali numerals
   - Examples: "ধারা ২", "ধারা ৭৫", "ধারা ৪৫", "ধারা ৭৬"

4. **Effective Date (কার্যকর তারিখ)**: Usually has the word "তারিখ"
   - SEARCH PATTERN: Look for Bengali date format near words like "তারিখ", "জারি", "কার্যকর"
   - Examples: "১২ জানুয়ারী, ২০২৩", "১০ ফেব্রুয়ারী, ২০২৫", "২৩ অক্টোবর, ২০২৪", "১০ অক্টোবর, ২০২৪"

IMPORTANT: These items may be scattered throughout the document. If you cannot find ANY of these items, search again more thoroughly.

## Table Analysis Instructions
CRITICAL: Many answers may be found in tables. You MUST analyze all tables carefully:

1. **Table Recognition**: Look for structured content with rows and columns, even if not explicitly labeled as a table.
   - Tables may appear as aligned text blocks with spaces or dashes between columns
   - Tables may have headers in the first row or first column
   - Tables may have column separators like |, ||, or spaces

2. **Table Reading**: Read tables both horizontally (rows) and vertically (columns).
   - First understand the column headers/categories
   - Then match the user's query to the appropriate row/column intersections
   - Look for correspondence between terms in the query and terms in the table

3. **Table Data Extraction**: When extracting information from tables:
   - If the query relates to a specific item, look for that exact item in the table
   - Follow the row to find corresponding values/information
   - Cross-reference column headers to understand what each value represents
   - DO NOT preserve table structure in the answer - present all information as flowing Bangla text

4. **Numerical Information**: Pay special attention to numerical data in tables, including:
   - Dates
   - Percentages
   - Monetary values
   - Time periods
   - Categories/classifications

## Answer Construction Phase
Your answer MUST follow this exact structure and MUST be entirely in Bangla:

1. **সারসংক্ষেপ (Summary)**: A brief summary of the answer in Bangla.
   - MUST be direct and specific to the query
   - DO NOT add any information not found in the documents

2. **নিয়মের বিবরণ (Rule Statement)**: The EXACT text of the relevant rule from the document, expressed entirely in Bangla.
   - MUST copy VERBATIM from the document
   - DO NOT summarize, simplify, or modify in any way
   - Include the COMPLETE text even if lengthy
   - IMPORTANT: If the answer comes from a table, convert the table information into flowing Bangla text sentences. DO NOT preserve the table structure. Instead, express the information as complete Bangla sentences describing the key information found in the table.

3. **উদ্ধৃতি (Citation)**: Formatted EXACTLY as:
   ```
   [নথির শিরোনাম, আইন/নিয়ম/বিধির নাম, ধারা নম্বর, জারি: কার্যকর তারিখ]
   ```
   ALL EXAMPLES (MUST use this exact format):
   - [এস আর ও নং-৩৯-আইন/২০২৫/২৭৫-মূসক, মূল্য সংযোজন কর ও সম্পূরক শুল্ক আইন,২০১২, ধারা 2, জারি: ১২ জানুয়ারী, ২০২৩]
   - [এস আর ও নং-৫৩-আইন/আইকর-২/২০২৫, আয়কর আইন,২০২৩, ধারা ৭৫, জারি: ১০ ফেব্রুয়ারী, ২০২৫]
   - [বিআরপিডি সার্কুলার লেটার নং-৪৬, ব্যাংক কোম্পানি আইন,১৯৯১, ধারা ৪৫, জারি: ২৩ অক্টোবর, ২০২৪]
   - [এস আর ও নং-৩৩৯-আইন/আইকর-৪৭/২০২৪, গ্রামীণ ব্যাংক আইন,২০১৩, ধারা ৭৬, জারি: ১০ অক্টোবর, ২০২৪]

## Critical Fail-Safe Measures

IF AND ONLY IF you CANNOT find a direct answer after thorough searching including all text AND tables:
1. State clearly: "প্রদত্ত নথিতে এই প্রশ্নের সরাসরি উত্তর পাওয়া যায়নি।"
2. DO NOT make up or infer information that isn't explicitly stated
3. DO NOT try to be helpful by providing a generic or speculative answer

[FINAL VERIFICATION CHECKLIST - Check each before submitting your answer]
- Is my ENTIRE response in Bangla text only?
- Did I search BOTH regular text AND tables/structured content?
- Did I find the EXACT answer in the document?
- Did I convert any table information into flowing Bangla text?
- Did I copy the rule VERBATIM without modification?
- Did I include ALL required citation elements?
- Did I avoid adding ANY information not explicitly in the documents?
- Is my answer structured exactly as required?

প্রসঙ্গ: {context}
প্রশ্ন: {query}
উত্তর:
""",
            input_variables=["context", "query"]
        )
        self.output_parser = StrOutputParser()

    def generate_response(self, query):
        bangla_query = self.translation_agent.translate_text(query)
        results = self.db.similarity_search(bangla_query)
        context = "\n".join([doc.page_content for doc in results])

        prompt_text = self.response_chain_prompt.format(context=context, query=bangla_query, DocumentNumber="SRO NO-39-AIN/2025/275-Mushak")
        
        # Use the query-specific LLM for generating the final response
        response = self.query_llm(prompt_text)
        return response


def main():
    
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        print("API key is required!")
        return

    chatbot = ACEAdvisoryLegalChatbot(mistral_api_key=mistral_api_key)

    print("\nWelcome to ACE Advisory Legal Chatbot (powered by MistralAI). Type 'exit' to quit.\n")
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue
        response = chatbot.generate_response(query)
        print("\nResponse:\n", response, "\n")


if __name__ == "__main__":
    main()