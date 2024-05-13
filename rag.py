from operator import itemgetter

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatMessagePromptTemplate
from langchain.prompts import (
    PromptTemplate,
    # ChatPromptTemplate,
    # MessagesPlaceholder,
    # SystemMessagePromptTemplate,
    # HumanMessagePromptTemplate,
)
from os.path import os
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    # metadata["summary"] = record.get("feedbackText")
    metadata["timestamp"] = record.get("createdDate")
    metadata["totalRevenue"] = record.get("totalRevenue")
    metadata["productNames"] = record.get("productNames")

    return metadata

class ChatCSV:
    vector_store = None
    retriever = None
    memory = None
    chain = None
    db = None
    llm = os.getenv("LLM")
    vllmhost = os.getenv("VLLMHOST")
    token = os.getenv("MAXTOKEN")
    temp = os.getenv("TEMPERATURE")
    
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.getenv("PGVECTOR_DRIVER"),
        host=os.getenv("PGVECTOR_HOST"),
        port=int(os.getenv("PGVECTOR_PORT")),
        database=os.getenv("PGVECTOR_DATABASE"),
        user=os.getenv("PGVECTOR_USER"),
        password=os.getenv("PGVECTOR_PASSWORD"),
    )

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.

        This constructor sets up the following components:
        - A ChatOllama model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """

        self.model = ChatOpenAI(
            model=self.llm,
            openai_api_key="EMPTY",
            openai_api_base=self.vllmhost,
            max_tokens=self.token,
            temperature=self.temp,
        )
        # Initialize the RecursiveCharacterTextSplitter with specific chunk settings.
        # Your tone should be professional and informative
        # self.prompt = ChatPromptTemplate.from_messages(
        #     messages=[
        #         SystemMessagePromptTemplate.from_template("You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative."),
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         HumanMessagePromptTemplate.from_template("{question}"),
        #     ]
        # )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question", output_key="answer",return_messages=True)

    def ingest(self, ingest_path: str, index: bool, type: str):
        '''
        Ingests data from a CSV file containing resumes, process the data, and set up the
        components for further analysis.

        Parameters:
        - csv_file_path (str): The file path to the CSV file.

        Usage:
        obj.ingest("/path/to/data.csv")

        This function uses a CSVLoader to load the data from the specified CSV file.

        Args:
        - file.path (str): The path to the CSV file.
        - encoding (str): The character encoding of the file (default is 'utf-8').
        - source_column (str): The column in the CSV containing the data (default is "Resume").
        '''        
        embeddings=FastEmbedEmbeddings()
        if index:
            print("loading indexes")
            vector_store = PGVector(
                collection_name="rag-nvidia-chat",
                connection_string=self.CONNECTION_STRING,
                embedding_function=embeddings,
            )
            self.retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.7,
                },
            )
        else:
            if type == "web":
                loader = WebBaseLoader(ingest_path)
            elif type == "pdf":
                loader = PyPDFLoader(
                    file_path=ingest_path,
                )
            # loads the data
            data = loader.load()
            # splits the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            all_splits = text_splitter.split_documents(data)
            self.db = PGVector.from_documents(
                embedding=embeddings,
                documents=all_splits,
                collection_name="rag-nvidia-chat",
                connection_string=self.CONNECTION_STRING,
                pre_delete_collection=False,
            )
            # sets up the retriever
            self.retriever = self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.2,
                },
            )

        # Define a processing chain for handling a question-answer scenario.
        # The chain consists of the following components:
        # 1. "context" from the retriever
        # 2. A passthrough for the "question"
        # 3. Processing with the "prompt"
        # 4. Interaction with the "model"
        # 5. Parsing the output using the "StrOutputParser"

        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"]))
            )
                | self.prompt
                | self.model
                | StrOutputParser())
        self.chain = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough(), "chat_history": RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history")}
        ).assign(answer=rag_chain_from_docs)

    def ask(self, query: str):
        """
        Asks a question using the configured processing chain.

        Parameters:
        - query (str): The question to be asked.

        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """
        
        # load memory for history
        self.memory.load_memory_variables({})
        response = self.chain.invoke(query)
        print(response)
        query = {"question": query}
        return response

    def clear(self):
        """
        Clears the components in the question-answering system.

        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        # Set the vector store to None.
        self.vector_store = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None