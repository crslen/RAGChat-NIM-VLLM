from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from os.path import os
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class ChatCSV:
    vector_store = None
    retriever = None
    history_aware_retriever = None
    memory = None
    model = None
    chain = None
    db = None
    llm = os.getenv("LLM")
    api_key = os.getenv("API_KEY")
    vllmhost = os.getenv("VLLMHOST")
    token = os.getenv("MAXTOKEN")
    temp = os.getenv("TEMPERATURE")
    collection_name = os.getenv("COLLECTION_NAME")
    store = {}
    
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
        - A ChatOpenAI model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """
        
        self.model = ChatOpenAI(
            model=self.llm,
            openai_api_key=self.api_key,
            openai_api_base=self.vllmhost,
            max_tokens=self.token,
            temperature=self.temp,
        )

    def load_model(self, model_llm: str):
        print(model_llm)
        self.model = None
        self.__init__()

    def ingest(self, ingest_path: str, index: bool, type: str):
        '''
        Ingests data from a web url or pdf file containing and set up the
        components for further analysis.
        '''      
        embeddings=FastEmbedEmbeddings()
        if index:
            print("loading indexes")
            vector_store = PGVector(
                collection_name=self.collection_name,
                connection=self.CONNECTION_STRING,
                embeddings=embeddings,
                use_jsonb=True,
            )
            self.retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.3,
                },
            )
            print(vector_store.get_collection)
        else:
            if type == "web":
                loader = WebBaseLoader(ingest_path)
            elif type == "pdf":
                loader = PyPDFLoader(
                    file_path=ingest_path,
                )
            elif type == "csv":
                loader = CSVLoader(file_path=ingest_path)
            # loads the data
            data = loader.load()
            # splits the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            all_splits = text_splitter.split_documents(data)
            self.db = PGVector.from_documents(
                embedding=embeddings,
                documents=all_splits,
                collection_name=self.collection_name,
                connection=self.CONNECTION_STRING,
                use_jsonb=True,
                pre_delete_collection=False,
            )
            # sets up the retriever
            self.retriever = self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.3,
                },
            )

    def ask(self, query: str, kb, prompt):
        """
        Asks a question using the configured processing chain.

        Parameters:
        - query (str): The question to be asked.

        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """

        qa_system_prompt = prompt

        if kb:
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            qa_system_prompt = qa_system_prompt + "\\nUse the following pieces of retrieved context to answer the question.\\n\\n{context} "
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            self.history_aware_retriever = create_history_aware_retriever(
                self.model, self.retriever, contextualize_q_prompt
            )
            print(self.prompt)
            question_answer_chain = create_stuff_documents_chain(self.model, self.prompt)
            rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        else:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

        ### Statefully manage chat history ###

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            print(self.store[session_id])
            return self.store[session_id]

        if not kb:
            rag_chain = (
                    self.prompt
                    | self.model
                    | StrOutputParser())

        self.chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = self.chain.invoke({"input": query},
                                         config={
                                        "configurable": {"session_id": "abc123"}
                                        },  # constructs a key "abc123" in `store`.
                                    )
        print(response)

        if kb:
            return response["answer"]
        else:
            return response

    def clear(self):
        """
        Clears the components in the question-answering system.

        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        # Set the vector store to None.
        self.vector_store = None

        # Set the history to None
        self.history_aware_retriever = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None