import streamlit as st
import dotenv, os
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

dotenv.load_dotenv()
repo_path = os.getcwd()+"/repo"
persist_directory = 'db'

if len(os.listdir(repo_path)) == 0:
    repo = Repo.clone_from("https://github.com/hwchase17/langchain", to_path=repo_path)

loader = GenericLoader.from_filesystem(
    repo_path+"/libs/langchain/langchain",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)
documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=2000, 
    chunk_overlap=200)
texts = python_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(disallowed_special=())

if len(os.listdir(os.getcwd()+'/db')) == 0:
    db = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = db.as_retriever(
    search_type="mmr", # Also test "similarity"
    search_kwargs={"k": 8},
)

# Set LLM
llm = ChatOpenAI(model_name="gpt-4") 

# Set Memory
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

def search(inputan):
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    result = qa(inputan)
    return result['answer']

def main():
    st.set_page_config(page_title='🦜🔗 Ask the Repo App')
    st.title('🦜🔗 Ask the Repo App')

    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')
    if query_text:
        result = search(query_text)
        if result:
            st.markdown(result)

if __name__ == '__main__':
    main()