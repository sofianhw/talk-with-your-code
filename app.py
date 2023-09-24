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
dir_path = os.getcwd()+"/"
repo_path = dir_path+"repo"
persist_directory = 'db'
db_path = dir_path+persist_directory

if os.path.exists(repo_path) == False:
    os.makedirs(repo_path)

if os.path.exists(db_path) == False:
    os.makedirs(db_path)

if len(os.listdir(repo_path)) == 0:
    repo = Repo.clone_from("https://github.com/sofianhw/data-platform", to_path=repo_path)

loader = GenericLoader.from_filesystem(
    repo_path+"/",
    glob="**/*",
    suffixes=[".yml"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)
documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=2000, 
    chunk_overlap=200)
texts = python_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(disallowed_special=())

if len(os.listdir(db_path)) == 0:
    db = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = db.as_retriever(
    search_type="mmr", # Also test "similarity"
    search_kwargs={"k": 8},
)

# Set LLM and Memory
llm = ChatOpenAI(model_name="gpt-4") 
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

def search(inputan):
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    result = qa(inputan)
    return result['answer']

while True:
    print('\n')
    inputan = input('Question : ')
    answer = search(inputan)
    print('Result : %s' % answer)