# Document loading and QA retrieval

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

loader = TextLoader('./the-story-of-the-exodus.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(texts, embeddings, collection_name="story-of-the-exodus")

llm = OpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

print(chain.invoke({'query': "Why did the Israelites leave Egypt?"}))
