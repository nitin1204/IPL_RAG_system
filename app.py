#load data
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df1 = pd.read_csv('DATA/ipl_batsman.csv')
    df2 = pd.read_csv('DATA/ipl_bowler.csv')
    df3 = pd.read_csv('DATA/matches.csv')
    return pd.concat([df1, df2, df3], ignore_index=True)

# merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# merged_df.to_csv('final.csv', index=False)
# df = pd.read_csv("final.csv")

# || data to text ||

import os
def df_to_text(df):
  texts = []
  for i in range(df.shape[0]):
    row = df.iloc[i]
    text = ""
    for col in df.columns:
      text += col+":"+str(row[col])+","
    texts.append(text)

  df["text"] = texts
  folder_path ="Data"
  os.makedirs(folder_path,exist_ok=True)
  file_path = os.path.join(folder_path,"Criket_data.txt")
  with open(file_path,"w",encoding="utf-8") as f:
    for line in df["text"]:
      f.write(line+"\n")

df = load_data()
texts = df_to_text(df)


# || text to document ||
from langchain_community.document_loaders.text import TextLoader
def text_to_document():
  doct_loader = TextLoader("Data/Criket_data.txt")
  doct = doct_loader.load()
  return doct
obj_doct = text_to_document()


# || document to chunk ||
from langchain_text_splitters import RecursiveCharacterTextSplitter
def doct_to_chunk(document,chunk_size=300,chunk_overlap=30):
  text_spliter = RecursiveCharacterTextSplitter(
      chunk_size = chunk_size,
      chunk_overlap = chunk_overlap
  )
  chunk_doct = text_spliter.split_documents(document)
  return chunk_doct
chunk = doct_to_chunk(obj_doct)
print(len(chunk))

# || chunk to embedding ||
from sentence_transformers import SentenceTransformer
@st.cache_resource
class EmbeddingManager:
  def __init__(self,model_name = "all-MiniLM-L6-v2"):
    self.model_name = model_name
    print("Loadding model...",self.model_name)
    self.model = SentenceTransformer(self.model_name)
    print("Embedding Dimension",self.model.get_sentence_embedding_dimension())
  def chunk_embed(self,text):
    embeddings = self.model.encode(text,show_progress_bar=True)
    print("embeddign shape:",embeddings.shape)
    return embeddings
embedding_model = EmbeddingManager()

# || embedding to vector store ||
import chromadb
import uuid
import os
@st.cache_resource
class vectorstoremanager:
  def __init__(self,persist_directory = "DATA/vector_data",collection_name="text_document"):
    self.persist_directory = persist_directory
    self.collection_name = collection_name
    self.collection = None
    self.client = None
    self.initialize_store()

  def initialize_store(self):
    os.makedirs(self.persist_directory,exist_ok=True)
    # create cilent
    self.client = chromadb.PersistentClient(path=self.persist_directory)

    # create collection
    self.collection = self.client.get_or_create_collection(
        name = self.collection_name,
        metadata = {"description":"vector store collection does not match of embedding"}
    )
    print("initialized the vector store with collection",self.collection_name)
    print("docs in collection",self.collection.count())

  def add_doctument(self,document,embeddings):
    if len(document) != len(embeddings):
      raise ValueError("num of document  does not match num of embedding ")

    ids = []
    embedding_list = []
    documents_content = []
    all_metadata = []
    for i,(doc,embedding) in enumerate(zip(document,embeddings)):
      doc_id = f"doc_{uuid.uuid4()}"
      ids.append(doc_id)

      metadata = dict(doc.metadata)
      metadata['doc_index'] = i
      metadata['content_length'] = len(doc.page_content)
      all_metadata.append(metadata)

      documents_content.append(doc.page_content)
      embedding_list.append(embedding.tolist())

    self.collection.add(
        ids = ids,
        documents = documents_content,
        metadatas = all_metadata,
        embeddings = embedding_list
    )
    print("total document add in vector store",len(documents_content))
    print("docs in collection",self.collection.count())
vector_store = vectorstoremanager()

text = [doc.page_content for doc in chunk]
embedding = embedding_model.chunk_embed(text)
vector_store.add_doctument(chunk,embedding)

# || retrival ||
from sklearn.metrics.pairwise import cosine_similarity

class RAGretrival:
    def __init__(self,embedding_model,vector_store):
        self.embedding = embedding_model
        self.vectore_data = vector_store
    def retrival(self,query,top_k=5,score_threshhold=0.0):
        query_embedding = self.embedding.chunk_embed([query])[0]
        result = self.vectore_data.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results = top_k
        )
        # cosine similarity
        retrival_doct= []
        if result["documents"] and result["documents"][0]:
            id = result["ids"][0]
            metadata = result["metadatas"][0]
            document = result["documents"][0]
            distances = result["distances"][0]

            for i,(doc_id, meta, doc, dist) in enumerate(zip(id,metadata,document,distances)):
                similarity_score = 1-dist

                if similarity_score >= score_threshhold:
                    retrival_doct.append({
                        "id":doc_id,
                        "metadata":meta,
                        "document":doc,
                        "distance":dist,
                        "similarity_score":similarity_score,
                        "rank":i+1
                    })   
            print(f"retrieved {len(retrival_doct)} documents")          
        else:
            print("no document found")
        return retrival_doct   
    
rag_retrival = RAGretrival(embedding_model,vector_store)
respons = rag_retrival.retrival("total kkr run")
print(respons)



# || llm || 

from langchain_groq import ChatGroq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="qwen/qwen3-32b",
        temperature=0.1
    )
llm = load_llm()   

def generate_output(query,retrival,llm,top_k=3):
    result =retrival.retrival(query,top_k)
    context = "\n".join(doc["document"] for doc in result) if result else ""
    if not context:
        print("we found no reterival content for the given qurey")
    
    prompt = f""" use given content to generate the answer for the query
                content:{context}
                query:{query}"""
    
    respons = llm.invoke(prompt)
    return respons.content


answer = generate_output("hight run in IPL",rag_retrival,llm)
print(answer)


# =========================
# STREAMLIT UI
# =========================
st.title("🏏 IPL Chatbot")

df = load_data()
texts = df_to_text(df)

embedding_model = EmbeddingManager()
collection = vector_store.collection
llm = load_llm()

query = st.text_input("Ask your question")

if query:
    docs = rag_retrival.retrival(query) 
    result = generate_output(query, rag_retrival, llm)

    st.subheader("📄 Retrieved Data")
    st.write(docs)

    st.subheader("🤖 Answer")
    st.write(result)