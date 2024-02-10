import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever


'''
The general flow is:
- we load data using TextLoaders, PDFLoaders etc using LangChain
- those loaders return Documents in the form: Document(page_content='...
- we can apply chunking using for example RecursiveCharacterTextSplitter
- with split_documents we turn the Documents turns them into an array of strings
- in order to use those splits as Documents it is necessary a convertion (get_documents_embeddings)
'''

load_dotenv()

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n" + d.page_content for i, d in enumerate(docs)]))

# returns 2 Documents (merged) that origin from the 2 textloaders
# please note that get_documents and merge_loaders have the same result
def get_documents():
    loader = TextLoader("./dogs.txt")
    data = loader.load()
    loader = TextLoader("./restaurant.txt")
    data2 = loader.load()
    docs = data + data2
    return docs

# returns 2 Documents (merged) that origin from the 2 textloaders
# please note that get_documents and merge_loaders have the same result
def merge_loaders():
    dogs_loader = TextLoader("./dogs.txt")
    restaurants_loader = TextLoader("./restaurant.txt")
    merge_loaders = MergedDataLoader(loaders=[dogs_loader, restaurants_loader])
    merge_docs = merge_loaders.load()
    return merge_docs

    # SAME AS THIS
    # loader = TextLoader("./dogs.txt")
    # data = loader.load()
    # loader = TextLoader("./restaurant.txt")
    # data2 = loader.load()
    # docs = data + data2

# returns the chunks using RecursiveCharacterTextSplitter
# chunks are "Documents" and they are in the form: page_content='...
def get_chunks(docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=10)
    chunks=text_splitter.split_documents(docs)
    return chunks

# returns the data_vectors from the docs (Documents) provided
# a RecursiveCharacterTextSplitter is applied and the split is applied to the whole set of Documents
def get_data_vectors(docs, embedding):
    chunks=get_chunks(docs)
    return get_documents_embeddings(chunks, embedding)

# returns the Documents from the split_documents applied
def get_documents_embeddings(docs, embedding):
    return [embedding.embed_query(doc.page_content) for doc in docs]

def get_similarity(embedding, data_vectors):
    vector1 = embedding.embed_query("How is the whether??")
    vector2 = embedding.embed_query("What is the Name of the Dogschool?")
    vector3 = embedding.embed_query("What food do you offer?")

    cosine_sims_1 = [cosine_similarity([vector1], [data_vector])[0][0] for data_vector in data_vectors]
    cosine_sims_2 = [cosine_similarity([vector2], [data_vector])[0][0] for data_vector in data_vectors]
    cosine_sims_3 = [cosine_similarity([vector3], [data_vector])[0][0] for data_vector in data_vectors]

    # print(cosine_sims_1)
    # print(cosine_sims_2)
    # print(cosine_sims_3)
    # [0.698265465053605, 0.7123417407597779]
    # [0.8724829472326514, 0.7433352871312556]
    # [0.6958046369817004, 0.7693273179771332]
    # cosin_sims_i means that i want to look for the similarity of query i with data_vector1 and data_vector2
    # so for example cosin_sims_1 is [0.698265465053605, 0.7123417407597779]
    # it means that the query 1 has a similarity of 0.69 with data_vector1 and 0.71 with data_vector2
    # Remember that: 
    # query 1 -> no relation with documents
    # query 2 -> related to docs
    # query 3 -> related to restaurants

    # plot generation
    x = np.arange(len(data_vectors))

    plt.scatter(x, cosine_sims_1, label='Weather', alpha=0.7)
    plt.scatter(x, cosine_sims_2, label='Dogschool', alpha=0.7)
    plt.scatter(x, cosine_sims_3, label='Restaurant', alpha=0.7)

    plt.ylabel('Cosine Similarity')
    plt.title('Consine Similarity between query and data vectors')
    plt.legend()

    plt.show()

def get_similarity_smaller_chunks(docs,embedding):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=10)
    docs=text_splitter.split_documents(docs)
    data_vectors = [embedding.embed_query(doc.page_content) for doc in docs]
    #print(len(data_vectors))
    #35

    get_similarity(embedding=embedding,data_vectors=data_vectors)

'''
Small chunks can be more selective and provide better results 
but the context provided to the LLM can be too small
that’s what we can work with Parent Document Retriever:
- we send the query to a small selective chunk
- we then can send the larger document with more context to the LLM
'''
def get_parent_relevant_documents(docs, embedding):
    # creation of the 2 splitters (one for small chunks and one for large chunks)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

    # creation of the chroma vector store
    vectorstore = Chroma(
        collection_name="full_documents", 
        embedding_function=embedding
    )
    store = InMemoryStore()

    # creation of the retriever with chroma and the 2 splitters
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(docs, ids=None)
    query="What is the name of the dog school?"
    result=vectorstore.similarity_search(query)
    print(result)
    #[Document(page_content='A1: The school is called "Canine Academy".', metadata={'doc_id': '24868a95-1570-4be1-9943-4a393f236dc1', 'source': './dogs.txt'}), Document(page_content='Fiktive Hundeschule: Canine Academy\nQ1: What is the name of the dog training school?', metadata={'doc_id': '3255cc5e-3581-43b7-8566-3f696f1ca489', 'source': './dogs.txt'}), Document(page_content='Q3: What training programs are offered at Canine Academy?', metadata={'doc_id': 'b6014e80-c852-4cd6-80b3-4ab871b85d9c', 'source': './dogs.txt'}), Document(page_content='Q7: Does Canine Academy provide training for service or therapy dogs?', metadata={'doc_id': 'b5c70370-4256-4bfa-8491-3b7e3ca9aa06', 'source': './dogs.txt'})]
    result=retriever.get_relevant_documents(query)
    print(result)
    #[Document(page_content='A1: The school is called "Canine Academy".', metadata={'source': './dogs.txt'}), Document(page_content='Fiktive Hundeschule: Canine Academy\nQ1: What is the name of the dog training school?', metadata={'source': './dogs.txt'}), Document(page_content='Q3: What training programs are offered at Canine Academy?', metadata={'source': './dogs.txt'}), Document(page_content='Q7: Does Canine Academy provide training for service or therapy dogs?', metadata={'source': './dogs.txt'})]

'''
Nuances in the question can lead to different results 
if the question does not capture the embeddings semantically well
MultiQueryRetriever creates variations of the question and performs those queries
the response are the top 4 results from those query variations
'''
def get_multi_query_retriever(docs,embedding):

    llm = ChatOpenAI(
        temperature=0,
        max_tokens=800,
        model_kwargs={"top_p": 0, "frequency_penalty": 0, "presence_penalty": 0},
    )
    
    # creation of the chroma vector store
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embedding
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), 
        llm=llm
    )

    query="What is the name of the dog school?"
    unique_docs = retriever.get_relevant_documents(query)
    print(unique_docs)
    #print(len(unique_docs))
    # 4
    # we are retrieving 4 unique documents which means that:
    # - variations of the query are generated
    # - each query is performed and documents are obtained
    # - duplicated documents are removed
    # - we finally get the unuque documents as response


embedding = OpenAIEmbeddings()#chunk_size=1)

# load files
docs = get_documents()
# print(docs)
# 2
# note that docs is the concatenation of 2 Documents (loader.load() generates a documents)

# documents embedding
#data_vectors = get_documents_embeddings(docs,embedding)
#print(len(data_vectors))
# 2
# there is one data vector embedding for each document

# similarity for large chunks
#get_similarity(embedding,data_vectors)

# re-chunk original docs into smaller chunks
chunks=get_chunks(docs)
#print(len(chunks))
# 35

# similarity for small chunks
#get_similarity_smaller_chunks(docs, embedding)

# get relevant documents using parent retriever
#get_parent_relevant_documents(docs, embedding)

# get relevant documents using multi query retriever
get_multi_query_retriever(chunks, embedding)

# docs=merge_loaders()
# data_vectors=get_data_vectors(docs,embedding)
# print(len(data_vectors))