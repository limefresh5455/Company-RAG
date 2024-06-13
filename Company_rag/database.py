import os
import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

folder_path = "Company_data"
md_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]

docs = []
for file in md_files:
    file_path = os.path.join(folder_path, file)
    json_file_path = os.path.join(folder_path, f".metadata.{file[:-2]}json")

    with open(file_path) as temp:
        text = temp.read()

    with open(json_file_path,'r') as json_object:
        json_text = json.load(json_object)

    new_doc = Document(
        page_content = text,
        metadata = json_text
    )

    docs.append(new_doc)

db = FAISS.from_documents(docs, embedding)
db.save_local("companies_vector_database",index_name="companies")