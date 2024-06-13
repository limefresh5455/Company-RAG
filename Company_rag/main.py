from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
import gradio as gr

embedding = OpenAIEmbeddings()


vectorstore = FAISS.load_local("companies_vector_database", embedding, index_name="companies", allow_dangerous_deserialization=True)

# code from langchain
retriever = vectorstore.as_retriever()

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are working as a official assistant who answers user's qustions about companies like a diplomatic person based on relavant data in the database
here is the relavant data to answer user's question:
{context}
Now Answer the question based only on the relavant data but do not mention it in front of user in your response, only answer as much as user asks
Question: {question}
Answer:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo',max_tokens=300)

chat_history = []


def bot(question,history):
    ans = conversational_qa_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )

    yield ans.content
    print(chat_history)
    chat_history.append(HumanMessage(content= question))
    chat_history.append(AIMessage(content= ans.content))

gr.ChatInterface(bot).launch()