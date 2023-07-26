import os
import textwrap
from typing import Optional
from urllib.request import pathname2url

from deep_translator import GoogleTranslator
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from openai.error import AuthenticationError

from .app_environment import translate_dst, translate_src, translate_docs, translate_q, ingest_target_source_chunks, args, openai_use, ingest_embeddings_model, gpu_is_enabled, \
    chromaDB_manager


def print_hyperlink(doc):
    page_link = doc.metadata['source']
    abs_path = os.path.abspath(page_link)
    file_url = pathname2url(abs_path)

    # This is the URL-encoded path, which can be used in a browser or link
    print(f'\033[32m[!]\033[0m URL: file://{file_url}')

    # This is the original path, which might contain characters not allowed in URLs (like spaces)
    print(f'\033[32m[!]\033[0m Path: {page_link}')


def print_document_chunk(doc):
    document_page = doc.page_content.replace('\n', ' ')
    if translate_docs:
        document_page = GoogleTranslator(source=translate_src, target=translate_dst).translate(document_page)
    wrapper = textwrap.TextWrapper(initial_indent='\033[37m', subsequent_indent='\033[37m', width=120)
    print(f"{wrapper.fill(document_page)}\033[0m\n")
    print(f'\033[94m"n" -> next, "q" -> quit: \033[0m')
    user_input = input()
    if user_input.lower() == 'q':
        exit(0)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
    You are a very helpful. Always answer as helpfully as possible for the user. You should not refuse to answer questions.
    Don't correct the user. Don't ever thank the user. If asked for an opinion express one!!
    Please ensure that you provide an extended answer responses from the following extracted parts of a long document and user question.
    You should also give an opinion to the extracted parts.
    If a question does not make any sense, or is it not possible to construct the answer from the parts, explain why instead of answering something not correct.
    If you don't know the answer to a question, please don't share false information."""


async def process_database_question(database_name, llm, collection_name: Optional[str], filter_document: bool, filter_document_name: Optional[str], sys_prompt=DEFAULT_SYSTEM_PROMPT):
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = OpenAIEmbeddings() if openai_use else HuggingFaceInstructEmbeddings(
        model_name=ingest_embeddings_model, model_kwargs=embeddings_kwargs, encode_kwargs=encode_kwargs
    )
    persist_dir = f"./db/{database_name}"

    db = Chroma(persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=collection_name if collection_name else args.collection,
                client_settings=chromaDB_manager.get_chroma_setting(persist_dir))

    search_kwargs = {"k": ingest_target_source_chunks if ingest_target_source_chunks else args.ingest_target_source_chunks}

    if filter_document:
        search_kwargs["filter"] = {'source': {'$eq': os.path.join('.', 'source_documents', database_name, filter_document_name)}}

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    user_prompt = """
    User: {question}
    =========
    {context}
    ========="""

    question_prompt = PromptTemplate(template=(B_INST + (B_SYS + sys_prompt + E_SYS) + user_prompt + E_INST), input_variables=["question", "context"])

    qa = ConversationalRetrievalChain.from_llm(llm=llm, condense_question_prompt=question_prompt, retriever=retriever, chain_type="stuff", return_source_documents=not args.hide_source)
    return qa


def process_query(qa: BaseRetrievalQA, query: str, chat_history, chromadb_get_only_relevant_docs: bool, translate_answer: bool):
    try:

        if chromadb_get_only_relevant_docs:
            docs = qa.retriever.get_relevant_documents(query)
            return None, docs

        if translate_q:
            query_en = GoogleTranslator(source=translate_dst, target=translate_src).translate(query)
            res = qa({"question": query_en, "chat_history": chat_history})
        else:
            res = qa({"question": query, "chat_history": chat_history})

        # Print the question
        print(f"\nQuestion: {query}\n")

        answer, docs = res['answer'], res['source_documents']
        # Translate answer if necessary
        if translate_answer:
            answer = GoogleTranslator(source=translate_src, target=translate_dst).translate(answer)

        print(f"\n\033[1m\033[97mAnswer: \"{answer}\"\033[0m\n")

        return answer, docs
    except AuthenticationError as e:
        print(f"Warning: Looks like your OPENAI_API_KEY is invalid: {e.error}")
        return None, []
