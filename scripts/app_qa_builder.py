import os
import textwrap
from typing import Optional
from urllib.request import pathname2url

from deep_translator import GoogleTranslator
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai.error import AuthenticationError

from scrapalot_browse import speak_chunk
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
    print(f'\033[94m"n" -> next, "s" -> speak, "q" -> quit: \033[0m')
    user_input = input()
    if user_input.lower() == 'q':
        exit(0)
    elif user_input.lower() == 's':
        speak_chunk(document_page)


async def process_database_question(database_name, llm, collection_name: Optional[str]):
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = OpenAIEmbeddings() if openai_use else HuggingFaceEmbeddings(
        model_name=ingest_embeddings_model, model_kwargs=embeddings_kwargs, encode_kwargs=encode_kwargs
    )
    persist_dir = f"./db/{database_name}"

    db = Chroma(persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=collection_name if collection_name else args.collection,
                client_settings=chromaDB_manager.get_chroma_setting(persist_dir)
                )

    retriever = db.as_retriever(search_kwargs={"k": ingest_target_source_chunks if ingest_target_source_chunks else args.ingest_target_source_chunks})

    template = """You are a an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question.
    Provide a conversational answer (about {answer_length} words) based on the context provided.
    If you can't find the answer in the context below, just say
    "Hmm, I'm not sure." Don't try to make up an answer. If the question is not related to the context, politely respond
    that you are tuned to only answer questions that are related to the context.

    Question: {question}
    =========
    {context}
    =========
    Answer:"""
    question_prompt = PromptTemplate(template=template, input_variables=["question", "answer_length", "context"])

    qa = ConversationalRetrievalChain.from_llm(llm=llm, condense_question_prompt=question_prompt, retriever=retriever, chain_type="stuff", return_source_documents=not args.hide_source)
    return qa


def process_query(qa: BaseRetrievalQA, query: str, answer_length: int, chat_history, chromadb_get_only_relevant_docs: bool, translate_answer: bool):
    try:

        if chromadb_get_only_relevant_docs:
            docs = qa.retriever.get_relevant_documents(query)
            return None, docs

        if translate_q:
            query_en = GoogleTranslator(source=translate_dst, target=translate_src).translate(query)
            res = qa({"question": query_en, "answer_length": answer_length, "chat_history": chat_history})
        else:
            res = qa({"question": query, "answer_length": answer_length, "chat_history": chat_history})

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
