#!/usr/bin/env python3
import concurrent
import os
import sys
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from time import monotonic
from typing import List, Optional, Dict

from dotenv import set_key
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import Chroma
from tqdm import tqdm

from ingestion_storage.app_storage import AbstractStorage
from ingestion_storage.app_storage_google_drive import GoogleDriveStorage
from ingestion_storage.app_storage_local import LocalStorage
from scripts.app_environment import (
    ingest_chunk_size,
    ingest_chunk_overlap,
    ingest_embeddings_model,
    ingest_persist_directory,
    ingest_source_directory,
    args,
    chromaDB_manager,
    gpu_is_enabled, ingest_storage_type, openai_use)
from scripts.app_utils import display_directories, load_single_document


def load_documents(storage: AbstractStorage, source_dir: str, collection_name: Optional[str], ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files.
    :param storage: The storage object.
    :param source_dir: The path of the source documents directory.
    :param collection_name: The name of the collection to exclude files from.
    :param ignored_files: A list of filenames to be ignored.
    :return: A list of Document objects loaded from the source documents.
    """
    collection_dir = os.path.join(source_dir, collection_name) if collection_name else source_dir
    print(f"Loading documents from {collection_dir}")
    all_files = storage.list_files_src(collection_dir)
    filtered_files = [file_path for file_path in all_files if storage.is_file(file_path) and file_path not in ignored_files]

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        load_with_storage = partial(load_single_document, storage)
        results = []
        future_to_file = {executor.submit(load_with_storage, file_path): file_path for file_path in filtered_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(filtered_files), desc='Loading new documents', ncols=80):
            file_path = future_to_file[future]
            try:
                docs = future.result()
            except Exception as exc:
                print(f"\n - {file_path}: \n error: {repr(exc)}")
                continue

            print(f"\n\033[32m\033[2m\033[38;2;0;128;0m{docs[0].metadata.get('source', '')} \033[0m")
            results.extend(docs)

    return results


def get_language(file_extension: str) -> Language:
    ext_to_lang = {
        ".java": Language.JAVA,
        ".js": Language.JS,
        ".py": Language.PYTHON
    }

    return ext_to_lang.get(file_extension)


def split_documents(documents: list[Document]) -> Dict[Optional[Language], list[Document]]:
    lang_to_docs = defaultdict(list)
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        lang = get_language(file_extension)
        lang_to_docs[lang].append(doc)  # Default to None if the file extension does not match any language

    return lang_to_docs


def process_documents(storage: AbstractStorage, collection_name: Optional[str] = None, ignored_files: List[str] = []) -> List[Document]:
    """
    Processes all documents in the source directory and returns them as a list of strings.
    :param storage: The storage object.
    :param collection_name: The name of the collection to exclude files from.
    :param ignored_files: A list of filenames to be ignored.
    :return: A list of strings representing the processed documents.
    """
    documents = load_documents(storage, source_directory, collection_name if db_name != collection_name else None, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")

    texts = []
    for lang, docs in split_documents(documents).items():
        if lang is None:
            split_docs = RecursiveCharacterTextSplitter(
                chunk_size=ingest_chunk_size if ingest_chunk_size else args.ingest_chunk_size,
                chunk_overlap=ingest_chunk_overlap if ingest_chunk_overlap else args.ingest_chunk_overlap
            ).split_documents(docs)
        else:
            print(f"Ingesting {lang} file:")
            split_docs = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=ingest_chunk_size if ingest_chunk_size else args.ingest_chunk_size,
                chunk_overlap=ingest_chunk_overlap if ingest_chunk_overlap else args.ingest_chunk_overlap
            ).split_documents(docs)

        texts.extend(split_docs)

    print(f"Split into {len(texts)} chunks of text (max. {ingest_chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(storage: AbstractStorage, persist_directory: str) -> bool:
    """
    Checks if a Chroma vectorstore already exists in the given directory.
    :param storage: The storage object.
    :param persist_directory: The path of the vectorstore directory.
    :return: True if the vectorstore exists, False otherwise.
    """
    if storage.path_exists(os.path.join(persist_directory, 'index')):
        if storage.path_exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and storage.path_exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = storage.list_files_db(persist_directory)
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def prompt_storage_type():
    print(f"\033[94mSelect a storage type or 'q' to quit:\n\033[0m")
    print("1. Local Drive")
    print("2. Google Drive")
    print("3. Google Cloud")
    print("4. S3 Bucket")
    # Add additional options for other storage types
    print(f"5. Use current storage type: {ingest_storage_type}")

    user_choice = input('\nEnter your choice ("q" for quit): ').strip()
    choose_storage_type = ingest_storage_type

    if user_choice == "1":
        choose_storage_type = 'local_storage'
        set_key('.env', 'INGEST_STORAGE_TYPE', choose_storage_type)
    elif user_choice == "2":
        choose_storage_type = 'google_drive_storage'
        set_key('.env', 'INGEST_STORAGE_TYPE', choose_storage_type)
        # Prompt the user to enter any necessary credentials for Google Drive
    # Add additional cases for other storage types
    elif user_choice == "3":
        pass
    elif user_choice == "4":
        pass
    elif user_choice == "5":
        pass
    elif user_choice == "q":
        exit(0)
    else:
        print("\n\033[91m\033[1m[!] \033[0mInvalid choice. Please try again.\033[91m\033[1m[!] \033[0m\n")

    set_key('.env', 'INGEST_STORAGE_TYPE', choose_storage_type)

    return choose_storage_type


def prompt_user():
    """
    Prompts the user to select a storage type and enter any necessary credentials.
    Then prompts the user to select an existing directory or create a new one to store source material.
    Sets the storage type and directory paths as environment variables and returns them.
    :return: The selected storage type, source directory path, and the selected database directory path.
    """

    def _create_directory(directory_name):
        """
        Creates a new directory with the given directory_name in the ./source_documents directory.
        It also creates a corresponding directory in the ./db directory for the database files.
        It sets the directory paths as environment variables and returns them.
        :param directory_name: The name for the new directory.
        :return: The path of the new directory and the path of the database directory.
        """
        directory_path = os.path.join(".", "source_documents", directory_name)
        db_path = os.path.join(".", "db", directory_name)

        os.makedirs(directory_path)
        os.makedirs(db_path)
        set_key('.env', 'INGEST_SOURCE_DIRECTORY', directory_path)
        set_key('.env', 'INGEST_PERSIST_DIRECTORY', db_path)
        print(f"Created new directory: {directory_path}")
        return directory_path, db_path

    while True:
        print(f"\033[94mSelect an option or 'q' to quit:\n\033[0m")
        print("1. Select an existing directory")
        print("2. Create a new directory")
        print(f"3. Use current ingest_source_directory: {ingest_source_directory}")

        user_choice = input('\nEnter your choice ("q" for quit): ').strip()

        if user_choice == "1":
            directories = display_directories()
            while True:  # Keep asking until we get a valid directory number
                existing_directory = input("\n\033[94mEnter the number of the existing directory (q for quit, b for back): \033[0m")
                if existing_directory == 'q':
                    raise SystemExit
                elif existing_directory == 'b':
                    break
                try:
                    selected_directory = directories[int(existing_directory) - 1]
                    selected_directory_path = os.path.join(".", "source_documents", selected_directory)
                    selected_db_path = os.path.join(".", "db", selected_directory)

                    if not os.listdir(selected_directory_path):
                        print(f"\033[91m\033[1m[!]\033[0m Selected directory: '{selected_directory}' is empty \033[91m\033[1m[!]\033[0m")
                        directories = display_directories()  # Display directories again if the selected one is empty
                    else:
                        if not os.path.exists(selected_db_path):
                            os.makedirs(selected_db_path)
                        set_key('.env', 'INGEST_SOURCE_DIRECTORY', selected_directory_path)
                        set_key('.env', 'INGEST_PERSIST_DIRECTORY', selected_db_path)
                        print(f"Selected directory: {selected_directory_path}")
                        return selected_directory_path, selected_db_path
                except (ValueError, IndexError):
                    print("\n\033[91m\033[1m[!] \033[0mInvalid choice. Please try again.\033[91m\033[1m[!] \033[0m\n")
                    directories = display_directories()  # Display directories again if the input is invalid
        elif user_choice == "2":
            new_directory_name = input("Enter the name for the new directory: ")
            selected_directory_path, selected_db_path = _create_directory(new_directory_name)
            input("Place your source material into the new folder and press enter to continue...")
            return selected_directory_path, selected_db_path
        elif user_choice == "3":
            return ingest_source_directory, ingest_persist_directory
        elif user_choice == "q":
            exit(0)
        else:
            print("\n\033[91m\033[1m[!] \033[0mInvalid choice. Please try again.\033[91m\033[1m[!] \033[0m\n")


def create_embeddings():
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {'device': 'cpu'}
    embeddings = OpenAIEmbeddings() if openai_use else HuggingFaceInstructEmbeddings(
        model_name=ingest_embeddings_model if ingest_embeddings_model else args.ingest_embeddings_model,
        model_kwargs=embeddings_kwargs
    )
    return embeddings


def get_chroma(collection_name: str, embeddings, persist_dir):
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=chromaDB_manager.get_chroma_setting(persist_dir)
    )


def process_and_add_documents(storage_instance, collection, chroma_db, collection_name):
    ignored_files = [metadata['source'] for metadata in collection['metadatas']]
    texts = process_documents(storage=storage_instance, collection_name=collection_name, ignored_files=ignored_files)
    num_elements = len(texts)
    index_metadata = {"elements": num_elements}
    print(f"Creating embeddings. May take some minutes...")
    chroma_db.add_documents(texts, index_metadata=index_metadata)


def process_and_persist_db(storage_instance, database, collection_name):
    print(f"Collection: {collection_name}")
    process_and_add_documents(
        storage_instance=storage_instance,
        collection=database.get(), chroma_db=database, collection_name=collection_name)
    database.persist()


def create_and_persist_db(embeddings, texts, persist_dir, collection_name):
    num_elements = len(texts)
    index_metadata = {"elements": num_elements}
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
        client_settings=chromaDB_manager.get_chroma_setting(persist_dir),
        index_metadata=index_metadata
    )
    db.persist()


def get_ingestion_storage(chosen_storage_type: Optional[str] = ingest_storage_type) -> AbstractStorage:
    if chosen_storage_type == 'local_storage':
        return LocalStorage()
    elif chosen_storage_type == 'google_drive_storage':
        return GoogleDriveStorage()
    # Add additional cases for other storage types
    else:
        raise ValueError(f'Unsupported storage type: {chosen_storage_type}')


def main(storage_instance: AbstractStorage, source_dir: str, persist_dir: str, db_name: str, sub_collection_name: Optional[str] = None):
    embeddings = create_embeddings()
    collection_name = sub_collection_name or db_name

    start_time = monotonic()

    if does_vectorstore_exist(storage_instance, persist_dir):
        print(f"Appending to existing vectorstore at {persist_dir}")
        db = get_chroma(collection_name, embeddings, persist_dir)
        process_and_persist_db(storage_instance=storage_instance, database=db, collection_name=collection_name)
    else:
        print(f"Creating new vectorstore from {source_dir}")
        texts = process_documents(storage=storage_instance, collection_name=collection_name, ignored_files=[])
        create_and_persist_db(embeddings, texts, persist_dir, collection_name)

    print("Ingestion complete! You can now run scrapalot_main.py to query your documents")
    print(f"\033[94mTook {round(((monotonic() - start_time) / 60), 2)} min to process the ingestion!\033[0m")


if __name__ == "__main__":
    try:

        if args.ingest_dbname:
            db_name = args.ingest_dbname
            source_directory = os.path.join(".", "source_documents", db_name)
            persist_directory = os.path.join(".", "db", db_name)

            storage = get_ingestion_storage()
            storage.create_directory(source_directory)
            storage.create_directory(persist_directory)

            if args.collection:
                sub_collection_name = args.collection
                main(storage, source_directory, persist_directory, db_name, sub_collection_name)
            else:
                main(storage, source_directory, persist_directory, db_name)
        else:
            storage_type = prompt_storage_type()
            storage = get_ingestion_storage(storage_type)
            source_directory, persist_directory = prompt_user()
            db_name = os.path.basename(persist_directory)
            main(storage, source_directory, persist_directory, db_name)
    except SystemExit:
        print("\n\033[91m\033[1m[!] \033[0mExiting program! \033[91m\033[1m[!] \033[0m")
        sys.exit(1)
