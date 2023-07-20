import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Tuple
from urllib.parse import unquote

from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from langchain.callbacks import StreamingStdOutCallbackHandler
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

from scrapalot_main import get_llm_instance
from scripts.app_environment import translate_src, translate_q, chromaDB_manager, translate_a, model_n_answer_words, api_host, api_port, api_scheme
from scripts.app_qa_builder import process_database_question, process_query

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

app = FastAPI(title="scrapalot-chat API")

origins = [
    "http://localhost:3000", "http://localhost:8000", "https://scrapalot.com"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="scrapalot-research-assistant-ui/static"), name="static")

load_dotenv()


###############################################################################
# model classes
###############################################################################
class QueryBody(BaseModel):
    database_name: str
    collection_name: str
    question: str
    translate_chunks: bool = True
    locale: str


class TranslationBody(BaseModel):
    locale: str


class SourceDirectoryDatabase(BaseModel):
    name: str
    path: str


class SourceDirectoryFile(BaseModel):
    id: str
    name: str


class SourceDirectoryFilePaginated(BaseModel):
    total_count: int
    items: List[SourceDirectoryFile]


class TranslationItem(BaseModel):
    src_lang: str
    dst_lang: str
    text: str


class LLM:
    def __init__(self):
        self.instance = None

    def get_instance(self):
        if not self.instance:
            self.instance = get_llm_instance(StreamingStdOutCallbackHandler())
        return self.instance


###############################################################################
# init
###############################################################################
chat_history = []
llm_manager = LLM()
executor = ThreadPoolExecutor(max_workers=5)


@app.on_event("startup")
async def startup_event():
    llm_manager.get_instance()


###############################################################################
# helper functions
###############################################################################

def get_llm():
    return llm_manager.get_instance()


def list_of_collections(database_name: str):
    client = chromaDB_manager.get_client(database_name)
    return client.list_collections()


async def get_files_from_dir(database: str, page: int, items_per_page: int) -> Tuple[List[SourceDirectoryFile], int]:
    all_files = []

    for root, dirs, files in os.walk(database):
        for file in sorted(files):  # Added sorting here.
            if not file.startswith('.'):
                all_files.append(SourceDirectoryFile(id=str(uuid.uuid4()), name=file))
    start = (page - 1) * items_per_page
    end = start + items_per_page
    return all_files[start:end], len(all_files)


def run_ingest(database_name: str, collection_name: Optional[str] = None):
    if database_name and not collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name], check=True)
    if database_name and collection_name:
        subprocess.run(["python", "scrapalot_ingest.py",
                        "--ingest-dbname", database_name, "--collection", collection_name], check=True)


async def get_database_file_response(absolute_file_path: str) -> Union[FileResponse]:
    return FileResponse(absolute_file_path)


###############################################################################
# API
###############################################################################
@app.get("/api")
async def root():
    return {"ping": "pong!"}


@app.get('/api/databases')
async def get_database_names_and_collections():
    base_dir = "./db"
    try:
        database_names = \
            sorted([name for name in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, name))])

        database_info = []
        for database_name in database_names:
            collections = list_of_collections(database_name)
            database_info.append({
                'database_name': database_name,
                'collections': collections
            })

        return database_info
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/{database_name}", response_model=SourceDirectoryFilePaginated)
async def get_database_files(database_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = os.path.join(".", "source_documents")
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    files, total_count = await get_files_from_dir(database_dir, page, items_per_page)
    return {"total_count": total_count, "items": files}


@app.get("/api/database/{database_name}/collection/{collection_name}", response_model=List[SourceDirectoryFile])
async def get_database_collection_files(database_name: str, collection_name: str, page: int = Query(1, ge=1), items_per_page: int = Query(10, ge=1)):
    base_dir = os.path.join(".", "source_documents")
    absolute_base_dir = os.path.abspath(base_dir)
    collection_dir = os.path.join(absolute_base_dir, database_name, collection_name)
    if not os.path.exists(collection_dir) or not os.path.isdir(collection_dir):
        raise HTTPException(status_code=404, detail="Collection not found")
    files = await get_files_from_dir(collection_dir, page, items_per_page)
    return files


@app.get("/api/database/{database_name}/file/{file_name}", response_model=None)
async def get_database_file(database_name: str, file_name: str) -> Union[HTMLResponse, FileResponse]:
    base_dir = os.path.join(".", "source_documents")
    absolute_base_dir = os.path.abspath(base_dir)
    database_dir = os.path.join(absolute_base_dir, database_name)
    if not os.path.exists(database_dir) or not os.path.isdir(database_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    absolute_file_path = os.path.join(database_dir, unquote(file_name))
    if not os.path.exists(absolute_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return await get_database_file_response(absolute_file_path)


@app.post('/api/query')
async def query_files(body: QueryBody, llm=Depends(get_llm)):
    database_name = body.database_name
    collection_name = body.collection_name
    question = body.question
    locale = body.locale
    translate_chunks = body.translate_chunks

    try:
        if translate_q:
            question = GoogleTranslator(source=locale, target=translate_src).translate(question)

        seeking_from = database_name + '/' + collection_name if collection_name and collection_name != database_name else database_name
        print(f"\n\033[94mSeeking for answer from: [{seeking_from}]. May take some minutes...\033[0m")
        qa = await process_database_question(database_name, llm, collection_name)
        answer, docs = process_query(qa, question, model_n_answer_words, chat_history, chromadb_get_only_relevant_docs=False, translate_answer=False)

        if translate_a or locale != 'en' and translate_src == 'en':
            answer = GoogleTranslator(source=translate_src, target=locale).translate(answer)

        source_documents = []
        for doc in docs:
            if translate_chunks:
                doc.page_content = GoogleTranslator(source=translate_src, target=locale).translate(doc.page_content)

            document_data = {
                'content': doc.page_content,
                'link': doc.metadata['source'],
            }
            if 'page' in doc.metadata:
                document_data['page'] = doc.metadata['page']
            if 'total_pages' in doc.metadata:
                document_data['total_pages'] = doc.metadata['total_pages']

            source_documents.append(document_data)

        response = {
            'answer': answer,
            'source_documents': source_documents
        }
        return response
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_files(request: Request):
    form = await request.form()
    database_name = form['database_name']
    collection_name = form.get('collection_name')  # Optional field

    files = form["files"]  # get files from form data

    # make sure files is a list
    if not isinstance(files, list):
        files = [files]

    saved_files = []
    source_documents = os.path.join(".", "source_documents")
    try:
        for file in files:
            content = await file.read()  # read file content
            if collection_name and database_name != collection_name:
                file_path = os.path.join(source_documents, database_name, collection_name, file.filename)
            else:
                file_path = os.path.join(source_documents, database_name, file.filename)

            saved_files.append(file_path)
            with open(file_path, "wb") as f:
                f.write(content)

            # assuming run_ingest is defined elsewhere
            run_ingest(database_name, collection_name)

            response = {
                'message': "OK",
                'files': saved_files,
                "database_name": database_name
            }
            return response
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
async def translate(item: TranslationItem):
    return {"translated_text": GoogleTranslator(source=item.src_lang, target=item.dst_lang).translate(item.text)}


###############################################################################
# Frontend
###############################################################################
@app.get("/")
def home():
    return FileResponse('scrapalot-research-assistant-ui/index.html')


@app.get("/{catch_all:path}")
def read_root(catch_all: str):
    return FileResponse('scrapalot-research-assistant-ui/index.html')


# commented out, because we use web UI
if __name__ == "__main__":
    import uvicorn

    path = 'api'
    # cert_path = "cert/cert.pem"
    # key_path = "cert/key.pem"
    print(f"Scrapalot API is now available at {api_scheme}://{api_host}:{api_port}/{path}")
    uvicorn.run(app, host=api_host, port=int(api_port))
    # uvicorn.run(app, host=host, port=int(port), ssl_keyfile=key_path, ssl_certfile=cert_path)
