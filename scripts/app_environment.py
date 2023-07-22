import argparse
import multiprocessing
import os
import pathlib

import chromadb
import torch
from chromadb import Settings
from dotenv import load_dotenv

######################################################################
# General env calculations
######################################################################

# Number of CPU cores
n_cpu = multiprocessing.cpu_count()

# Now, if you want to use, say, 80% of your CPU cores:
cpu_model_n_threads = int(0.8 * n_cpu)

######################################################################
# GPU
######################################################################

# If CUDA is available, you can set a number of threads for GPU.
# Note: setting a specific number of threads for a GPU isn't as straightforward as for a CPU.
# For the sake of this example, let's assume you have determined that your GPU can efficiently use 'x' threads.
gpu_model_n_threads = int(os.environ.get("GPU_MODEL_N_THREADS", "16"))


# If you want to check if your system supports cuda:
def is_cuda_available() -> bool:
    return torch.cuda.is_available()


# currently supported devices: "cpu" and "cuda"
os_device_types = [
    "cpu", "cuda", "mps"
]


def detect_device():
    """
    Detects the type of device available.
    Returns:
        device: the available device. If multiple devices are available, priority is given in the order of the os_device_types list.
    """

    for device_type in os_device_types:
        if device_type == "cuda" and torch.cuda.is_available():
            print("CUDA GPU is available")
            device = torch.device("cuda")
            break
        elif device_type == "mps" and torch.cuda.is_available():  # 'mps' device type detection is treated as 'cuda' here for compatibility.
            print("Apple M1/M2 chip is available")
            device = torch.device("cuda")
            break
        elif device_type == "cpu":
            print("CPU is available")
            device = torch.device("cpu")
            break
    else:
        raise SystemError("None of the specified devices are available.")

    return device


# running_on_device = detect_device()

# Load environment variables
load_dotenv()

os_running_environment = os.environ.get('OS_RUNNING_ENVIRONMENT', "windows")

# Define the folder for storing database
ingest_persist_directory = os.environ.get('INGEST_PERSIST_DIRECTORY', 'db')

# Basic variables for ingestion
ingest_source_directory = os.environ.get('INGEST_SOURCE_DIRECTORY', 'source_documents')
ingest_embeddings_model = os.environ.get('INGEST_EMBEDDINGS_MODEL', 'hkunlp/instructor-large')
ingest_chunk_size = int(os.environ.get("INGEST_CHUNK_SIZE", "1000"))
ingest_chunk_overlap = int(os.environ.get("INGEST_OVERLAP", "100"))
ingest_target_source_chunks = int(os.environ.get('INGEST_TARGET_SOURCE_CHUNKS', '6'))

# Set the basic model settings
model_type = os.environ.get("MODEL_TYPE", "llamacpp")
model_n_ctx = os.environ.get("MODEL_N_CTX", "4096")
model_temperature = float(os.environ.get("MODEL_TEMPERATURE", "0.4"))
model_use_mlock = os.environ.get("MODEL_USE_MLOCK", "true") == "true"
model_verbose = os.environ.get("MODEL_VERBOSE", "false") == "true"
model_top_p = float(os.environ.get("MODEL_TOP_P", "0.9"))
model_n_batch = int(os.environ.get('MODEL_N_BATCH', "1024"))

# Settings specific for LLAMA
model_path_or_id = os.environ.get("MODEL_ID_OR_PATH")

# Setting specific for OpenAI models
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_use = os.environ.get("OPENAI_USE", "false") == "true"

# Setting specific for Huggingface models
huggingface_model_base_name = os.environ.get("MODEL_HF_BASE_NAME")

# Setting specific for GPT4All (can be llama or gptj)
gpt4all_backend = os.environ.get("GPT4ALL_BACKEND", "gptj")

# Setting specific for LLAMA GPU models
gpu_is_enabled = "true" if is_cuda_available() else "false"

# Setting specific for a database
db_get_only_relevant_docs = os.environ.get("DB_GET_ONLY_RELEVANT_DOCS", "false") == "true"

# Set desired translation preferences
translate_q = os.environ.get("TRANSLATE_QUESTION", "true") == "true"
translate_a = os.environ.get("TRANSLATE_ANSWER", "true") == "true"
translate_docs = os.environ.get("TRANSLATE_DOCS", "true") == "true"
translate_src = os.environ.get('TRANSLATE_SRC_LANG', "en")
translate_dst = os.environ.get('TRANSLATE_DST_LANG', "hr")

# Set the desired column width and the number of columns
cli_column_width = int(os.environ.get("CLI_COLUMN_WIDTH", "30"))
cli_column_number = int(os.environ.get("CLI_COLUMN_NUMBER", "4"))

# API
api_host = os.environ.get("API_HOST", "0.0.0.0")
api_port = int(os.environ.get("API_PORT", "8000"))
api_scheme = os.environ.get("API_SCHEME", "http")
api_base_url = os.environ.get("API_BASE_URL", f"{api_scheme}://{api_host}:{api_port}/api")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='scrapalot-chat: Ask questions to your documents without an internet connection, using the power of LLMs.')
    parser.add_argument(
        "--hide-source", "-S",
        action='store_true',
        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument(
        "--mute-stream", "-M",
        action='store_true',
        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    parser.add_argument(
        "--log-level", "-l",
        default=None,
        help='Set log level, for example -l INFO')

    parser.add_argument(
        "--ingest-embeddings-model",
        default=ingest_embeddings_model,
        help="Embeddings model name",
    )
    parser.add_argument(
        "--model-path-or-id",
        default=model_path_or_id,
        help="Model path",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--collection",
        help="Saves the embedding in a collection name as specified"
    )
    parser.add_argument(
        "--ingest-chunk-size",
        type=int,
        default=ingest_chunk_size,
        help="Chunk size",
    )
    parser.add_argument(
        "--ingest-chunk-overlap",
        type=int,
        default=ingest_chunk_overlap,
        help="Chunk overlap",
    )
    parser.add_argument(
        "--ingest-target-source-chunks",
        type=int,
        default=ingest_target_source_chunks,
        help="Target source chunks",
    )
    parser.add_argument(
        "--ingest-dbname",
        type=str,
        help="Name of the database directory",
    )
    parser.add_argument(
        "--ingest-all",
        action="store_true",
        help="Ingest all directories in source_documents"
    )

    return parser.parse_args()


args = parse_arguments()


class ChromaDBClientManager:
    def __init__(self):
        self.clients = {}

    @staticmethod
    def get_chroma_setting(persist_dir: str):
        return Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )

    def get_client(self, database_name: str):
        if database_name not in self.clients:
            persist_directory = f"./db/{database_name}"
            self.clients[database_name] = chromadb.Client(self.get_chroma_setting(persist_directory))
        return self.clients[database_name]


chromaDB_manager = ChromaDBClientManager()
