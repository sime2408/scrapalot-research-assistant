import glob
import os
from typing import List

from ingestion_storage.app_storage import AbstractStorage
from scripts.app_utils import LOADER_MAPPING


class LocalStorage(AbstractStorage):
    def __init__(self):
        # Code to configure local storage
        pass

    def get_type(self) -> str:
        return "local_storage"

    def create_directory(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

    def list_dirs_src(self, parent_dir: str) -> List[str]:
        return os.listdir(parent_dir)

    def list_files_src(self, directory: str) -> List[str]:
        all_files = []
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(directory, f"*{ext}"), recursive=False)
        )
        return all_files

    def list_files_db(self, persist_directory: str) -> List[str]:
        list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
        list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
        return list_index_files

    def get_file_path(self, file_path: str) -> str:
        return file_path

    def upload_file(self, file_path: str, data: bytes) -> None:
        # Code to upload a file to Local Storage
        pass

    def delete_file(self, file_path: str) -> None:
        # Code to delete a file from Local Storage
        pass

    def path_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_directory(self, directory: str) -> bool:
        return os.path.isdir(directory)

    def is_file(self, file_path: str) -> bool:
        return os.path.isfile(file_path)
