import os
import tempfile
from typing import List, Optional

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from ingestion_storage.app_storage import AbstractStorage


class GoogleDriveStorage(AbstractStorage):

    def __init__(self):
        # Don't create the GoogleAuth and GoogleDrive instances here because gauth can't be pickled (serialized)
        self._gauth = None
        self._drive = None
        pass

    @property
    def gauth(self):
        if self._gauth is None:
            self._gauth = GoogleAuth()
            self._gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
        return self._gauth

    @property
    def drive(self):
        if self._drive is None:
            self._drive = GoogleDrive(self.gauth)
        return self._drive

    def get_type(self) -> str:
        return "google_drive_storage"

    def get_folder_id(self, path: str) -> Optional[str]:
        path_parts = path.split(os.sep)
        parent_id = 'root'
        for part in path_parts:
            # Skip over any "." in the path
            if part == ".":
                continue
            query = f"title='{part}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.drive.ListFile({'q': query}).GetList()
            if len(results) == 1:
                parent_id = results[0]['id']
            else:
                return None  # Folder not found
        return parent_id

    def create_directory(self, directory: str) -> None:
        # Google Drive doesn't have a concept of directories in the way local file systems do.
        # Everything is a file, and "directories" are just special files that can have other files as children.
        # So to create a directory, we actually create a file with the MIME type for a Google Drive folder.
        dir_path, folder_name = os.path.split(directory)
        parent_id = self.get_folder_id(dir_path)
        if parent_id is not None:
            folder_metadata = {
                'title': folder_name,
                'parents': [{'id': parent_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()
        else:
            raise FileNotFoundError(f"Directory {dir_path} not found")

    def directory_root(self, directory: str) -> str:
        # Normalize path (replace '\' with '/')
        directory = os.path.normpath(directory)

        # Split the directory path into components
        directory_components = directory.split(os.sep)

        # Start from the root of the drive
        current_parent_id = 'root'
        # For each component in the directory path
        for component in directory_components:
            # Search for a folder with the current title and parent ID
            query = f"title='{component}' and '{current_parent_id}' in parents and trashed=false"
            results = self.drive.ListFile({'q': query}).GetList()

            # If no such folder exists, the directory does not exist
            if len(results) == 0:
                raise FileNotFoundError(f"Directory '{directory}' does not exist in Google Drive")

            # Otherwise, update the current parent ID and continue with the next component
            current_parent_id = results[0]['id']

        return current_parent_id

    def list_dirs_src(self, directory: str) -> List[str]:

        current_parent_id = self.directory_root(directory)

        # Once the parent directory is found, list all subdirectories in it
        query = f"'{current_parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        dir_list = self.drive.ListFile({'q': query}).GetList()

        return [src_directory['title'] for src_directory in dir_list]

    def list_files_src(self, directory: str) -> List[str]:
        current_parent_id = self.directory_root(directory)

        # Once the parent directory is found, list all files in it
        query = f"'{current_parent_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        return [os.path.join(directory, file['title']) for file in file_list]

    def list_files_db(self, persist_directory: str) -> List[str]:
        return self.list_files_src(persist_directory)

    def get_file_path(self, file_path: str) -> str:
        dir_path, file_name = os.path.split(file_path)
        parent_id = self.get_folder_id(dir_path)
        if parent_id is not None:
            query = f"'{parent_id}' in parents and title='{file_name}' and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            if len(file_list) == 1:
                google_drive_file = self.drive.CreateFile({'id': file_list[0]['id']})
                temp_file_path = tempfile.mktemp(suffix=".pdf")  # Create a temp file path
                google_drive_file.GetContentFile(temp_file_path)  # Download the file to the temp file
                return temp_file_path
            else:
                raise FileNotFoundError(f"File {file_path} not found")
        else:
            raise FileNotFoundError(f"Directory {dir_path} not found")

    def upload_file(self, file_path: str, data: bytes) -> None:
        dir_path, file_name = os.path.split(file_path)
        parent_id = self.get_folder_id(dir_path)
        if parent_id is not None:
            file_metadata = {
                'title': file_name,
                'parents': [{'id': parent_id}]
            }
            file = self.drive.CreateFile(file_metadata)
            file.SetContentString(data)
            file.Upload()
        else:
            raise FileNotFoundError(f"Directory {dir_path} not found")

    def delete_file(self, file_path: str) -> None:
        dir_path, file_name = os.path.split(file_path)
        parent_id = self.get_folder_id(dir_path)
        if parent_id is not None:
            query = f"'{parent_id}' in parents and title='{file_name}' and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            if len(file_list) == 1:
                file_list[0].Delete()
            else:
                raise FileNotFoundError(f"File {file_path} not found")
        else:
            raise FileNotFoundError(f"Directory {dir_path} not found")

    def path_exists(self, path: str) -> bool:
        # Normalize path (replace '\' with '/')
        path = os.path.normpath(path)

        # Split the path into components
        path_components = path.split(os.sep)

        # Start from the root of the drive
        current_parent_id = 'root'

        # For each component in the path
        for component in path_components:
            # Search for a file or folder with the current title and parent ID
            query = f"title='{component}' and '{current_parent_id}' in parents and trashed=false"
            results = self.drive.ListFile({'q': query}).GetList()

            # If no such file or folder exists, the path does not exist
            if len(results) == 0:
                return False

            # Otherwise, update the current parent ID and continue with the next component
            current_parent_id = results[0]['id']

        # If we found a file or folder for every component, the path exists
        return True

    def is_directory(self, directory: str) -> bool:
        # If method get_folder_id returns None, the path does not exist or is not a directory
        # Otherwise, the path is a directory
        return self.get_folder_id(directory) is not None

    def is_file(self, file_path: str) -> bool:
        return self.path_exists(file_path)
