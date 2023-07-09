import os
from typing import List, Optional

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from ingestion_storage.app_storage import AbstractStorage


class GoogleDriveStorage(AbstractStorage):

    def __init__(self):
        self.gauth = GoogleAuth()
        self.gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
        self.drive = GoogleDrive(self.gauth)

    def get_folder_id(self, path: str) -> Optional[str]:
        path_parts = path.split(os.sep)
        parent_id = 'root'
        for part in path_parts:
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
        folder_metadata = {
            'title': directory,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = self.drive.CreateFile(folder_metadata)
        folder.Upload()

    def list_files_src(self, directory: str) -> List[str]:
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

        # Once the parent directory is found, list all files in it
        query = f"'{current_parent_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        return [os.path.join(directory, file['title']) for file in file_list]

    def list_files_db(self, persist_directory: str) -> List[str]:
        # Normalize path (replace '\' with '/')
        persist_directory = os.path.normpath(persist_directory)

        # Split the directory path into components
        directory_components = persist_directory.split(os.sep)

        # Start from the root of the drive
        current_parent_id = 'root'

        # For each component in the directory path
        for component in directory_components:
            # Search for a folder with the current title and parent ID
            query = f"title='{component}' and '{current_parent_id}' in parents and trashed=false"
            results = self.drive.ListFile({'q': query}).GetList()

            # If no such folder exists, the directory does not exist
            if len(results) == 0:
                raise FileNotFoundError(f"Directory '{persist_directory}' does not exist in Google Drive")

            # Otherwise, update the current parent ID and continue with the next component
            current_parent_id = results[0]['id']

        # Once the parent directory is found, list all files in it
        query = f"'{current_parent_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        return [file['title'] for file in file_list]

    def download_file(self, file_path: str) -> bytes:
        dir_path, file_name = os.path.split(file_path)
        parent_id = self.get_folder_id(dir_path)
        if parent_id is not None:
            query = f"'{parent_id}' in parents and title='{file_name}' and trashed=false"
            file_list = self.drive.ListFile({'q': query}).GetList()
            if len(file_list) == 1:
                file_content = file_list[0].GetContentString()
                return file_content
            else:
                raise FileNotFoundError(f"File {file_path} not found")
        else:
            raise FileNotFoundError(f"Directory {dir_path} not found")

    def upload_file(self, file_path: str, data: bytes) -> None:
        file = self.drive.CreateFile({'title': os.path.basename(file_path)})
        file.SetContentString(data)
        file.Upload()

    def delete_file(self, file_path: str) -> None:
        file_list = self.drive.ListFile({'q': f"title='{file_path}'"}).GetList()
        if len(file_list) == 1:
            file_list[0].Delete()
        else:
            raise FileNotFoundError(f"File {file_path} not found")

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

    def is_file(self, file_path: str) -> bool:
        return self.path_exists(file_path)
