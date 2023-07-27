from abc import ABC, abstractmethod
from typing import List


class AbstractStorage(ABC):

    @abstractmethod
    def get_type(self) -> str:
        """Get the type of this storage"""
        pass

    @abstractmethod
    def create_directory(self, directory: str) -> None:
        """Create a directory."""
        pass

    @abstractmethod
    def list_dirs_src(self, parent_dir: str) -> List[str]:
        """List all subdirectories in the given directory."""
        pass

    @abstractmethod
    def list_files_src(self, directory: str) -> List[str]:
        """List all files in the given directory."""
        pass

    @abstractmethod
    def list_files_db(self, persist_directory: str) -> List[str]:
        """Create a directory."""
        pass

    @abstractmethod
    def get_file_path(self, file_path: str) -> str:
        """Download a file and return its contents."""
        pass

    @abstractmethod
    def upload_file(self, file_path: str, data: bytes) -> None:
        """Upload data to a file."""
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> None:
        """Delete a file."""
        pass

    @abstractmethod
    def path_exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    def is_directory(self, directory: str) -> bool:
        """Check if a directory exists."""
        pass

    @abstractmethod
    def is_file(self, file_path: str) -> bool:
        """Check if a file exists."""
        pass
