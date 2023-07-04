import os
import subprocess


def convert_files(root_folder, src_extension, dst_extension, converter_cmd):
    # Walk through the root folder and its sub-folders
    for dir_path, dirs, files in os.walk(root_folder):
        for filename in files:
            f_name = os.path.join(dir_path, filename)
            if f_name.endswith(src_extension):
                # Generate new filename
                new_file_name = os.path.splitext(f_name)[0] + dst_extension
                # Use the specified command line tool
                subprocess.run([converter_cmd, f_name, new_file_name], shell=True)


# Source and target file extensions
convert_files('source_documents', '.epub', '.pdf', 'ebook-convert')
