import argparse
import os
import subprocess


def convert_files(root_folder, src_extension, dst_extension, converter_cmd):
    """
    The Requirement to run this script is to install Calibre eBook management software
    :param root_folder: where your documents are stored
    :param src_extension: from which format you want to convert
    :param dst_extension: to which format you want to convert
    :param converter_cmd: CMD tool to convert the files
    :return:
    """
    # Walk through the root folder and its sub-folders
    for dir_path, dirs, files in os.walk(root_folder):
        for filename in files:
            f_name = os.path.join(dir_path, filename)
            if f_name.endswith(src_extension):
                # Generate new filename
                new_file_name = os.path.splitext(f_name)[0] + dst_extension
                # Use the specified command line tool
                subprocess.run([converter_cmd, f_name, new_file_name], shell=True)
                # Delete the source file after conversion
                os.remove(f_name)


# Create an argument parser
parser = argparse.ArgumentParser(description="Convert files from one format to another.")
parser.add_argument("-s", "--subdirectory", help="Subdirectory to parse instead of 'source_documents'.")
args = parser.parse_args()

# Source and target file extensions
root_folder = 'source_documents'
if args.subdirectory:
    root_folder = os.path.join(root_folder, args.subdirectory)

convert_files(root_folder, '.epub', '.pdf', 'ebook-convert')
