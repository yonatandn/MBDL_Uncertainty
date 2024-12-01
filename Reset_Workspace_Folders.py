import os


def delete_files_in_folder(folder_path):
    try:
        # List all files in the given folder
        files = os.listdir(folder_path)

        for file in files:
            file_path = os.path.join(folder_path, file)

            # Check if it is a file before deleting
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        print("All files deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


delete_files_in_folder('.data/StateSpace/test/')
delete_files_in_folder('.data/StateSpace/train/')
delete_files_in_folder('.data/StateSpace/valid/')
delete_files_in_folder('.figures/')
delete_files_in_folder('.model_saved/')
delete_files_in_folder('.results/')