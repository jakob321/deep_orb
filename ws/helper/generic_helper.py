import pickle
import os

def save_data(file_name, *objects, path="", use_full_path=False):
    """
    Save multiple arbitrary objects to a file using pickle.
    
    Args:
        file_name (str): Name of the file to save to
        *objects: Objects to save
        path (str): Optional subdirectory path
        use_full_path (bool): If True, file_name is used as is; otherwise it's processed
    """
    print("saving data to file")
    
    if use_full_path:
        s = file_name
    else:
        file_name = file_name.replace(".", "").replace("/", "_")
        s = os.path.join("saved_values", path, file_name)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(s), exist_ok=True)
    
    try:
        with open(s, "wb") as f:
            pickle.dump(objects, f)  # Save all objects as a tuple
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


def load_data(file_name, use_full_path=False):
    """
    Load saved data from a file if it exists, otherwise return False.
    
    Args:
        file_name (str): Name of the file to load from
        use_full_path (bool): If True, file_name is used as is; otherwise it's processed
    
    Returns:
        The loaded objects or False if loading failed
    """
    if use_full_path:
        s = file_name
    else:
        file_name = file_name.replace(".", "").replace("/", "_")
        s = os.path.join("saved_values", file_name)
    
    print(f"trying to load: {s}")
    
    if not os.path.exists(s):  # Check if file exists
        return False
    
    try:
        print("loading data from file")
        with open(s, "rb") as f:
            loaded_data = pickle.load(f)  # Load all saved objects
            # If there's only one object in the tuple, return just that object
            if isinstance(loaded_data, tuple) and len(loaded_data) == 1:
                return loaded_data[0]
            return loaded_data
    except (pickle.UnpicklingError, EOFError, Exception) as e:  # Handle corrupted files
        print(f"No file to load or corrupted file: {e}")
        return False
