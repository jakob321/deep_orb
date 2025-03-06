import pickle
import os

def save_data(file_name, *objects):
    """Save multiple arbitrary objects to a file using pickle."""
    print("saving data to file")
    file_name = file_name.replace(".", "").replace("/", "_")
    s = "saved_values/"+file_name
    try:
        with open(s, "wb") as f:
            pickle.dump(objects, f)  # Save all objects as a tuple
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_data(file_name):
    """Load saved data from a file if it exists, otherwise return False."""
    file_name = file_name.replace(".", "").replace("/", "_")
    s = "saved_values/"+file_name
    print("trying to load: "+s)
    if not os.path.exists(s):  # Check if file exists
        return False
    try:
        print("loading data from file")
        with open(s, "rb") as f:
            return pickle.load(f)  # Load all saved objects
    except (pickle.UnpicklingError, EOFError, Exception) as e:  # Handle corrupted files
        print(f"No file to load or corrupted file: {e}")
        return False
