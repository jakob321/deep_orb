import h5py

# Open the HDF5 file
file_path = "../../datasets/midair/MidAir/PLE_training/winter/sensor_records.hdf5"

# with h5py.File(file_path, "r") as f:
#     def print_hdf5_structure(name, obj):
#         print(name, ":", type(obj))

#     print("File Structure:")
#     f.visititems(print_hdf5_structure)

with h5py.File(file_path, "r") as f:
    dataset_path = "trajectory_6000/groundtruth/position"

    if dataset_path in f:
        data = f[dataset_path][:]  # Load the dataset as a NumPy array
        print("Shape of dataset:", data.shape)
        print("First few entries:\n", data[:5])  # Print first 5 entries
    else:
        print("Dataset not found in the file.")
