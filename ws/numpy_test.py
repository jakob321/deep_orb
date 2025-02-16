import orbslam3
import numpy as np

# Create a NumPy array
arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
result = orbslam3.multiply_array(arr)

print(arr)
print("Modified array:", result)
