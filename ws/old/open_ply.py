import numpy as np
import matplotlib.pyplot as plt

def load_pfm(file):
    with open(file, 'rb') as f:
        # Read header: "PF" for color, "Pf" for grayscale
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception("Not a PFM file.")

        # Read image dimensions
        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())
        
        # Read scale factor; its sign indicates endianness
        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'  # little-endian
            scale = -scale
        else:
            endian = '>'  # big-endian

        # Read the data and reshape it
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        
        # Flip vertically so the image is oriented correctly
        data = np.flipud(data)
    return data

# Load the PFM file
data = load_pfm("test2.pfm")
print(data.shape)

# If your image is grayscale, you might want to use a colormap:
if data.ndim == 2:
    plt.imshow(data, cmap='gray')
else:
    plt.imshow(data)
plt.axis('off')  # Hide axis ticks
plt.show()
