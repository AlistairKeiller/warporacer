import matplotlib.pyplot as plt
from skimage import filters
from pathlib import Path
from yaml import safe_load

from cv2 import (
    IMREAD_GRAYSCALE,
    imread,
)

OCC_THRESH = 230

path = Path(".\\maps\\stata_basement.yaml")
meta = safe_load(path.read_text())
img_path = path.parent / meta["image"]
raw = imread(str(img_path), IMREAD_GRAYSCALE)
free = raw >= OCC_THRESH

image = free
edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()