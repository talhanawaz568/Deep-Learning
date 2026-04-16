import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 1. Setup - Create a dummy image if one doesn't exist
# In your lab, you would use: img = load_img('path/to/your/image.jpg')
# For this script, we'll create a simple colored square
img_array = np.zeros((150, 150, 3), dtype=np.uint8)
img_array[30:70, 30:70, 0] = 255  # Red square

# 2. Define Augmentation Strategy
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 3. Visualize
img_array_expanded = np.expand_dims(img_array, 0)
aug_iter = datagen.flow(img_array_expanded, batch_size=1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 5, 1)
plt.imshow(img_array)
plt.title("Original")

for i in range(2, 6):
    batch = next(aug_iter)
    image = batch[0].astype('uint8')
    plt.subplot(1, 5, i)
    plt.imshow(image)
    plt.title(f"Augmented {i-1}")

plt.savefig('augmentation_lab.png')
print("✓ Augmentation visualization saved to 'augmentation_lab.png'")
