import numpy as np
from PIL import Image

img = Image.open('data/input/test_images/section8-image.png')
lr_img = np.array(img)

img.save('data/input/test_images/compressed.jpeg','JPEG', dpi=[300, 300], quality=50)
compressed_img = Image.open('data/input/test_images/compressed.jpeg')
compressed_lr_img = np.array(compressed_img)

compressed_img.resize(size=(compressed_img.size[0]*2, compressed_img.size[1]*2), resample=Image.BICUBIC).save('n-base.png')

