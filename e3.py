import numpy as np
from PIL import Image

img = Image.open('data/input/test_images/section8-image.png')
lr_img = np.array(img)

from ISR.models import RDN

rdn = RDN(weights='noise-cancel')

sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img).save('e3.png')


