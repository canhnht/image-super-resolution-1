import numpy as np
from PIL import Image

img = Image.open('data/input/test_images/section8-image.png')
lr_img = np.array(img)

from ISR.models import RDN

#rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
#rdn.model.load_weights('weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
rdn = RDN(weights='psnr-small')

sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img).save('e2.png')


