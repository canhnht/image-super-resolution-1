import numpy as np
from PIL import Image

img = Image.open('data/input/test_images/section8-image.png')
lr_img = np.array(img)

from ISR.models import RDN

#rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
#rdn.model.load_weights('weights/sample_weights/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')
rdn = RDN(weights='psnr-large')

sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img).save('e1.png')


