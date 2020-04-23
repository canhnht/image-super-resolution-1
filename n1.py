import numpy as np
from PIL import Image

div2k_x2 = '/home/acworks/Documents/ac-kakudai-kirei/div2k-2017/DIV2K_valid_LR_unknown/X2'
fname = '0801x2.png'
#compressed_img = Image.open('data/input/test_images/compressed.jpeg')
compressed_img = Image.open(div2k_x2 + "/" + fname)
compressed_lr_img = np.array(compressed_img)

from ISR.models import RDN

rdn = RDN(weights='psnr-large')

sr_img = rdn.predict(compressed_lr_img)
Image.fromarray(sr_img).save('n1-' + fname)


