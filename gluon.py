from gluoncv.data import CitySegmentation
from gluoncv.utils.viz import get_color_pallete
import scipy.misc as misc
from matplotlib import pyplot as plt

# train_dataset = CitySegmentation(root='/home/anson/sda/dataset/cityscapes/', split='train')

# print('Training images:', len(train_dataset))

# val_dataset = CitySegmentation(root='/home/anson/sda/dataset/cityscapes/', split='val')
# print('Validation images:', len(val_dataset))
# img, mask = val_dataset[0]
# print(mask[3, :])

lable = misc.imread('/home/anson/sda/dataset/cityscapes/gtFine/val/frankfurt/frankfurt_000000_001016_gtFine_labelIds.png')
plt.imsave('val.png', lable)
mask = get_color_pallete(lable, dataset='citys')
plt.imshow(mask)
# display
plt.show()




