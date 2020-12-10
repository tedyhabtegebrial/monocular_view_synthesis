from PIL import Image
from torchvision.transforms import ToTensor, Compose
import torchvision
tran = Compose([ToTensor()])
img = tran(Image.open('./41800_novel.png'))
print(img.shape)
torchvision.utils.save_image(img[[2,1,0],:,:] * 2.0 - 1.0, './41800_novel_mod.png')
