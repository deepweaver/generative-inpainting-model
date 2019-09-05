

from test_raven import _get_generated_image


image_root = "/Users/tiany/GitHub/generative-inpainting-model/paper_pics/gestalt_images/"
image_filenames = ['brush2.png', 'fish_fin2.png', 'rocket2.png', 'spin2.png', 'yoyo2.png']
image_masks = ['brush2_mask.png', 'fish_fin2_mask.png', 'rocket2_mask.png', 'spin2_mask.png', 'yoyo2_mask.png']
image_inpainted = ['brush2_out.png', 'fish_fin2_out.png', 'rocket2_out.png', 'spin2_out.png', 'yoyo2_out.png']

images = []
for name in image_filenames: 

    images.append(cv2.imread(image_root + name))

masks = []
for name in image_masks:
    masks.append(cv2.imread(image_root + name))

inpainted = []
for i in range(len(images)):
    generated, _ = _get_generated_image(images[i],masks[i])
    cv2.imwrite(image_root+image_inpainted[i], generated)
