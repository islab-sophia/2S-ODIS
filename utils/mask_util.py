import cv2
import numpy as np
import random
from PIL import Image

def mask_image(img=None,height=None,width=None,random_mask_inversion=True):
    if height is None and width is None:
        height,width = img.height,img.width
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    color = (0,0,0)

    for _ in range(random.randint(0, 2)):
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        thickness = random.randint(10, 50)
        cv2.line(canvas, pt1, pt2, color, thickness)
    for _ in range(random.randint(0, 2)):
        center = (random.randint(0, width), random.randint(0, height))
        radius = random.randint(30, 200)
        cv2.circle(canvas, center, radius, color, -1)
    for _ in range(random.randint(0, 2)):
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        cv2.rectangle(canvas, pt1, pt2, color,-1)

    if random_mask_inversion:
        if random.random()<0.5:
            canvas = 255-canvas
    return canvas/255.0

if __name__ == '__main__':
    img = Image.open("test_3.jpg")
    mask = 1-mask_image(img,random_mask_inversion=False)
    masked_img = mask * np.array(img)/255.
    import matplotlib.pyplot as plt
    print(np.array(img).max(),mask.max())
    plt.imshow(masked_img)
    plt.show()
    recon_img = (np.clip(masked_img * 255, 0, 255)).astype(np.uint8)
    Image.fromarray(recon_img).save("masked_img.png")