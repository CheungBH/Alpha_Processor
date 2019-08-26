import cv2

def resize_image(img, scale=None, size=None):
    assert scale is not None or size is not None
    if scale == None:
        return cv2.resize(img, size)
    else:
        height, width = img.shape[:2]
        return cv2.resize(img, (int(scale * width), int(scale * height)))
