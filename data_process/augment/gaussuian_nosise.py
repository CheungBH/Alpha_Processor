import numpy as np
import cv2


def gaussian_noise(img, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(img/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return img + out


if __name__ == '__main__':
    img_path = "../../img/test.jpg"
    saved_path = img_path.replace(".jpg", "_gau.jpg")
    img = cv2.imread(img_path)
    cut_img = gaussian_noise(img)
    cv2.imwrite(saved_path, cut_img)
