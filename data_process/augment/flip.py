import cv2


def flip_image(img):
    return cv2.flip(img, 1)


if __name__ == '__main__':

    img=cv2.imread("lena.png")
    cv2.imshow("original",img)

    #水平镜像
    h_flip=cv2.flip(img,1)
    cv2.imshow("Flipped Horizontally",h_flip)

    #垂直镜像
    v_flip=cv2.flip(img,0)
    cv2.imshow("Flipped Vertically",v_flip)

    #水平垂直镜像
    hv_flip=cv2.flip(img,-1)
    cv2.imshow("Flipped Horizontally & Vertically",hv_flip)

    cv2.waitKey(0)
