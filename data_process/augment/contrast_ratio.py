
def adjust_image(img, alpha, beta):
    rows, cols, channels = img.shape
    dst = img.copy()

    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = img[i, j][c] * alpha + beta
                if color > 255:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 255
                elif color < 0:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 0
                else:
                    dst[i, j][c] = color
    return dst
