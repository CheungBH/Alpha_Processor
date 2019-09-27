import data_process.augment.flip as flip
import data_process.augment.gaussuian_nosise as gau
import data_process.augment.cut as cut
import data_process.augment.contrast_ratio as cr
import data_process.augment.sp_noise as sp
import data_process.augment.resize as resize
import cv2
import os

video_stored_path = "Video/0827"


def augment_video(path):
    os.makedirs(video_stored_path, exist_ok=True)
    cam = cv2.VideoCapture(path)
    cnt = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #writer = cv2.VideoWriter(os.path.join(video_stored_path, "flip.mp4"), fourcc, 20.0, (640, 480))
    # writer_flip = cv2.VideoWriter(os.path.join(video_stored_path, "flip.mp4"), fourcc, 20.0, (640, 480))

    while True and not cv2.waitKey(1) == 27:
        ret, img = cam.read()
        if ret:
            # writer_flip.write(img)
            cv2.imshow("origin", img)
            cv2.imwrite(os.path.join(video_stored_path, "input_{}.jpg".format(cnt)), img)

            # img_flip = flip.flip_image(img)
            # cv2.imwrite(os.path.join(video_stored_path, "flip/{}.jpg".format(cnt)), img_flip)
            #
            # img_resize = resize.resize_image(img, scale=0.5)
            # writer_resize = cv2.VideoWriter(os.path.join(video_stored_path, "resize.mp4"), fourcc, 20.0, (img_resize.shape[0], img_resize.shape[1]))
            # writer_resize.write(img_resize)
            #
            # img_cut = cut.cut_image(img, bottom=100, top=100, left=20, right=30)
            # writer_cut = cv2.VideoWriter(os.path.join(video_stored_path, "cut.mp4"), fourcc, 20.0, (img_cut.shape[0], img_cut.shape[1]))
            # writer_cut.write(img_cut)
            #
            # img_cr = cr.adjust_image(img, alpha=0.5, beta=40)
            # write_cr = cv2.VideoWriter(os.path.join(video_stored_path, "cr.mp4"), fourcc, 20.0,(img_cr.shape[0], img_cr.shape[1]))
            # write_cr.write(img_cr)
            #
            # img_gau_noise = gau.gaussian_noise(img)
            # writer_gau = cv2.VideoWriter(os.path.join(video_stored_path, "gaussian.mp4"), fourcc, 20.0,(img_gau_noise.shape[0], img_gau_noise.shape[1]))
            # writer_gau.write(img_gau_noise)
            #
            # img_sp = sp.sp_noise(img, prob=0.1)
            # writer_sp = cv2.VideoWriter(os.path.join(video_stored_path, "sp.mp4"), fourcc, 20.0,(img_sp.shape[0], img_sp.shape[1]))
            # writer_sp.write(img_sp)

            cnt += 1
            # print(cnt)

        else:
            break


def augment_image(src_path, flip_path=None, resize_path=None, cr_path=None, cut_path=None, gau_path=None, sp_path=None):
    img = cv2.imread(src_path)

    flip_img = flip.flip_image(img)
    cv2.imwrite(flip_path, flip_img)

    resize_img = resize.resize_image(img, scale=0.5)
    cv2.imwrite(resize_path, resize_img)

    cut_img = cut.cut_image(img, bottom=100, top=50, left=20)
    cv2.imwrite(cut_path, cut_img)

    cr_img = cr.adjust_image(img, alpha=0.8, beta=40)
    cv2.imwrite(cr_path, cr_img)

    gau_img = gau.gaussian_noise(img, var=0.0000001)
    cv2.imwrite(gau_path, gau_img)

    sp_img = sp.sp_noise(img, prob=0.001)
    cv2.imwrite(sp_path, sp_img)


def augment_image_folder(folder_path):
    os.makedirs(os.path.join(folder_path, "flip"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "resize"),exist_ok=True)
    os.makedirs(os.path.join(folder_path, "cr"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "cut"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "gau_noise"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "sp_noise"), exist_ok=True)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isdir(image_path):
            continue
        else:
            augment_image(image_path, flip_path=os.path.join(folder_path, "flip", image_name), resize_path=os.path.join(folder_path, "resize", image_name), cr_path=os.path.join(folder_path, "cr", image_name), cut_path=os.path.join(folder_path, "cut", image_name), gau_path=os.path.join(folder_path, "gau_noise", image_name), sp_path=os.path.join(folder_path, "sp_noise", image_name))


if __name__ == '__main__':
    # main_folder = 'img/0826'
    # augment_image_folder(main_folder)
    os.makedirs(video_stored_path, exist_ok=True)
    augment_video("Video/origin.mp4")
