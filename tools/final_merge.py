import os


def division(ls):
    dpath=[]
    lpath=[]
    for i in ls:
        if i[-9:-4] == 'label':
            lpath.append(i)
        else:
            dpath.append(i)
    return lpath, dpath


def merge_txt(txtlist, folder, name):
    for filename in txtlist:     #逐一读取需要合并的文件
        final_path = os.path.join(folder, name + ".txt")#不同动作的文件合并进不同的最终文件中
        final_file = open(final_path, 'a')      #将最终文件打开为追加写模式
        for line in open(filename):
            final_file.writelines(line)       #将源文件的每一行写入最终文件中
        final_file.close()


def merge(main_folder):
    txt_ls = []
    final_folder = os.path.join(main_folder, "all")
    os.makedirs(final_folder, exist_ok=True)
    for txt_name in os.listdir(main_folder):
        if "all" not in txt_name:
            txt_ls.append(os.path.join(main_folder, txt_name))
    label_path, data_path = division(txt_ls)
    merge_txt(label_path, final_folder, 'label')
    merge_txt(data_path, final_folder, 'data')


if __name__ == '__main__':
    path = r'C:\Users\hkuit164\Desktop\AlphaPose-pytorch\temp\sport00\zzz_data'
    merge(path)

