import os
import shutil

source_path = '/home/share/chenaoxuan/laion2b_chinese_release/bench/00000/'
target_path = '/home/share/chenaoxuan/vq-vae-2-pytorch-master/dataset/0/'

if __name__ == '__main__':
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    n = 5000
    count = 0
    i = 0
    while True:
        if count == n:
            break
        name_img = str(i).zfill(15) + '.jpg'
        name_txt = str(i).zfill(15) + '.txt'
        i += 1
        srcfile_img = os.path.join(source_path, name_img)
        tarfile_img = os.path.join(target_path, name_img)
        srcfile_txt = os.path.join(source_path, name_txt)
        tarfile_txt = os.path.join(target_path, name_txt)
        if not os.path.isfile(srcfile_img) and not os.path.isfile(srcfile_txt):
            print("%s and %s not exist!" % (srcfile_img, srcfile_txt))
            continue
        count += 1
        if not os.path.isfile(tarfile_img):
            shutil.copy(srcfile_img, tarfile_img)
            print("copy %s -> %s" % (srcfile_img, tarfile_img))
        else:
            print("%s is exist!" % (tarfile_img))
        if not os.path.isfile(tarfile_txt):
            shutil.copy(srcfile_txt, tarfile_txt)
            print("copy %s -> %s" % (srcfile_txt, tarfile_txt))
        else:
            print("%s is exist!" % (tarfile_txt))
