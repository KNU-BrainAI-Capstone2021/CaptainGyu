import os
import zipfile
from torchvision import transforms
from PIL import Image

def conversion(src_im_dir, storage_dir, DF_part):
    storage_im_dir = os.path.join(storage_dir, 'img')
    storage_seg_dir = os.path.join(storage_dir, 'seg')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unloader = transforms.ToPILImage()

    src_file_name_list = os.listdir(src_im_dir)
    src_file_dir_list = []
    for idx, file in enumerate(src_file_name_list):
        src_file_dir_list.append(os.path.join(src_im_dir, src_file_name_list[idx]))

    for file_name_index, file_dir in enumerate(src_file_dir_list):
        x = read_image(file_dir) / 255.0
        output = DF_part(x.unsqueeze(0).to(device))
        op_im = output[0].squeeze(0)
        op_seg = output[1].squeeze(0)

        op_seg = op_seg.reshape(256 * 256)
        for idx, unit in enumerate(op_seg):
            if unit != 1:
                op_seg[idx] = 0
        op_seg = op_seg.reshape((1, 256, 256))
        print(op_im.shape)
        print(op_seg.shape)

        op_im = unloader(op_im)
        op_seg = unloader(op_seg)

        op_im.save(os.path.join(storage_im_dir, src_file_name_list[file_name_index]))
        op_seg.save(os.path.join(storage_seg_dir, src_file_name_list[file_name_index]))

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

