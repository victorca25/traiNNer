import sys
import os.path
import glob
import pickle
import lmdb
import cv2
import argparse

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.progress_bar import ProgressBar
    from utils.util import scandir
except ImportError:
    pass



def prepare_lmdb_keys(folder_path):
    """Prepare image path list and keys for dataset.
    Args:
        folder_path (str): Folder path.
    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        map_size=None):
    """Make lmdb from images.
    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt
    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.
    The meta_info.txt is a specified txt file to record the meta information
    of the datasets. Each line in the txt file records 1)image name 
    (with extension), 2)image shape, and 3)compression level, separated 
    by a white space.
    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1
    The image name is used without extension as the lmdb key.
    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing 'batch' number of images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None
    """

    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    #### check if the lmdb file exist
    if os.path.exists(lmdb_path):
        print('Folder [{:s}] already exists. Exit.'.format(lmdb_path))
        sys.exit(1)

    # create lmdb environment
    if map_size is None:
        # obtain data size for one image
        img = cv2.imread(
            os.path.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = ProgressBar(len(img_path_list)) #tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)  # txn is a Transaction object
    txt_file = open(os.path.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        # pbar.update(1)
        # pbar.set_description(f'Write {key}')
        pbar.update('Write {}'.format({key}))
        key_byte = key.encode('ascii')
        _, img_byte, img_shape = read_img_worker(
            os.path.join(data_path, path), key, compress_level)
        h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    # pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def read_img_worker(path, key, compress_level):
    """Read image worker.
    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.
    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-images_path', type=str, required=True, 
        help='Path to image folder. Example: D:/hr')
    parser.add_argument(
        '-lmdb_path', type=str, required=False, 
        help='Path to output lmdb. Must end with .lmdb Example: D:/hr.lmdb')

    args = parser.parse_args()
    img_folder = args.images_path

    if args.lmdb_path:
        lmdb_save_path = args.lmdb_path
    else:
        lmdb_save_path = img_folder.rstrip("/")
        lmdb_save_path += '.lmdb'
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")

    return img_folder, lmdb_save_path




def main():

    img_folder, lmdb_save_path = parse_options()
    
    img_path_list, keys = prepare_lmdb_keys(img_folder)
    create_lmdb_from_imgs(img_folder, lmdb_save_path, img_path_list, keys)

    

if __name__ == '__main__':
    main()