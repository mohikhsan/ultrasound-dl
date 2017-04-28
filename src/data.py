"""
Load training and train data from image files
"""
import os
import numpy as np
from skimage.io import imsave, imread

db_path = 'db/'
img_height = 480
img_width = 640

def create_db(roi=None):
    """
    Create db files from raw images

    Parameters:
        -roi: region of interest [y,height,x,width]
    """
    if roi is not None:
        roi_y = roi[0]
        roi_height = roi[1]
        roi_x = roi[2]
        roi_width = roi[3]
    else:
        roi_y = 0
        roi_height = img_height
        roi_x = 0
        roi_width = img_width

    subjects = [name for name in os.listdir(db_path) if os.path.isdir(os.path.join(db_path,name))]

    num_img = 0
    db_structure = []
    for subject in subjects:
        subject_path = os.path.join(db_path, subject)
        num_subject_files = len([fname for fname in os.listdir(subject_path) if os.path.isfile(os.path.join(subject_path, fname))])
        num_img += num_subject_files
        db_structure.append([subject,num_subject_files])

    imgs = np.ndarray((num_img, roi_height, roi_width), dtype=np.uint8)

    idx = 0
    print("Loading image...")
    for subject in subjects:
        subject_path = os.path.join(db_path, subject)
        images = os.listdir(subject_path)
        for image_fname in images:
            if image_fname.endswith('.png'):
                img = imread(os.path.join(subject_path, image_fname), as_grey=True)
                if roi is not None:
                    img = np.array([img[roi_y:roi_y+roi_height,roi_x:roi_x+roi_width]])
                else:
                    img = np.array([img])
                imgs[idx] = img

                if idx % 100 == 0:
                    print('Completed {0}/{1} images'.format(idx, num_img))

                idx += 1

    print("Loading complete.")

    np.save('img_db.npy',imgs)

    print("Images saved to img_db.npy")

def load_db():
    try:
        return np.load('img_db.npy')
    except:
        return None
