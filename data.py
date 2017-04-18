"""
Load training and train data from image files
"""
import os
import numpy as np
from skimage import imsave, imread

db_path = 'db/'

roi_width = 300
roi_height = 200
roi_x = 100
roi_y = 100

def create_db():
    subjects = os.listdir(db_path)
    print(subjects)
    num_img = 0
    for subject in subjects:
        subject_path = os.path.join(db_path, subject)
        num_subject_files = len([fname for fname in os.listdir(subject_path) if os.path.isfile(os.path.join(subject_path, fname))])
        num_img += num_subject_files
        print("Subject: " + subject + " Files: " + str(num_subject_files))
    print(num_img)
