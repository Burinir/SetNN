import os
import glob
import trimesh
import numpy as np
import h5py
DATA_DIR = os.path.join(os.getcwd(), "Datasets\ModelNet40\\")

#from https://keras.io/examples/vision/pointnet/:
#returns np arrays of train and test data
def parse_dataset(num_points=10000):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[q,w,e,r,t,z,u,i,o,p,a,s,d,f,g,h,j,k,l,y,x,c,v,b,n,m]*"))
    print(len(folders)) #should be 40

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("\\")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(10000) #sample 10000 points from the cad files
print("dataset has been parsed")
print(CLASS_MAP)

# {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone',
# 10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box', 17: 'guitar', 18: 'keyboard', 19: 'lamp',
# 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand', 24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink',
# 30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}

with h5py.File('ModelNet40_cloud.h5', 'w' ) as h5f:
    h5f.create_dataset('tr_cloud', data=train_points)
    h5f.create_dataset('tr_label', data=train_labels)
    h5f.create_dataset('test_cloud', data=test_points)
    h5f.create_dataset('test_label', data=test_labels)
    h5f.close()
