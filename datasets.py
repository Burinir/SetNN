import Datasets.modelnet40_pointcloud
import Datasets.maximum



def getdata(dataset):
    if dataset == 'pointcloud100':
        return Datasets.modelnet40_pointcloud.PointCloudData('Datasets\\ModelNet40_cloud.h5',
                            batch_size=64, down_sample=100, do_standardize=True, do_augmentation = False)
    elif dataset == 'pointcloud1000':
        return Datasets.modelnet40_pointcloud.PointCloudData('Datasets\\ModelNet40_cloud.h5',
                            batch_size=64, down_sample=10, do_standardize=True, do_augmentation = False)
    elif dataset == 'pointcloud5000':
        return Datasets.modelnet40_pointcloud.PointCloudData('Datasets\\ModelNet40_cloud.h5',
                            batch_size=64, down_sample=2, do_standardize=True, do_augmentation = False)
    elif dataset == 'maximum':
        return Datasets.maximum.MaximumData()
    else:
        raise ValueError('Invalid net {}'.format(dataset))
