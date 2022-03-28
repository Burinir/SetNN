import Datasets.modelnet40_pointcloud
import Datasets.maximum
import Datasets.MNIST
import Datasets.cardinality
import Datasets.clustering
import Datasets.mode



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
    elif dataset == 'max4':
        return Datasets.maximum.Max4Data()
    elif dataset == 'minmax' or dataset == 'min2max2':
        return Datasets.maximum.Min2Max2Data()
    elif dataset == 'mnist_sum' or dataset =='MNIST_sum' or dataset == 'mnistsum' or dataset == 'MNISTsum':
        return Datasets.MNIST.MNIST_sum_data()
    elif dataset == 'cardinality':
        return Datasets.cardinality.CardinalityData()
    elif dataset == 'eq2':
        return Datasets.clustering.Eq2Data()
    elif dataset == 'mode':
        return Datasets.mode.ModeData()
    else:
        raise ValueError('Invalid net {}'.format(dataset))
