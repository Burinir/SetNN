# SetNN
This is a program to train and compare different Set Neural Networks on different datasets. 

## How to use:
### Arguments:

--dataset={maximum, max4, min2max2, cardinality, mode, eq2, pointcloud100, pointcloud1000, pointcloud5000}   
\--net={deepset, set_transformer, pointnet, repset}  
Note: pointnet only works on the point cloud variations  
\--num_epochs={For how many epochs you want to train the model}
\--test_freq={After how many epochs should the Network be tested}
\--pool={max, mean, sum, default}

### Example:  
```
python run.py --net=deepset --dataset=pointcloud100 --num_epochs=10
```

### How to set up point cloud:
The datasets called pointcloud{100,1000,5000} are based on the ModelNet40 dataset. You will need to download the models (http://modelnet.cs.princeton.edu/) and extract the zip file to Datasets\ModelNet40\. You can then run ``python Datasets\createPointCloud.py`` (takes a few minutes), which creates Datasets\ModelNet40_cloud.h5. After that you can delete the downloaded files again and use the dataset in the program.

## Datasets:
#### Maximum:
Return the maximum of a set of numbers
#### Max4:
Return the biggest four elements of a set of numbers
#### Min2max2:
Return the two smallest and the two biggest elements of a set of numbers
#### Pointcloud:
Object classification with 40 classes from a point cloud with 100, 1000, 5000 points, by editing the code, the point cloud can be perturbed as well}
#### Cardinality:
Return the number of elements of a set
#### Mode:
Return the value of the most common element of a multiset
#### Eq2:
Return 1 if the set is a multiset, 0 otherwise
