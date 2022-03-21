# SetNN
This is a program to train and compare different Set Neural Networks on different datasets

## How to use:
### Arguments: 
 
--dataset={maximum, pointcloud100, pointcloud1000, pointcloud5000}   
\--net={deepset, set_transformer, pointnet}  
Note: pointnet only works on the point cloud variations  
\--num_epochs={For how many epochs you want to train the model}

### Example:  
```
python run.py --net=deepset --dataset=pointcloud100 --num_epochs=10
```

### How to set up point cloud:
The datasets called pointcloud{100,1000,5000} are based on the ModelNet40 dataset. You will need to download the models (http://modelnet.cs.princeton.edu/) and extract the zip file to Datasets\ModelNet40\. You can then run ``python Datasets\createPointCloud.py`` (takes a few minutes), which creates Datasets\ModelNet40_cloud.h5. After that you can delete the downloaded files again and use the dataset in the program.
