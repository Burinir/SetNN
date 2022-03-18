# SetNN
This is a program to train and compare different Set Neural Networks on different datasets

## use:
### Arguments: 
 
--dataset={maximum, pointcloud100, pointcloud1000, pointcloud5000}   
\--net={deepset, set_transformer, pointnet}  
Note: pointnet only works on the pointcloud variations  
\--num_epochs={For how many epochs you want to train the model}

Example:  
```
python run.py --net=deepset --dataset=pointcloud100 --num_epochs=10
```
