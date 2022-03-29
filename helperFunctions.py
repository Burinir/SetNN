import torch
#returns number of elements that where "correct"
def wasCorrect(pred, Y, problemtype):
    if(problemtype == 0): #predict what something is, like point cloud
        return (pred.argmax(dim=1) == Y).sum().item()
    elif(problemtype == 1): #find a absolute value, like max
        #i call it correct if within 5 of max, would be better if it was correct if it was bigger then second biggest, but that would be slower :(
        return (torch.abs(Y-pred)<5).sum().item()
    elif(problemtype == 2):
        return (torch.abs(Y-pred)<1).sum()/4
    elif problemtype == 3: #mnist MNIST_sum
        return (torch.round(pred)==Y).sum().item()
    elif(problemtype == 4): #mode
        #correct if correctly rounded
        return (torch.abs(Y-pred)<0.5).sum().item()

def getProblemType(dataset):
    if dataset == 'maximum' or dataset == 'cardinality' or dataset == 'sum' or dataset == 'mean':
        return 1
    elif dataset=='max4' or dataset =='minmax':
        return 2
    elif dataset == 'mnist_sum' or dataset =='MNIST_sum' or dataset == 'mnistsum' or dataset == 'MNISTsum':
        return 3
    elif dataset=='mode':
        return 4
    else: return 0

def getError(pred, Y, problemtype):
    if problemtype == 0:
        raise ValueError("A Error does not make sence for a classifier")
    elif(problemtype == 1 or problemtype == 4): #find a absolute value, like max
        return torch.abs(pred-Y).sum().item()
    elif(problemtype == 2):
        return  torch.abs(pred-Y).sum()/4
    elif problemtype == 3: #mnist MNIST_sum
        return torch.abs(pred-Y).sum().item()

#returns number of elements that where "correct", with higher "correctness expectations"
# def wasCorrectTest(pred, Y, problemtype):
#     if(problemtype == 0): #predict what something is, like point cloud
#         return (pred.argmax(dim=1) == Y).sum().item()
#     elif(problemtype == 1): #find a absolute value, like max
#         #i call it correct if within 5 of max, would be better if it was correct if it was bigger then second biggest, but that would be slower :(
#         return (torch.abs(Y-pred)<5).sum().item()
#     elif(problemtype == 2):
#         sum = torch.logical_and(0.9*Y<pred, 1.1*Yc>pred).sum()
#         return sum/4
#     elif problemtype == 3: #mnist MNIST_sum
#         return (torch.round(pred)==Y).sum().item()
