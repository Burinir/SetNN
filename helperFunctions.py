import torch
#returns number of elements that where "correct"
def wasCorrect(pred, Y, problemtype):
    if(problemtype == 0): #predict what something is, like point cloud
        return (pred.argmax(dim=1) == Y).sum().item()
    elif(problemtype == 1): #find a absolute value, like max
        #i call it correct if within 10% of max, would be better if it was correct if it was bigger then second biggest, but that would be slower :(
        return (0.9*Y < pred).sum().item()
    elif(problemtype == 2):
        #print(Y.shape, pred.shape)
        sum = (0.9*Y.reshape((64,4))<pred).sum()
        return sum/4

def getProblemType(dataset):
    if(dataset == 'maximum'):
        return 1
    elif(dataset=='max4' or dataset =='minmax'):
        return 2
    else: return 0


#returns number of elements that where "correct", with higher "correctness expectations"
def wasCorrectTest(pred, Y, problemtype):
    if(problemtype == 0): #predict what something is, like point cloud
        return (pred.argmax(dim=1) == Y).sum().item()
    elif(problemtype == 1): #find a absolute value, like max
        #i call it correct if within 10% of max, would be better if it was correct if it was bigger then second biggest, but that would be slower :(
        return torch.logical_and(0.9*Y < pred, 1.1*Y>pred).sum().item()
    elif(problemtype == 2):
        Yc=Y.reshape((64,4))
        sum = torch.logical_and(0.9*Yc<pred, 1.1*Yc>pred).sum()
        return sum/4
