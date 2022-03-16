def wasCorrect(pred, Y, problemtype):
    if(problemtype == 0): #predict what something is, like point cloud
        return (pred.argmax(dim=1) == Y).sum().item()
    elif(problemtype == 1): #find a absolute value, like max
        #i call it correct if within 10% of max, would be better if it was correct if it was bigger then second biggest, but that would be slower :(
        return (0.9*Y < pred).sum().item()

def getProblemType(dataset):
    if(dataset == 'maximum'):
        return 1
    else: return 0
