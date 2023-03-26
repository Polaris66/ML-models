class Metrics:
    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test

    def accuracy(self):
        accuracy = 0
        for i in range(self.y_pred.shape[0]):
            if(self.y_pred[i] == self.y_test[i]):
                accuracy += 1
        return accuracy/self.y_pred.shape[0]
    
    def precision(self):
        precision = 0
        count = 0
        for i in range(self.y_pred.shape[0]):
            if self.y_pred[i]>0:
                if self.y_test[i]>0:
                    precision+=1
                count+=1
        return precision/count
    
    def recall(self):
        recall = 0
        count = 0
        for i in range(self.y_pred.shape[0]):
            if self.y_test[i]>0:
                if self.y_pred[i]>0:
                    recall+=1
                    count+=1
                else:
                    count+=1
        return recall/count