class Metrics:
    def misclassifications(self,y_pred,y_test):
        count = 0
        for i in range(len(y_pred)):
            if(y_pred[i]!=y_test.iloc[i]):
                count+=1
        return count