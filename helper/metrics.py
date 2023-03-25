class Metrics:
    def misclassifications(self, y_pred, y_test):
        count = 0
        for i in range(y_pred.shape[0]):
            if(y_pred[i] != y_test[i]):
                count += 1
        return count
