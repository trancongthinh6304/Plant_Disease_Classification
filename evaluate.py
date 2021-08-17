from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, X_test, y_test, class_name):
    y_pred = model.predict(X_test)
    print(classification_report(y_test.argmax(axis=1), 
          y_pred.argmax(axis=1), 
          target_names = class_name))
    print(confusion_matrix(y_test.argmax(axis=1), 
          y_pred.argmax(axis=1)))