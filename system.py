import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

class clasificacion:
    def __init__(self):
        pass
    def load_data(self):
        path = "C:/Users/mayum/OneDrive/Email attachments/Documentos/Universidad/Octavo semestre/Python/Pinelines/"
        dataset = pd.read_csv(path + "iris_dataset.csv",sep=";", decimal = ",")
        prueba = pd.read_csv(path + "iris_prueba.csv",sep=";", decimal = ",")
        covariables = [ x for x in dataset.columns if x not in ["y"]]

        x= dataset.get(covariables)
        y= dataset["y"]

        x_nuevo = prueba.get(covariables)
        y_nuevo = prueba ["y"]
        return x, y , x_nuevo, y_nuevo
    
    def preprocessing_z(self,x): # funcion para estandarizar
        z= preprocessing.StandardScaler()
        z.fit(x)
        x_z = z.transform(x)
        return z, x_z
    
    def trainning_model(self, x,y):
        x,y ,x_nuevo, y_nuevo = self.load_data()
        x_train, x_test , y_train , y_test = train_test_split ( x,y,test_size = 0.5)
        z_1, x_train_z = self.preprocessing_z(x_train)
        x_test_z = z_1.transform(x_test)

        modelo1 = LogisticRegression(random_state =123)
        parametros = {'C': np.arange(0.1,5.1,0.1)}
        grilla1 = GridSearchCV( estimator = modelo1 , param_grid = parametros ,  scoring = make_scorer(accuracy_score), cv = 5 , n_jobs = -1)
        grilla1.fit(x_train_z, y_train)
        y_hat_test = grilla1.predict(x_test_z)

        z_2, x_train_z = self.preprocessing_z(x_test)
        x_train_z = z_2.transform(x_train)

        modelo2 = LogisticRegression(random_state =123)
        grilla2 = GridSearchCV( estimator = modelo2 , param_grid = parametros ,  scoring = make_scorer(accuracy_score), cv = 5 , n_jobs = -1)
        grilla2.fit(x_test_z, y_test)
        y_hat_train = grilla2.predict(x_train_z)

        z,x_z = self.preprocessing_z(x)
        x_nuevo_z= z.transform(x_nuevo)
        u1 = accuracy_score( y_test, y_hat_test)
        u2 = accuracy_score( y_train, y_hat_train)
    
        if np.abs (u1 - u2)<10:
            modelo_completo = LogisticRegression(random_state =123)
            grilla_completa = GridSearchCV(estimator = modelo_completo, param_grid = parametros,
            scoring = make_scorer(accuracy_score), cv = 5, n_jobs = -1)
            grilla_completa.fit(x_z,y)
        else:
            grilla_completa = LogisticRegression( random_state = 123)
            grilla_completa.fit(x_z,y)
        y_hat_nuevo = grilla_completa.predict(x_nuevo_z)
        metrica= accuracy_score(y_nuevo,y_hat_nuevo)
    
        return metrica
    def modeloClasificacion(self):
        try:
            x, y, x_nuevo, y_nuevo = self.load_data()
            metrica = self.trainning_model(x,y)

            return {"succes":True,"Precision": metrica}
        except Exception as e:
            return {"succes":False,"Error":str(e)}