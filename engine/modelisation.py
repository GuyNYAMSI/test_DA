import pandas as pd
import numpy as np
#% matplotlib inline
import seaborn as sns
import pickle as pk
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import  Model
from sklearn. preprocessing import  OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
from h2o.automl import H2OAutoML
import h2o
from sklearn.linear_model import GammaRegressor, TweedieRegressor
from sklearn.metrics import  mean_squared_error

from linkquarto.link_color import Mise_enforme, cramers_v


couleur = Mise_enforme()
pallete = [couleur.lkp_green,couleur.lkp_blue,couleur.lkp_magenta,couleur.lkp_comp_blue,couleur.lkp_grey, couleur.lkp_purple]
pal = sns.color_palette(pallete, len(pallete))
SEED = 1759

class Modelisation:
    """
    Dans cette classe, nous définissons les différents modèles à utiliser
    """
    def __init__(self, chemin_base):
        self.chemin_base = chemin_base
        self.y_train = None
        self.y_test =None
        self.x_train = None
        self.x_test = None
        self.offset_train = None
        self.offset_test = None
        self.chemin_modele = "modeles/"
    
    def preprocessing(self):
        df  = pd.read_csv(self.chemin_base, sep=";")
        
        # classe d'age avec un pas de 10 
        df['AgeGroup'] = df.Age //10 
        df['AgeGroup'] = df['AgeGroup'].astype(str)
        df = df.drop(["Age"], axis=1)

        
        # variables d'intérêt
        df_y = df[["ClaimInflate", "InitialClaimInflate"]]

        # Standardisation des variables numériques
        col_num = df.columns[df.dtypes!="object"].to_list()
        col_num.remove("InitialClaimInflate")
        col_num.remove("ClaimInflate")
        df_num = df[col_num]
        scaler = StandardScaler()
        df_num[df_num.columns] = pd.DataFrame(scaler.fit_transform(df_num), index=df_num.index)
        
        # Sélection des variables catégorielles et numériques
        col_cat = df.columns[df.dtypes=="object"].to_list()
        df_cat = df[col_cat]

        # Encodage
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop="first")
        encoded_data = ohe.fit_transform(df_cat)
        df_cat = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(df_cat.columns.to_list()))
        
        #jointure et splitage en train-test
        df_std = pd.concat([df_num, df_cat, df_y], axis=1)
        y = df_std.ClaimInflate
        X  = df_std.drop(["ClaimInflate"], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X,
                                                                                y, 
                                                                                test_size=0.2, 
                                                                                random_state=SEED)

    def model_glm(self, famille="G"):
        if famille=="T":
            glm = TweedieRegressor() 
            glm.fit(self.x_train, self.y_train)
            y_pred = glm.predict(self.x_test)
        else:
            glm = GammaRegressor() 
            glm.fit(self.x_train.drop(["InitialClaimInflate"], axis=1), self.y_train)
            y_pred = glm.predict(self.x_test.drop(["InitialClaimInflate"], axis=1))
                             
        # Sauvegarde
        with open(os.path.join(self.chemin_modele, "model_glm_{}.pkl".format(famille)), 'wb') as file:
            pk.dump(glm, file)
        print("Sauvegarde du modèle : {}.pkl".format(glm))
        return  mean_squared_error(self.y_test, y_pred, squared=False)     
    
    def model_lgbm(self):

        model_lgb = LGBMRegressor(objective='regression', metric='rmse')
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [ 200, 300, 500],
            'min_child_samples': [70, 80, 100]
        }      
        grid_search = GridSearchCV(estimator=model_lgb, 
                                   param_grid=param_grid, 
                                   cv=5,
                                   scoring='neg_mean_squared_error', 
                                   verbose=10)
        grid_search.fit(self.x_train, 
                        self.y_train, 
                        eval_set=[(self.x_test, self.y_test)], 
                        eval_metric='rmse')
        
        best_model_lgb = grid_search.best_estimator_
        preds = best_model_lgb.predict(self.x_test, axis=1)

        # Sauvegarde
        with open(os.path.join(self.chemin_modele, "best_model_lgb.pkl"), 'wb') as file:
            pk.dump(model_lgb, file)
        print("Sauvegarde du modèle : {}.pkl".format(best_model_lgb))
        return mean_squared_error(self.y_test, preds, squared=False)

    def model_automl(self):
        
        h2o.init()
        xtrain_ = self.x_train
        xtrain_["y"] = self.y_train
        xtrain = h2o.H2OFrame(xtrain_)

        auto_ml = H2OAutoML(max_models=10, seed=SEED, max_runtime_secs=500,stopping_metric='rmse',preprocessing = ["target_encoding"])
        auto_ml.train(x=self.x_train.columns.to_list(), y="y", training_frame = xtrain)
        print(auto_ml.leaderboard)
        auto_ml.explain(xtrain,include_explanations=['varimp','shap_summary'])
        h2o.remove(xtrain)
        preds = auto_ml.predict(h2o.H2OFrame(self.x_test)).as_data_frame().values.flatten()
        # Sauvegarde
        #h2o.download_model(auto_ml.leader, path=self.chemin_modele)
        
        return np.sqrt(mean_squared_error(self.y_test, preds))
     
    def model_rn(self):

        # Création du modèle
        inputs = Input(shape = (self.x_train.shape[1],), name = "Input")

        dense1 = Dense(units = 50, activation = "leaky_relu", name = "Dense_1")
        dense2 = Dense(units = 80, activation = "leaky_relu", name = "Dense_2")
        dense3 = Dense(units = 30, activation = "leaky_relu", name = "Dense_3")
        dense4 = Dense(units = 1, name = "Dense_4")
        x=dense1(inputs)
        x=dense2(x)
        x=dense3(x)
        outputs=dense4(x)
        model_rn = Model(inputs = inputs, outputs = outputs)
        model_rn.summary()

        model_rn.compile(loss = "mse",
                      optimizer = "adam",
                      metrics = ["mse"])

        model_rn.fit(self.x_train,self.y_train,epochs=300, batch_size=32,validation_split=0.2)
        

        preds = model_rn.predict(self.x_test)
        model_rn.save(os.path.join(self.chemin_modele, "model_rn.h5"))
        return np.sqrt(mean_squared_error(self.y_test, preds))
         

def main(path_base):
    
    Modele = Modelisation(path_base)
    Modele.preprocessing()
    rmse = {"rmse modèle gamma ": Modele.model_glm(),
            "rmse modèle tweedie ": Modele.model_glm("T"),
            "rmse modèle lgb ": Modele.model_lgbm(),
            "rmse modèle rn": Modele.model_rn()
            #"rmse modèle AutoML": Modele.model_automl()
            }
    with open(os.path.join(Modele.chemin_modele, "rmse.pkl"), 'wb') as file:
        pk.dump(rmse, file)
        print("Sauvegarde des métriques : {}".format(rmse))
    
        
if __name__ == '__main__':
 # Vérifier l'usage 
  
    if len(sys.argv) < 2: 
        print("Usage : ./modelisation.py path/df_cleans") 
        exit(1)
        
    # Définir le chemin vers la base de données 
    path_base = sys.argv[1] 

    main(path_base)
    
