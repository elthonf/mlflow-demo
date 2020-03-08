import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import joblib
# from sklearn.metrics import accuracy_score


import random

def get_a_funnyName():
    f = open("../../../datasets/names/funnynames.txt", "r")
    nomes = f.readlines()
    return random.choice(nomes).rstrip('\n')


if __name__ == "__main__":
    # Carrega os dados
    mydf = pd.read_csv('../../../datasets/statistical/BaseDefault01.csv')

    # Identifica no dataset as variáveis independentes e a variavel alvo
    targetcol = 'default'
    y = mydf[targetcol]

    experimento = mlflow.get_experiment_by_name('Risco de Credito')
    if experimento is None:
        experimento = mlflow.create_experiment('Risco de Credito')

    # Cria o Classifier (modelo 1 com todas as colunas e parâmetros padrão do Sckit Learn)
    independentcols = ['renda', 'idade', 'etnia', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    x = mydf[independentcols]
    run_name = get_a_funnyName()
    with mlflow.start_run(experiment_id=experimento.experiment_id, run_name=("MyNameIs.. " + run_name) ):
        from sklearn.ensemble import RandomForestClassifier as rfc

        clf = rfc()
        clf.fit(X=x, y=y)

        clf.independentcols = independentcols
        clf_acuracia = clf.score(X=x, y=y)
        print("Modelo 01 (classificador), criado com acurácia de: [{0}]".format(clf_acuracia))

        mlflow.log_param("criterion", clf.criterion)
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("min_samples_leaf", clf.min_samples_leaf)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("Inputs", independentcols)

        mlflow.log_metric("acuracia", clf_acuracia )
        mlflow.log_metric("minha métrica customizada", 10 )

        mlflow.sklearn.log_model(clf, "modelo_mlf")

        pass

    # Cria modelo 2, classificador com alguns parâmetros aleatórios!
    independentcols = ['renda', 'idade', 'etnia', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    x = mydf[independentcols]
    run_name = get_a_funnyName()
    with mlflow.start_run(experiment_id=experimento.experiment_id, run_name=("MyNameIs.. " + run_name)):
        from sklearn.ensemble import RandomForestClassifier as rfc

        clf = rfc(criterion=random.choice(["gini", "entropy"]),
                  n_estimators=random.randint(50, 150),
                  min_samples_leaf=random.randint(1, 3),
                  max_depth=random.choice([None, random.randint(3, 100)]))
        clf.fit(X=x, y=y)

        clf.independentcols = independentcols
        clf_acuracia = clf.score(X=x, y=y)
        print("Modelo 02 (classificador), criado com acurácia de: [{0}]".format(clf_acuracia))

        mlflow.log_param("criterion", clf.criterion)
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("min_samples_leaf", clf.min_samples_leaf)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("Inputs", independentcols)

        mlflow.log_metric("acuracia", clf_acuracia)
        mlflow.log_metric("minha métrica customizada", 10)

        mlflow.sklearn.log_model(clf, "modelo_mlf")

        pass

    pass