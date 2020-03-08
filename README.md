# Repositório mlflow-demo
Demonstração de utilização do ML Flow.

Pacote criado em **Python 3.7.6**


## Instalação dos pacotes necessários:



```
pip install -r requirements.txt
```


Se tiver problemas com SSL, usar esta comando:
``` 
pip install --trusted-host pypi.org --trusted-host  files.pythonhosted.org -r requirements.txt
```


## Treino de modelos

Estar no diretório `training`.

Executar no python o arquivo `treinamlf.py`



## Consulta ML Flow com interface gráfica:

Estar no diretório `training` e executar:

```
mlflow ui --port 5000 --backend-store-uri localdevelop/statmlflow/training/mlruns/
```

## Serving dos modelos

Para servir o modelo, rodar (preferencialmente, escolha a porta 1234 pois o client está conectando nesta porta):
``` 
mlflow models serve -m <PATH-DO_MODELO> -p <PORTA>
```

Exemplo (substitua o 2c3fa136d6cf485bbcd3209a689d9a21):
```
mlflow models serve -m localdevelop/statmlflow/training/mlruns/1/2c3fa136d6cf485bbcd3209a689d9a21/artifacts/modelo_mlf/ -p 1234
```

## Um exemplo de cliente para um modelo que estiver rodando:

Estar no diretório "training".