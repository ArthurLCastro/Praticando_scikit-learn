import pandas as pd
from sklearn.tree import DecisionTreeRegressor


melbourne_file_path = './curso_kaggle_intro_to_ml/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

y = melbourne_data.Price

# Escolhendo as 'Features' (as 'entradas' do modelo):
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Definindo o modelo:
melbourne_model = DecisionTreeRegressor()

# Treinando o modelo:
melbourne_model.fit(X, y)

# Realizando previsoes
previsao = melbourne_model.predict(X.head())

# Comparando valores
print(f'Preços reais:\n{y.head()}\n\n')
print(f'Preços previstos:\n{previsao}')
