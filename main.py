from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("datos_transporte.csv")

# Seleccionar características relevantes
X = df[['pasajeros', 'tiempo_espera']]

# Aplicar clustering
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

# Visualizar resultados
plt.scatter(df['pasajeros'], df['tiempo_espera'], c=df['cluster'])
plt.xlabel("Pasajeros")
plt.ylabel("Tiempo de espera")
plt.title("Clustering de estaciones")
plt.show()
