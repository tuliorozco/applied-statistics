

## Conclusiones

En este archivo se relacionarán las conclusiones obtenidas a partir del ejercicio de modelo logístico de clasificación y de regresión Ridge y Lasso.

El mejor modelo de regresión para estimar el precio de acuerdo a las métricas encontradas es KKN de REGRESIÓN, ya que presenta un R2 de 80.1% en comparación con el modelo *Rigde* y *Lasso* que presentaron un R2 de 74.2%. Además, se muestra que gracias a la métrica del RMSE es menor en KNN de regresión (5258) en comparación con el RMSE de *Rigde* y *Lasso* (5979). Por lo que el mejor modelo que puede estimar mejor el valor de precio es KNN de REGRESIÓN
 
Para el mejor modelo de clasificación que permite estimar, si un vehículo es de alta o baja de manda, es el KNN de CLASIFICACIÓN. Puesto que con un knn_neighbors = 15 obtuvo un ROC AUC de de 0.963 y una *accuracy* de 0.895, en comparación con Clasificación para Ridge y Lasso que obtuvieron 0.957 y 0.891 respectivamente. También, en logra presentar un buen balance entre precisión y recall. Aunque ambos modelos son muy efectivos, KNN Clasificación ofrece una ligera ventaja en capacidad de discriminación, por lo que se recomienda como el mejor modelo para predecir alta demanda.