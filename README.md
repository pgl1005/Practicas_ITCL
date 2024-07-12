
# Aprendizaje en ITCL <img src="https://github.com/user-attachments/assets/408526ee-cde2-4843-a1e5-3b29865b822a" align="right" width="250">

Dos carpetas principales existentes en este repositorio se centran en el aprendizaje de varios modelos de aprendizaje automático y la implementación de Federated Learning (FL) utilizando una variedad de técnicas de agregación, como el bagging.
El repositorio de aprendizaje federado tiene una estructura que contiene scripts de implementación de aprendizaje federado que utilizan una variedad de modelos, como redes neuronales, redes neuronales convolucionales (CNN) y redes neuronales recurrentes (LSTM). Además, para integrar los modelos locales en un modelo global, se utilizan técnicas de aprendizaje federado y bagging.

# Estructura del repositorio
El repositorio contiene 2 carpetas _FederatedLearning_ y _ScriptsDeFormacion_. La carpeta de FederatedLearning contiene scripts de implementación de aprendizaje federado que utilizan una variedad de modelos, como redes neuronales, redes neuronales convolucionales (CNN), redes neuronales recurrentes (LSTM) o árboles de decisión. Estos scripts servirán posteriormente para el proyecto Europeo-AI4Hope descrito en la memoria de prácticas.
La carpeta ScriptsDeFormacion contiene ejemplos de modelos de aprendizaje de árboles de decisión, redes neuronales, CNN y LSTM. Estos ejemplos se realizaron durante la primera semana de prácticas, los cuales ayudaron a la posterior implementación con Federated Learning.

# Contenido de las carpetas 
### Federated Learning
Los script en esta carpeta muestran cómo implementar Federated Learning con varios modelos. Sus archivos principales y sus descripciones son:

- ArbolDecision_Pytorch.ipynb: utiliza los árboles de decisión para implementar el aprendizaje federado a través de la librería Pytorch.
- ArbolDeDecisionSklearn.ipynb: utiliza los árboles de decisión para implementar el aprendizaje federado a través de la librería Sklearn.
- ArbolDeDecisionTensorFlow.ipynb: utiliza los árboles de decisión para implementar el aprendizaje federado a través de la librería TensorFlow.
- ModeloRedNeuronal.ipynb: utiliza redes neuronales simples para implementar el aprendizaje federado.
- RedesRecurrentesLSTM.ipynb: utiliza redes neuronales recurrentes para implementar el aprendizaje federado.
- CNN.ipynb: utiliza redes neuronales convolucionales para implementar el aprendizaje federado.

### Scripts de Formación
Esta carpeta contiene scripts de formación individuales para aprender sobre varios modelos de aprendizaje automático. Sus archivos principales y sus descripciones son:

- ArbolDeDecision.ipynb: muestra cómo construir, entrenar y evaluar un modelo de árbol de decisión. Describe los conceptos fundamentales de los árboles de decisión, la importancia de las características y cómo interpretar los resultados se explican.
- RedesNeuronales.ipynb: ofrece instrucciones paso a paso sobre cómo crear, entrenar y evaluar una red neuronal. Incluye ejemplos de configuración de capas, función de activación y procedimiento de entrenamiento y evaluación del modelo.
- RegresionLinealSimple.ipynb: proporciona instrucciones sobre cómo usar una regresión lineal simple. Se abordan los conceptos básicos de la regresión lineal, como el ajuste de una línea a los datos, la interpretación de los coeficientes y el rendimiento del modelo.
- RegresionLinealMultiple.ipynb: Se describe la importancia de las diferentes características, la interpretación de los coeficientes y la evaluación del rendimiento del modelo.
- RegresionPolinomica.ipynb: proporciona instrucciones sobre cómo usar la regresión polinómica para encontrar relaciones no lineales en los datos. Se proporciona una explicación de cómo modificar las características, ajustar un modelo polinómico y evaluar su desempeño en comparación con otros tipos de regresión.
