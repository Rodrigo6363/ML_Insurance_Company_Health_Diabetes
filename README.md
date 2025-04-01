# 🩺 Diabetes Prediction Project (2017–2023)

This project explores and evaluates different machine learning models to predict the likelihood of diabetes in individuals, using health-related data from **2017 to 2023**. The goal is to build an accurate and interpretable pipeline that can be applied to real-world clinical screening contexts.

---

## 📂 Project Structure

- **Data**: Preprocessed clinical and sociodemographic data.
- **Notebook**: Final pipeline including EDA, model training, evaluation, and comparison.
- **Figures**: All relevant visualizations and plots used for analysis and reporting.
- **Models**: Neural Network and best ML model saved for inference.
- **Utils**: Scripts used for visualizations and evaluations, used as a toolbox 
---

## 🧼 Data Preprocessing

The dataset includes both numerical and categorical features such as:
- Age, Weight, Height, Cholesterol levels, HbA1c, Blood pressure.
- Socio-demographics: Gender, Race, Country of birth, Income group.
- Feature engineering: BMI, CHOL ratio, Risk indices.

Steps:
- Missing value handling.
- Feature scaling with `StandardScaler`.
- Feature categorization (e.g., BMI categories, age groups).
- Train/test split.

---

## 📊 Exploratory Data Analysis

### 🎯 Target Distribution

![Target Distribution](src/img/target_distribution.png)

### 🔢 Numeric Features Distribution by Diabetes

![Numeric Distribution](src/img/bivariant.png)

### 🧮 Categorical Features Distribution by Target

![Categorical Distribution (Colored)](src/img/bivariant_cat.png)

### 🔣 Raw Categorical Feature Frequencies

![Categorical Frequencies](src/img/cat_dist.png)

### 🔗 Correlation Matrix

![Correlation Matrix](src/img/correlation_matrix.png)

---

## 🤖 Model Training & Evaluation

### 🧠 Neural Network (Keras)

A feedforward neural network with:
- Dense layers + BatchNormalization + Dropout.
- Loss: Binary Crossentropy.
- Metrics: Accuracy, AUC, Precision, Recall.

> Achieved high **recall (0.89)** but with lower **precision (0.40)**.

![Confusion Matrix NN](src/img/matrix_nn_pred.png)

---

### 📦 Machine Learning Models

Multiple models were trained and evaluated using cross-validation, including:
- Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, etc.

### 📈 Model Comparison (by Mean Recall)

![Model Comparison](src/img/recall.png)

### ✅ Best Performing Model: Gradient Boosting

- Best trade-off between accuracy, precision, and recall.
- Final evaluation:

![Confusion Matrix ML](src/img/matrix_ml_pred.png)

---

## 📌 Conclusions

- The **neural network** achieved **very high sensitivity**, ideal for screening use cases where missing a positive case is costly.
- The **machine learning models** (particularly Gradient Boosting and LightGBM) offered **more balanced performance**, with better precision and F1-score.
- Depending on the clinical goal (screening vs diagnosis), either model can be deployed.

---
## 📁 Files Included
- `data/`
  - `Prep_data/`: All the data to predict, clean and structured by categories.
  - `raw_data/`: All data scraped directly from the web, divided by categories.
  - `Train_data/`: All the data to train, clean and structured by categories.
  - `data_sample.csv`: Sample to show the general structure of the data.
- `img/`: All graphs and visualizations used.
- `models/`
  - `diabetes_nn_model.h5`: Trained Keras model.
  - `best_lightgbm_model.pkl`: Best trained ML model.
-`notebooks`: All notebooks used in the analisys and prediction of this data.
- `result_notebook/`
  - `Final_notebook.ipynb`: Final model pipeline, from preprocessing to evaluation.
- `utils`: Scripts used for visualizations and evaluations, used as a toolbox
- `requirements.txt`: List of packages used during the project


# Proyecto de Predicción de Diabetes (2017–2023)

Este proyecto explora y evalúa distintos modelos de aprendizaje automático para predecir la probabilidad de padecer diabetes, utilizando datos de salud recopilados entre **2017 y 2023**. El objetivo es construir una pipeline precisa e interpretable que pueda aplicarse en contextos clínicos reales de cribado.

---

## 📂 Estructura del Proyecto

- **Data**: Datos clínicos y sociodemográficos preprocesados.
- **Notebook**: Pipeline final con análisis exploratorio, entrenamiento, evaluación y comparación de modelos.
- **Figures**: Todas las visualizaciones y gráficos utilizados en el análisis y los informes.
- **Models**: Red neuronal y mejor modelo ML guardados para inferencia.
- **Utils**: Scripts usados para visualización y evaluación, organizados como una toolbox.
---

## 🧼 Preprocesamiento de Datos

El conjunto de datos incluye variables numéricas y categóricas como:
- Edad, peso, altura, niveles de colesterol, HbA1c, presión arterial.
- Sociodemográficos: género, raza, país de nacimiento, grupo de ingresos.
- Ingeniería de variables: IMC, ratio de colesterol, índices de riesgo.

Pasos:
- Tratamiento de valores faltantes.
- Escalado de características con `StandardScaler`.
- Categorización de variables (e.g., categorías de IMC, grupos de edad).
- División train/test.

---

## 📊 Análisis Exploratorio de Datos

### 🎯 Distribución del Target

![Distribución del Target](src/img/target_distribution.png)

### 🔢 Distribución de Variables Numéricas según Diabetes

![Distribución Numérica](src/img/bivariant.png)

### 🧮 Distribución de Variables Categóricas según Target

![Distribución Categórica](src/img/bivariant_cat.png)

### 🔣 Frecuencia de Variables Categóricas (sin procesar)

![Frecuencia Categórica](src/img/cat_dist.png)

### 🔗 Matriz de Correlación

![Matriz de Correlación](src/img/correlation_matrix.png)

---

## 🤖 Entrenamiento y Evaluación de Modelos

### 🧠 Red Neuronal (Keras)

Red neuronal densa con:
- Capas `Dense` + `BatchNormalization` + `Dropout`.
- Función de pérdida: Binary Crossentropy.
- Métricas: Accuracy, AUC, Precision, Recall.

> Se alcanzó un **recall alto (0.89)** pero una **precisión baja (0.40)**.

![Matriz de Confusión NN](src/img/matrix_nn_pred.png)

---

### 📦 Modelos de Aprendizaje Automático

Se entrenaron y evaluaron varios modelos con validación cruzada, incluyendo:
- Regresión logística, Random Forest, XGBoost, LightGBM, CatBoost, etc.

### 📈 Comparación de Modelos (por Recall medio)

![Comparación de Modelos](src/img/recall.png)

### ✅ Modelo con Mejor Desempeño: Gradient Boosting

- Mejor equilibrio entre precisión, recall y exactitud.
- Evaluación final:

![Matriz de Confusión ML](src/img/matrix_ml_pred.png)

---

## 📌 Conclusiones

- La **red neuronal** logró **muy alta sensibilidad**, ideal para escenarios de cribado donde es crucial no omitir casos positivos.
- Los **modelos de ML** (especialmente Gradient Boosting y LightGBM) ofrecieron un rendimiento más equilibrado, con mejor precisión y F1-score.
- Dependiendo del objetivo clínico (cribado vs diagnóstico), se puede optar por uno u otro modelo.

---

## 📁 Archivos Incluidos

- `data/`
  - `Prep_data/`: Todos los datos para la predicción, limpios y estructurados por categoría.
  - `raw_data/`: Datos sin procesar descargados desde la web, por categoría.
  - `Train_data/`: Datos para entrenamiento, limpios y estructurados.
  - `data_sample.csv`: Muestra representativa de la estructura de los datos.
- `img/`: Todas las gráficas y visualizaciones utilizadas.
- `models/`
  - `diabetes_nn_model.h5`: Modelo Keras entrenado.
  - `best_lightgbm_model.pkl`: Mejor modelo de ML entrenado.
- `notebooks`: Notebooks usados en el análisis y predicción.
- `result_notebook/`
  - `Final_notebook.ipynb`: Pipeline final, desde el preprocesamiento hasta la evaluación.
- `utils`: Scripts de evaluación y visualización usados como toolbox.
- `requirements.txt`: Lista de paquetes usados en el proyecto.
