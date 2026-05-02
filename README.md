# Análisis y clasificación de contenido en Netflix usando Machine Learning

Proyecto de Ciencia de Datos orientado al análisis exploratorio, visualización y clasificación automática de contenido audiovisual disponible en Netflix.

El objetivo principal es analizar patrones dentro del catálogo de Netflix y construir modelos de Machine Learning capaces de clasificar si un título corresponde a una **película** o una **serie**, utilizando variables descriptivas del dataset.

---

## Objetivo del proyecto

Analizar el catálogo de Netflix desde una perspectiva exploratoria y predictiva, aplicando técnicas de limpieza, transformación, visualización de datos y modelos de clasificación supervisada.

### Objetivos específicos

- Explorar la distribución de películas y series dentro del catálogo.
- Analizar la evolución del contenido según año de estreno.
- Identificar países, géneros, clasificaciones, directores y actores con mayor presencia.
- Preparar variables para entrenamiento de modelos de Machine Learning.
- Entrenar y comparar modelos de clasificación.
- Evaluar el rendimiento mediante métricas como Accuracy, Precision, Recall y F1-score.
- Detectar y corregir posibles fugas de información en las variables predictoras.

---

## Dataset utilizado

Se utilizó un dataset público de Netflix disponible en Kaggle.

El dataset contiene información sobre títulos disponibles en Netflix, incluyendo películas y series, junto con variables descriptivas como:

- `show_id`
- `type`
- `title`
- `director`
- `cast`
- `country`
- `date_added`
- `release_year`
- `rating`
- `duration`
- `listed_in`
- `description`

---

## Herramientas y tecnologías

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Flujo de trabajo

El proyecto sigue un flujo completo de Ciencia de Datos:

1. Carga del dataset.
2. Revisión inicial de estructura y calidad de datos.
3. Limpieza y tratamiento de valores nulos.
4. Transformación de variables.
5. Análisis exploratorio de datos.
6. Visualización de patrones relevantes.
7. Preparación de variables para Machine Learning.
8. Entrenamiento de modelos de clasificación.
9. Evaluación y comparación de resultados.
10. Interpretación final y conclusiones.

---

## Análisis exploratorio

Durante el análisis exploratorio se revisaron distintos aspectos del catálogo de Netflix:

- Distribución y proporción de películas y series.
- Evolución del contenido por año de estreno.
- Países con mayor cantidad de títulos.
- Géneros más frecuentes.
- Clasificaciones más comunes.
- Duración de películas.
- Cantidad de temporadas en series.
- Directores con mayor presencia.
- Actores y actrices con mayor aparición.

Las visualizaciones fueron diseñadas con una estética inspirada en Netflix, utilizando una paleta basada en negro, blanco, gris y rojo.

---

## Visualizaciones generadas

### 1. Distribución de contenido por tipo
Permite observar la cantidad total de películas y series disponibles en el catálogo.

<p align="center">
  <img width="884" alt="Distribución de contenido por tipo" src="https://github.com/user-attachments/assets/db28ad3a-d49f-4979-8b57-468b4b2b471e" />
</p>

---

### 2. Proporción de películas y series
Muestra el peso porcentual de películas y series dentro del catálogo analizado.

<p align="center">
  <img width="535" alt="Proporción de películas y series" src="https://github.com/user-attachments/assets/bb528acd-42e2-4372-b60f-ebdf4167fc47" />
</p>

---

### 3. Evolución del contenido por año de estreno
Visualiza cómo ha variado la cantidad de títulos según su año de lanzamiento.

<p align="center">
  <img width="900" alt="Evolución del contenido por año de estreno" src="https://github.com/user-attachments/assets/b12c8a8f-4e5e-4834-98e5-66d82f39a7fb" />
</p>

---

### 4. Top países con más contenido
Identifica los países con mayor presencia dentro del catálogo de Netflix.

<p align="center">
  <img width="850" alt="Top países con más contenido" src="https://github.com/user-attachments/assets/abcb9ba6-f3b9-4a98-8590-92e75de9e36d" />
</p>

---

### 5. Clasificaciones más frecuentes
Permite revisar las clasificaciones de audiencia más comunes dentro del catálogo.

<p align="center">
  <img width="850" alt="Clasificaciones más frecuentes" src="https://github.com/user-attachments/assets/268cb78a-deb1-46f1-b6e6-059fba1cd9d8" />
</p>

---

### 6. Géneros más frecuentes
Muestra las categorías o géneros con mayor cantidad de títulos disponibles.

<p align="center">
  <img width="850" alt="Géneros más frecuentes" src="https://github.com/user-attachments/assets/422cac6c-ca4a-4941-a11a-bc9eac9635de" />
</p>

---

### 7. Distribución de duración de películas
Analiza la duración de las películas expresada en minutos.

<p align="center">
  <img width="850" alt="Distribución de duración de películas" src="https://github.com/user-attachments/assets/5fc9129a-16e6-4f95-8f82-7c210cf09d6d" />
</p>

---

### 8. Distribución de temporadas en series
Permite observar cuántas temporadas tienen las series disponibles en el catálogo.

<p align="center">
  <img width="850" alt="Distribución de temporadas en series" src="https://github.com/user-attachments/assets/7b427cdf-6a53-4655-975d-b86063987e6d" />
</p>

---

### 9. Directores con más títulos
Identifica los directores con mayor cantidad de títulos registrados en el dataset.

<p align="center">
  <img width="850" alt="Directores con más títulos" src="https://github.com/user-attachments/assets/9234b998-7443-4bb8-a8b1-9ced8e22b8d1" />
</p>

---

### 10. Actores y actrices con más apariciones
Muestra las personas del elenco con mayor frecuencia de aparición dentro del catálogo.

<p align="center">
  <img width="850" alt="Actores y actrices con más apariciones" src="https://github.com/user-attachments/assets/52358561-7758-4af3-ac55-70abbc062cbb" />
</p>

---

## Machine Learning
El problema fue abordado como una tarea de clasificación supervisada.
El objetivo del modelo fue clasificar automáticamente si un contenido corresponde a una película o una serie.

### Variable objetivo
`python`
type
Las clases corresponden a : Movie - TV Show


### Variables predictoras consideradas inicialmente
En una primera etapa se consideraron las variables:
- release_year
- rating
- duration_number
- country
- listed_in

Sin embargo, durante el análisis se detectó que duration_number generaba una fuga de información, ya que en el dataset la duración representa minutos para películas y temporadas para series.
Esto permitía al modelo distinguir de forma casi perfecta entre ambas clases, pero de una manera poco realista, ya que la variable contenía información directamente relacionada con la clase objetivo.
## Por esta razón, se eliminó duration_number del entrenamiento final.
<img width="1232" height="585" alt="image" src="https://github.com/user-attachments/assets/703cc2f9-887a-499b-9435-7c212e9ec0db" />

### Variables finales utilizadas
Las variables utilizadas para el entrenamiento final fueron:
- release_year
- rating
- country
- listed_in

### Modelos entrenados

Se entrenaron y compararon tres modelos de clasificación:

Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier

Cada modelo fue evaluado utilizando las siguientes métricas:
- `Accuracy`
- `Precision`
- `Recall`
- `F1-score`

--- 
## Evaluación de modelos

Para evaluar el rendimiento de los modelos de clasificación, se utilizaron métricas estándar de aprendizaje supervisado:

- **Accuracy:** proporción total de predicciones correctas.
- **Precision:** capacidad del modelo para evitar falsos positivos.
- **Recall:** capacidad del modelo para identificar correctamente los casos positivos.
- **F1-score:** métrica balanceada entre precision y recall.

En este proyecto se priorizó el **F1-score**, ya que entrega una evaluación más equilibrada del rendimiento del modelo al considerar tanto los errores de clasificación como la capacidad de detección de cada clase.

---

### Comparación visual de métricas

La siguiente visualización compara el rendimiento de los modelos entrenados mediante subplots, permitiendo observar de forma separada el comportamiento de cada métrica.

<p align="center">
  <img width="923" alt="Comparación visual de modelos con subplot" src="https://github.com/user-attachments/assets/1af28d2f-8e2e-4c17-8921-00305e60e004" />
</p>

---

### Comparación mediante heatmap

El heatmap permite comparar de forma compacta el desempeño de los modelos, facilitando la identificación del modelo con mejor rendimiento general.

<p align="center">
  <img width="923" alt="Comparación visual de modelos con heatmap" src="https://github.com/user-attachments/assets/b3d9a258-68ad-440c-a66a-8c46fc3c6926" />
</p>

---

### Interpretación de la evaluación

La evaluación de los modelos permitió comparar su capacidad para clasificar contenidos como **Movie** o **TV Show**.

Durante el proceso se detectó que la variable `duration_number` generaba una fuga de información, ya que en el dataset la duración representa minutos para películas y temporadas para series. Esto hacía que los modelos obtuvieran resultados artificialmente altos.

Para obtener una evaluación más realista, se eliminó dicha variable del entrenamiento final. Esta decisión permitió construir un modelo más honesto y demostrar la importancia de revisar cuidadosamente las variables utilizadas antes de entrenar un modelo de Machine Learning.

---

## Principales hallazgos

A partir del análisis exploratorio y del proceso de modelamiento, se identificaron los siguientes hallazgos:

- El catálogo presenta una mayor proporción de películas que de series.
- Estados Unidos aparece como uno de los países con mayor presencia dentro del catálogo.
- Los géneros más frecuentes corresponden principalmente a dramas, comedias, documentales y contenido internacional.
- Las clasificaciones más comunes muestran una fuerte presencia de contenido orientado a audiencias adolescentes y adultas.
- La duración de películas se concentra principalmente en rangos tradicionales de largometraje.
- En series, predominan contenidos con pocas temporadas.
- La variable `duration_number` fue identificada como una fuente de fuga de información para el modelo.

---

## Conclusiones

Este proyecto permitió aplicar un flujo completo de Ciencia de Datos sobre un dataset de entretenimiento, integrando análisis exploratorio, visualización, limpieza, transformación de variables y modelos de clasificación supervisada.

El análisis permitió comprender mejor la composición del catálogo de Netflix, identificando patrones relacionados con el tipo de contenido, países, géneros, clasificaciones, duración y principales participantes del catálogo.

Desde el enfoque de Machine Learning, fue posible construir modelos capaces de clasificar automáticamente el tipo de contenido a partir de variables descriptivas del catálogo.

Uno de los aprendizajes más relevantes fue la detección de fuga de información en la variable `duration_number`. Esta observación permitió mejorar la validez del modelo y reforzar la importancia del criterio analítico en proyectos de Machine Learning.

En conjunto, el proyecto demuestra cómo un dataset de entretenimiento puede utilizarse para desarrollar un flujo completo de análisis de datos, desde la exploración visual hasta la construcción y evaluación de modelos predictivos.
