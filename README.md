# Evaluación No Supervisada: GMM, KDE y DBSCAN en Atenciones Hospitalarias de Diabetes (1999–2008)

**Machine Learning II — Parcial 2**  
**Maestría en Aprendizaje Automático e Inteligencia Artificial**

---

## 1. Descripción del Proyecto

Este repositorio contiene el desarrollo e implementación del Parcial 2 de Machine Learning II. El trabajo aplica tres enfoques de aprendizaje no supervisado (Gaussian Mixture Models, Kernel Density Estimation y DBSCAN) sobre el conjunto de datos de atención hospitalaria en diabetes de 130 centros de salud en Estados Unidos (1999–2008).

El análisis se estructura en un plano bidimensional compuesto por la estancia hospitalaria (`time_in_hospital`) y la cantidad de medicamentos administrados (`num_medications`), permitiendo caracterizar la concentración de atenciones, estimar la densidad asistencial y detectar casos clínicos atípicos.

---

## 2. Estructura del Repositorio

```
MLII_Parcial2/
├── data/
│   └── diabetic_data.csv       # Conjunto de datos crudo
├── src/
│   └── salud.py                # Funciones de carga, preprocesamiento y graficado
├── notebooks/
│   └── parcial_2_salud_unificado.ipynb  # Cuaderno Jupyter con la ejecución y análisis
├── reports/
│   ├── parcial_2_salud_unificado.pdf    # Informe exportado en PDF
│   └── parcial_2_salud_unificado.html   # Reporte en formato HTML
├── results/
│   ├── figures/                # Gráficos generados durante los experimentos
│   └── tables/                 # Tablas cuantitativas y de selección en CSV
├── ejecucion_paso_a_paso.txt   # Registro de auditoría del pipeline de ejecución
├── requirements.txt            # Librerías requeridas
└── README.md                   # Documentación de la entrega
```

---

## 3. Metodología y Auditoría de Datos

1. **Unidad de Análisis**: El dataset crudo contiene 101.766 registros. Se conservó únicamente el primer encuentro registrado por paciente único (71.518 registros) para evitar autocorrelación por reingresos repetidos.
2. **Espacio de Representación**: Se trabajó en el plano bidimensional de días de estancia y número de medicamentos, manteniendo las unidades originales para facilitar la interpretación.
3. **División de Datos**: Se definió una partición del 80% para desarrollo (57.214 registros) y un 20% para prueba congelado (14.304 registros) utilizando la semilla fija `random_state=42`.
4. **Preprocesamiento**:
   - Para GMM y KDE se utilizaron los valores naturales de las variables.
   - Para DBSCAN se aplicó estandarización z-score (`StandardScaler`) para equilibrar la escala entre los días de estancia y la cantidad de fármacos.

---

## 4. Resumen de Resultados

| Modelo | Configuración Seleccionada | Criterio de Selección | Resultado Clínico Principal |
| :--- | :--- | :--- | :--- |
| **GMM** | K=6, Covarianza full | Mínimo BIC (64.149,86) | Delimita zonas de atención con elipses de confianza al 95%. |
| **KDE** | h=0,2, Kernel gaussian | Log-densidad media en CV (-4,8019) | Muestra el núcleo continuo de mayor masa hospitalaria. |
| **DBSCAN** | eps=0,3, min_samples=15 | Coeficiente de Silueta | Separa un 3,75% de datos atípicos con estancias prolongadas y alta carga de medicamentos. |

---

## 5. Requisitos e Instalación

### Entorno Virtual e Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Ejecución del Proyecto

### Replicar Experimentos y Tablas
```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from salud import *; setup()"
```

### Consultar el Cuaderno Jupyter
```bash
jupyter notebook notebooks/parcial_2_salud_unificado.ipynb
```
