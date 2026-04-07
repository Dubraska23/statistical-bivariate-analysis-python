# 📊 Análisis Estadístico Invocando Principios de Data Governance
### Estudio de Microdatos CIS 3495 (Formato SPSS)

Este proyecto desarrolla un análisis estadístico bivariante e inferencial utilizando Python, aplicando un enfoque de **Data Stewardship** para garantizar la integridad, trazabilidad y calidad de los datos desde la ingesta hasta la visualización.

---

## 🛡️ Visión de Gobierno de Datos y Calidad
A diferencia de un análisis convencional, este proyecto integra controles de **Data Governance** fundamentales para un entorno corporativo:

1. **Gestión de Metadatos:** Se automatiza la extracción de etiquetas de negocio de los archivos `.sav` (SPSS). Esto asegura que el significado semántico de las variables se mantenga íntegro, evitando errores de interpretación humana.
2. **Data Cleansing & Integrity:** Implementación de protocolos de limpieza para el tratamiento de valores nulos y mapeo de categorías, asegurando que solo los datos que cumplen con los criterios de calidad entren en el modelo estadístico.
3. **Consistencia de Tipos:** Transformación controlada de variables numéricas a categóricas (`category` type en Pandas) para optimizar el uso de memoria y garantizar la coherencia en las pruebas de hipótesis.
4. **Reproducibilidad y Trazabilidad:** Código estructurado bajo estándares de documentación técnica que permiten auditar el origen de cada transformación.

---

## 🎯 Objetivos Analíticos
- **Análisis Bivariante:** Explorar relaciones entre variables sociodemográficas.
- **Validación de Hipótesis:** Aplicación de pruebas paramétricas (**ANOVA**, **T-Test**) para identificar diferencias significativas entre grupos.
- **Análisis de Varianza:** Implementación del **Test de Tukey** para comparaciones múltiples y detección de patrones no evidentes.
- **Correlación:** Análisis de la fuerza de asociación entre dimensiones del estudio.

---

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.10+
- **Librerías de Datos:** `Pandas`, `Numpy`.
- **Análisis Estadístico:** `Scipy`, `Statsmodels`, `Pingouin`.
- **Integración SPSS:** `Pyreadstat`.
- **Visualización:** `Seaborn`, `Matplotlib`.

---

## 📁 Estructura del Proyecto
- `analisis_bivariante.py`: Script principal con la lógica de limpieza y análisis.
- `3495.sav`: Dataset original (Microdatos del CIS).
- `README.md`: Documentación del proyecto y visión de gobierno.

---

## 🚀 Instalación y Uso
1. Clona el repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/statistical-bivariate-analysis.git](https://github.com/TU_USUARIO/statistical-bivariate-analysis.git)
2. Instala las dependencias necesarias:
   pip install pandas numpy seaborn matplotlib pyreadstat scipy pingouin statsmodels tabulate
3. Ejecuta el análisis:
   python analisis_bivariante.py
   
