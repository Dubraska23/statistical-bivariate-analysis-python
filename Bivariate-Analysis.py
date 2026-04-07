# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:44:27 2025

@author: Dubraska Veroes
"""

### Importar librerías

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat
import openpyxl
from scipy import stats
import pingouin as pg
import statsmodels.api as sm
from tabulate import tabulate
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %% FICHERO DE TRABAJO

# =============================================================================
# ABRIR EL FICHERO DE TRABAJO
# =============================================================================

# Abrir fichero SPSS

df, meta = pyreadstat.read_sav(r"C:\Users\karli\Downloads\Eva_medina\Archivos de trabajo-20251117\3495.sav")
(df.head())

print(df.columns.tolist())

# Reemplazar códigos por etiquetas en todas las variables que las tengan
for var, label_key in meta.variable_to_label.items():
    if var in df.columns:
        etiquetas = meta.value_labels[label_key]
        df[var] = df[var].replace(etiquetas).astype("category")

# Exploración rápida del DataFrame
df.shape # dimensión
df.dtypes # tipo de dato
df.info() # resumen del DdataFrame

# Guardar en tablas el listado de variables numéricas y categóricas
numericas = df.select_dtypes(include=['float64', 'int64']).columns
tabla_numericas = pd.DataFrame({
    'Variable': numericas,
    'Etiqueta': [meta.column_labels[meta.column_names.index(col)] for col in numericas]
})
tabla_numericas_copy = tabla_numericas.copy()

categoricas = df.select_dtypes(include=['object', 'category']).columns
tabla_categoricas = pd.DataFrame({
    'Variable': categoricas,
    'Etiqueta': [meta.column_labels[meta.column_names.index(col)] for col in categoricas]
})
tabla_categoricas_copy = tabla_categoricas.copy()

# Revisión de datos faltantes o duplicados
df.count() #valores no nulos
df.isna().sum() #valores faltantes
df.duplicated().sum() # filas duplicadas

print("¡Todo instalado correctamente!")

# =============================================================================
#%% A.1 Análisis categórica vs categórica: INGRESHOG_R3G by P15_1 a P15_4
# =============================================================================

# Paso 1: Recodificar ingresos en 3 grupos

mapa_ingresos = {
    'Menos o igual a 300 €': 'Bajo',
    'De 301 a 600 €': 'Bajo',
    'De 601 a 900 €': 'Bajo',
    'De 901 a 1.200 €': 'Medio',
    'De 1.201 a 1.800 €': 'Medio',
    'De 1.801 a 2.400 €': 'Medio',
    'De 2.401 a 3.000 €': 'Alto',
    'De 3.001 a 4.500 €': 'Alto',
    'De 4.501 a 6.000 €': 'Alto',
    'Más de 6.000 €': 'Alto',
    'No tienen ingresos de ningún tipo': 'Bajo',
    'NS/NC': np.nan  # si existiera
}

df['INGRESHOG_R3G'] = df['INGRESHOG'].map(mapa_ingresos)
df['INGRESHOG_R3G'] = df['INGRESHOG_R3G'].astype('category')

print(df['INGRESHOG_R3G'].value_counts(dropna=False))

print(df['INGRESHOG'].describe())
print(df['INGRESHOG_R3G'].value_counts(dropna=False))

# Paso 2: Definir variables
grupo = 'INGRESHOG_R3G'
variables = ['P15_1', 'P15_2', 'P15_3', 'P15_4']

# Función para aplicar etiquetas SPSS
def aplicar_etiquetas(idx, etiquetas):
    return [etiquetas.get(v, v) for v in idx]

# Paso 3: Análisis univariante de la variable de ingresos
tabla_ingresos = pd.DataFrame({
    'Frecuencia': df[grupo].value_counts(dropna=True),
    'Porcentaje': df[grupo].value_counts(normalize=True, dropna=True) * 100
})
print("=== Distribución de INGRESHOG_R3G ===")
print(tabla_ingresos)

# Paso 4: Tablas cruzadas y porcentajes por columna
for var in variables:
    print(f"\n=== TABLA CRUZADA: {var} by {grupo} ===")
    
    tabla = pd.crosstab(df[var], df[grupo], dropna=True)
    tabla_pct = pd.crosstab(df[var], df[grupo], normalize='columns', dropna=True) * 100

    # Aplicar etiquetas SPSS
    value_labels_var = meta.variable_value_labels.get(var, {})
    value_labels_grupo = meta.variable_value_labels.get(grupo, {})

    tabla.index = aplicar_etiquetas(tabla.index, value_labels_var)
    tabla.columns = aplicar_etiquetas(tabla.columns, value_labels_grupo)
    tabla_pct.index = aplicar_etiquetas(tabla_pct.index, value_labels_var)
    tabla_pct.columns = aplicar_etiquetas(tabla_pct.columns, value_labels_grupo)

    # Combinar tablas
    tabla_final = pd.concat({'Frecuencia': tabla, 'Porcentaje': tabla_pct}, axis=1)
    print(tabla_final)

#%%A.2 VAR. NOM/ORD by NOM/ORD – Estadístico X² y de asociación: P15_? by INGRESHOG_R3G

from scipy.stats import chi2_contingency

# Lista de variables P15
variables = ['P15_1', 'P15_2', 'P15_3', 'P15_4']
grupo = 'INGRESHOG_R3G'

resultados = []

for var in variables:

    # 1. Eliminar casos con valores perdidos en cualquiera de las dos variables
    df_temp = df[[var, grupo]].dropna()

    # 2. Tabla de contingencia limpia
    tabla = pd.crosstab(df_temp[var], df_temp[grupo])
    
    # Chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(tabla)
    n = tabla.sum().sum()
    
    # Phi (válido solo para 2x2, pero se calcula igual)
    phi = np.sqrt(chi2 / n)
    
    # V de Cramer
    k = min(tabla.shape)
    v_cramer = np.sqrt(chi2 / (n * (k - 1)))
    
    # Coeficiente de contingencia
    coef_contingencia = np.sqrt(chi2 / (chi2 + n))
    
    # Guardar resultados
    resultados.append({
        'Variable': var,
        'Chi2': round(chi2, 3),
        'gl': dof,
        'p-valor': round(p, 4),
        'Phi': round(phi, 3),
        'V_Cramer': round(v_cramer, 3),
        'Coef_Contingencia': round(coef_contingencia, 3)
    })

# Convertir a DataFrame
res_df = pd.DataFrame(resultados)
print("\n=== RESULTADOS ESTADÍSTICOS: P15_? by INGRESHOG_R3G ===")
print(res_df)

#%%A.3 Representacion grafica barras agrupadas

import seaborn as sns
import matplotlib.pyplot as plt

# Lista de variables P15
variables = ['P15_1', 'P15_2', 'P15_3', 'P15_4']
grupo = 'INGRESHOG_R3G'

for var in variables:
    # Asegurar que sean categóricas
    df[grupo] = df[grupo].astype('category')
    df[var] = df[var].astype('category')

    # Tabla de porcentajes por columna (por grupo de ingresos)
    tabla = pd.crosstab(df[var], df[grupo], dropna=True)
    tabla_pct = tabla.div(tabla.sum(axis=0), axis=1) * 100

    # Convertir a formato largo para Seaborn
    df_plot = tabla_pct.reset_index().melt(id_vars=var, var_name=grupo, value_name='Porcentaje')

    # Gráfico
    plt.figure(figsize=(8,6))
    ax = sns.barplot(data=df_plot, x=grupo, y='Porcentaje', hue=var)

    # Etiquetas
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                x=p.get_x() + p.get_width()/2,
                y=height + 1,
                s=f'{height:.1f}%',
                ha='center'
            )

    plt.ylabel('Porcentaje (%)')
    plt.xlabel('Nivel de ingresos')
    plt.title(f'Distribución de {var} por nivel de ingresos (Column %)')
    plt.legend(title=var)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()
    
# Representacion grafica barras apiladas

variables = ['P15_1', 'P15_2', 'P15_3', 'P15_4']
grupo = 'INGRESHOG_R3G'

for var in variables:
    # Asegurar que sean categóricas
    df[grupo] = df[grupo].astype('category')
    df[var] = df[var].astype('category')

    # Tabla de porcentajes por fila (por grupo de ingresos)
    tabla = pd.crosstab(df[grupo], df[var], dropna=True)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    # Orden de categorías
    categorias = tabla_pct.columns.tolist()

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,6))
    bottom = [0]*len(tabla_pct)
    colores = plt.cm.tab10.colors

    for i, cat in enumerate(categorias):
        ax.bar(tabla_pct.index, tabla_pct[cat], bottom=bottom, label=cat, color=colores[i % len(colores)])
        for j, val in enumerate(tabla_pct[cat]):
            if val > 0:
                ax.text(
                    x=j,
                    y=bottom[j] + val/2,
                    s=f'{val:.1f}%',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=10
                )
        bottom = [b + h for b, h in zip(bottom, tabla_pct[cat])]

    ax.set_ylabel('Porcentaje (%)')
    ax.set_xlabel('Nivel de ingresos')
    ax.set_title(f'Distribución de {var} por nivel de ingresos (Barras apiladas)')
    ax.set_ylim(0, 100)
    ax.legend(title=var, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()
   
# =============================================================================
#%% A.4. VAR. NOM/ORD by NOM/ORD (nxn) – P15_? 
# =============================================================================

# 1 Recodificación de ingresos en 3 grupos

mapa_ingresos = {
    'Menos o igual a 300 €': 'Bajo',
    'De 301 a 600 €': 'Bajo',
    'De 601 a 900 €': 'Bajo',
    'De 901 a 1.200 €': 'Medio',
    'De 1.201 a 1.800 €': 'Medio',
    'De 1.801 a 2.400 €': 'Medio',
    'De 2.401 a 3.000 €': 'Alto',
    'De 3.001 a 4.500 €': 'Alto',
    'De 4.501 a 6.000 €': 'Alto',
    'Más de 6.000 €': 'Alto',
    'No tienen ingresos de ningún tipo': 'Bajo',
    'NS/NC': np.nan
}
df['INGRESHOG_R3G'] = df['INGRESHOG'].map(mapa_ingresos).astype('category')

# 2 Análisis univariante

variables = ['P15_1', 'P15_2', 'P15_3', 'P15_4']

# --- Distribución de ingresos ---
tabla_ingresos = pd.DataFrame({
    'Frecuencia': df['INGRESHOG_R3G'].value_counts(dropna=True),
    'Porcentaje': df['INGRESHOG_R3G'].value_counts(normalize=True, dropna=True) * 100
})
print("=== Distribución de INGRESHOG_R3G ===")
print(tabla_ingresos)

# --- Distribución de cada ítem P15 ---
for var in variables:
    tabla_univar = pd.DataFrame({
        'Frecuencia': df[var].value_counts(dropna=True),
        'Porcentaje': df[var].value_counts(normalize=True, dropna=True) * 100
    })
    print(f"\n=== Distribución de {var} ===")
    print(tabla_univar)

# 3 Tablas cruzadas, Chi², residuos y mapas de calor

from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Función para aplicar etiquetas SPSS ---
def aplicar_etiquetas(idx, etiquetas):
    return [etiquetas.get(v, v) for v in idx]

# --- Análisis cruzado ---
resumen_chi2 = []
tablas_residuos = {}

for var in variables:
    print(f"\n=== TABLA CRUZADA: {var} by INGRESHOG_R3G ===")
    
    tabla = pd.crosstab(df[var], df['INGRESHOG_R3G'], dropna=True)
    tabla_pct = pd.crosstab(df[var], df['INGRESHOG_R3G'], normalize='columns', dropna=True) * 100

    value_labels_var = meta.variable_value_labels.get(var, {})
    value_labels_grupo = meta.variable_value_labels.get('INGRESHOG_R3G', {})

    tabla.index = aplicar_etiquetas(tabla.index, value_labels_var)
    tabla.columns = aplicar_etiquetas(tabla.columns, value_labels_grupo)
    tabla_pct.index = aplicar_etiquetas(tabla_pct.index, value_labels_var)
    tabla_pct.columns = aplicar_etiquetas(tabla_pct.columns, value_labels_grupo)

    tabla_final = pd.concat({'Frecuencia': tabla, 'Porcentaje': tabla_pct}, axis=1)
    print(tabla_final.to_string(index=False))

    # --- Estadístico Chi² ---
    chi2, p, dof, expected = chi2_contingency(tabla)
    n_total = tabla.sum().sum()
    v_cramer = np.sqrt(chi2 / (n_total * (min(tabla.shape) - 1)))

    resumen_chi2.append({
        'Variable': var,
        'Etiqueta': meta.column_labels[meta.column_names.index(var)],
        'Chi²': round(chi2, 3),
        'gl': dof,
        'p-valor': round(p, 4),
        'V de Cramer': round(v_cramer, 3)
    })

    # --- Residuos estandarizados corregidos ---
    observed = tabla.values
    expected = np.array(expected)
    row_totals = observed.sum(axis=1)[:, None]
    col_totals = observed.sum(axis=0)[None, :]
    resid_std_corr = (observed - expected) / np.sqrt(expected * (1 - row_totals/n_total) * (1 - col_totals/n_total))
    resid_std_corr_df = pd.DataFrame(resid_std_corr, index=tabla.index, columns=tabla.columns)

    tablas_residuos[var] = resid_std_corr_df

    # --- Mapa de calor de residuos ---
    plt.figure(figsize=(8,6))
    sns.heatmap(resid_std_corr_df, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Residuos estandarizados corregidos: {var}")
    plt.tight_layout()
    plt.show()

# --- Tabla resumen Chi² ---
tabla_resumen_chi2 = pd.DataFrame(resumen_chi2).sort_values(by='V de Cramer', ascending=False).reset_index(drop=True)
print("\n=== RESUMEN CHI² ===")
print(tabla_resumen_chi2)

# --- Mostrar residuos por variable ---
for var, tabla_res in tablas_residuos.items():
    print(f"\n=== RESIDUOS ESTANDARIZADOS CORREGIDOS: {var} ===")
    print(tabla_res.round(2))
    
    # %%B.1. VAR. METRICA by NOM/ORD
    
# Paso 1: Lista de variables P3
variables = ['P3_1', 'P3_2', 'P3_3', 'P3_4', 'P3_5', 'P3_6', 'P3_7', 'P3_8', 'P3_9', 'P3_10', 'P3_11', 'P3_12']
grupo = 'INGRESHOG_R3G'

# Paso 2: Estadísticos descriptivos univariantes
estadisticos = df[variables].describe().T
estadisticos['median'] = df[variables].median()
estadisticos['missing'] = df[variables].isna().sum()
estadisticos['missing_pct'] = df[variables].isna().mean() * 100

# Reemplazar índice por etiquetas SPSS
etiquetas = [meta.column_labels[meta.column_names.index(var)] for var in variables]
estadisticos.index = etiquetas

# Guardar tabla para Variable Explorer
tabla_estadisticos = estadisticos

# Mostrar en consola
print("=== Estadísticos descriptivos de P3_? ===")
print(tabla_estadisticos)

# Paso 3: Análisis bivariante – medias por grupo de ingresos
df[grupo] = df[grupo].astype('category')
df[variables] = df[variables].apply(pd.to_numeric, errors='coerce')

tabla_medias = df.groupby(grupo)[variables].mean()

# Etiquetas SPSS para columnas y filas
etiquetas_vars = [meta.column_labels[meta.column_names.index(var)] for var in variables]
tabla_medias.columns = etiquetas_vars

value_labels_grupo = meta.variable_value_labels.get(grupo, {})
etiquetas_grupo = {k: v for k, v in value_labels_grupo.items()}
tabla_medias.index = tabla_medias.index.map(lambda x: etiquetas_grupo.get(x, x))

# Transponer tabla
tabla_medias_transpuesta = tabla_medias.T
tabla_medias_final = tabla_medias_transpuesta

# Mostrar resultados
print("\n=== Medias de P3_? por nivel de ingresos ===")
print(tabla_medias_final)

#%%B.2 VAR. METRICA BY NOM/ORD (GRAFICO)
#Gráfico de medias con IC95%

from scipy import stats

# --- Variables ---
variables = ['P3_1', 'P3_2', 'P3_3', 'P3_4', 'P3_5', 'P3_6', 'P3_7', 'P3_8', 'P3_9', 'P3_10', 'P3_11', 'P3_12']
agrupadora = 'INGRESHOG_R3G'

# --- Preparar datos ---
df[agrupadora] = df[agrupadora].astype('category')
value_labels = meta.variable_value_labels.get(agrupadora, {})
etiquetas_ingresos = [value_labels.get(val, val) for val in df[agrupadora].cat.categories]

# --- Crear subplots ---
fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
axes = axes.flatten()

for i, variable in enumerate(variables):
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    grouped = df.groupby(agrupadora)[variable]
    medias = grouped.mean()
    desvios = grouped.std()
    ns = grouped.count()
    stderr = desvios / np.sqrt(ns)
    conf_int = stats.t.ppf(0.975, ns - 1) * stderr

    x = np.arange(len(etiquetas_ingresos))

    axes[i].errorbar(
        x, medias, yerr=conf_int,
        fmt='o', color='blue', ecolor='black', elinewidth=2, capsize=6, markersize=8
    )

    for j, mean in enumerate(medias):
        axes[i].text(x[j], mean + conf_int.iloc[j] + 0.05, f"{mean:.2f}",
                     ha='center', va='bottom', fontsize=9, color='black')

    etiqueta_variable = meta.column_labels[meta.column_names.index(variable)]
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(etiquetas_ingresos)
    axes[i].set_xlim(-0.5, len(etiquetas_ingresos) - 0.5)
    axes[i].set_title(f"{etiqueta_variable}")
    axes[i].grid(axis='y', linestyle='--', alpha=0.6)

fig.text(0.04, 0.5, 'Valor medio', va='center', rotation='vertical')
plt.tight_layout()
plt.show()


#Gráfico Box-Plot
# --- Crear subplots para los 12 boxplots ---
fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
axes = axes.flatten()

etiqueta_agrupadora = "Nivel de ingresos"
etiquetas_ingresos = [value_labels.get(val, str(val)) for val in df[agrupadora].cat.categories]

for i, variable in enumerate(variables):
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    etiqueta_variable = meta.column_labels[meta.column_names.index(variable)]

    df.boxplot(column=variable, by=agrupadora, grid=False, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="blue"),
               medianprops=dict(color="red", linewidth=2),
               ax=axes[i])

    axes[i].set_title(f"{etiqueta_variable} por {etiqueta_agrupadora}")
    axes[i].set_xlabel(etiqueta_agrupadora)
    axes[i].set_ylabel(etiqueta_variable if i % 4 == 0 else "")
    axes[i].set_xticks(range(1, len(etiquetas_ingresos) + 1))
    axes[i].set_xticklabels(etiquetas_ingresos)
    axes[i].grid(axis='y', linestyle='--', alpha=0.6)

plt.suptitle("")
plt.tight_layout()
plt.show()

# %%B.3. VAR. METRICA by NOM/ORD (2 grupos) - Estadísticos
from scipy import stats

# --- 1. Variables ---
variables = ['P3_1', 'P3_2', 'P3_3', 'P3_4', 'P3_5', 'P3_6', 'P3_7', 'P3_8', 'P3_9', 'P3_10', 'P3_11', 'P3_12']
agrupadora = 'INGRESHOG_R3G'

# --- 2. Asegurar tipos correctos ---
df[agrupadora] = df[agrupadora].astype('category')
df[variables] = df[variables].apply(pd.to_numeric, errors='coerce')

# --- 3. Etiquetas SPSS ---
etiquetas_vars = [meta.column_labels[meta.column_names.index(v)] for v in variables]
value_labels = meta.variable_value_labels.get(agrupadora, {})
etiquetas_grupo = [value_labels.get(val, str(val)) for val in df[agrupadora].cat.categories]

# --- 4. ANOVA y Kruskal-Wallis ---
resultados = []

for var, etiqueta_var in zip(variables, etiquetas_vars):
    grupos = [df[df[agrupadora] == g][var].dropna() for g in df[agrupadora].cat.categories]

    # ANOVA
    f_stat, p_anova = stats.f_oneway(*grupos)

    # Kruskal-Wallis
    h_stat, p_kw = stats.kruskal(*grupos)

    # Medias por grupo
    medias = [g.mean() for g in grupos]
    ns = [g.size for g in grupos]

    resultados.append({
        'Variable': etiqueta_var,
        f'Media {etiquetas_grupo[0]}': round(medias[0], 2),
        f'Media {etiquetas_grupo[1]}': round(medias[1], 2),
        f'Media {etiquetas_grupo[2]}': round(medias[2], 2),
        'ANOVA F': round(f_stat, 3),
        'ANOVA p-valor': round(p_anova, 4),
        'Kruskal H': round(h_stat, 3),
        'Kruskal p-valor': round(p_kw, 4),
        f'N {etiquetas_grupo[0]}': ns[0],
        f'N {etiquetas_grupo[1]}': ns[1],
        f'N {etiquetas_grupo[2]}': ns[2]
    })

tabla_anova = pd.DataFrame(resultados)
print("=== Comparación de medias: ANOVA y Kruskal-Wallis ===")
print(tabla_anova)

##Supuestos: Normalidad y Homogeneidad de Varianzas
# --- 5. Normalidad por grupo (Kolmogorov-Smirnov) ---
normalidad = []

for var, etiqueta_var in zip(variables, etiquetas_vars):
    fila = {'Variable': etiqueta_var}
    for i, grupo in enumerate(df[agrupadora].cat.categories):
        datos = df[df[agrupadora] == grupo][var].dropna()
        if len(datos) > 1:
            z = (datos - datos.mean()) / datos.std(ddof=0)
            stat, p = stats.kstest(z, 'norm')
        else:
            p = np.nan
        fila[f'KS {etiquetas_grupo[i]} (p)'] = round(p, 4) if p is not np.nan else None
    normalidad.append(fila)

tabla_normalidad = pd.DataFrame(normalidad)
print("\n=== Test de normalidad (Kolmogorov-Smirnov) ===")
print(tabla_normalidad)

# --- 6. Homogeneidad de varianzas (Levene) ---
homogeneidad = []

for var, etiqueta_var in zip(variables, etiquetas_vars):
    grupos = [df[df[agrupadora] == g][var].dropna() for g in df[agrupadora].cat.categories]
    stat, p = stats.levene(*grupos)
    homogeneidad.append({'Variable': etiqueta_var, 'Levene p-valor': round(p, 4)})

tabla_levene = pd.DataFrame(homogeneidad)
print("\n=== Test de homogeneidad de varianzas (Levene) ===")
print(tabla_levene)

# =============================================================================
#%% B.4. VAR. MÉTRICA by NOM/ORD (+ de 2 grupos): P3_? by INGRESHOG_R3G
# =============================================================================

import scipy.stats as stats
import pingouin as pg

# --- 1. Variables ---
variables = ['P3_1', 'P3_2', 'P3_3', 'P3_4']
agrupadora = 'INGRESHOG_R3G'

# --- 2. Asegurar tipos correctos ---
df[agrupadora] = df[agrupadora].astype('category')
df[variables] = df[variables].apply(pd.to_numeric, errors='coerce')

# --- 3. Obtener etiquetas SPSS ---
etiquetas_vars = [meta.column_labels[meta.column_names.index(v)] for v in variables]
value_labels = meta.variable_value_labels.get(agrupadora, {})
etiquetas_grupos = [value_labels.get(val, str(val)) for val in df[agrupadora].cat.categories]

# --- 4. Calcular medias por grupo ---
descriptivos = []

for var, etiqueta_var in zip(variables, etiquetas_vars):
    fila = {'Variable': etiqueta_var}
    for i, grupo in enumerate(df[agrupadora].cat.categories):
        data = df[df[agrupadora] == grupo][var].dropna()
        fila[f'Media {etiquetas_grupos[i]}'] = data.mean()
    descriptivos.append(fila)

tabla_descriptivos = pd.DataFrame(descriptivos)

# --- 5. Mostrar resultados ---
print("Estadísticos descriptivos (Media por grupo):")
print(tabla_descriptivos)

# --- 6. Boxplot para cada variable ---
for var, etiqueta_var in zip(variables, etiquetas_vars):
    plt.figure(figsize=(7,5))
    df.boxplot(column=var, by=agrupadora, grid=False, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="blue"),
               medianprops=dict(color="red", linewidth=2))
    plt.title(f"{etiqueta_var} por {agrupadora}")
    plt.suptitle("")
    plt.xlabel(agrupadora)
    plt.ylabel(etiqueta_var)
    plt.xticks(ticks=range(1, len(etiquetas_grupos)+1), labels=etiquetas_grupos)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- 7. ANOVA (F de Snedecor) ---
anova_resultados = []
for var, etiqueta_var in zip(variables, etiquetas_vars):
    grupos = [df[df[agrupadora] == cat][var].dropna() for cat in df[agrupadora].cat.categories]
    f_stat, p_val = stats.f_oneway(*grupos)
    anova_resultados.append({'Variable': etiqueta_var, 'F': round(f_stat,3), 'p-valor': round(p_val,4)})

tabla_anova = pd.DataFrame(anova_resultados)
print("\nANOVA (F de Snedecor):")
print(tabla_anova)

# --- 8. Contrastes de normalidad y homogeneidad ---
normalidad = []
homogeneidad = []

for var, etiqueta_var in zip(variables, etiquetas_vars):
    row_norm = {'Variable': etiqueta_var}
    grupos = []
    etiquetas_grupos = []

    for cat in df[agrupadora].cat.categories:
        grupo = df[df[agrupadora] == cat][var].dropna()
        grupos.append(grupo)
        etiquetas_grupos.append(value_labels.get(cat, str(cat)))

    for i, g in enumerate(grupos):
        etiqueta = etiquetas_grupos[i]
        stat, p = stats.shapiro(g)
        row_norm[f'Shapiro-Wilk {etiqueta} (p)'] = round(p, 4)

    normalidad.append(row_norm)

    stat, p = stats.levene(*grupos)
    homogeneidad.append({'Variable': etiqueta_var, 'Levene p-valor': round(p, 4)})

tabla_normalidad = pd.DataFrame(normalidad)
tabla_levene = pd.DataFrame(homogeneidad)

print("\nNormalidad (Shapiro-Wilk) por grupo:")
print(tabla_normalidad)
print("\nHomogeneidad de varianzas (Levene):")
print(tabla_levene)

# --- 9. Welch ANOVA (F robusta) ---
robust_results = []

for var, etiqueta_var in zip(variables, etiquetas_vars):
    anova_res = pg.welch_anova(dv=var, between=agrupadora, data=df)
    f_val = round(anova_res['F'].values[0], 3)
    p_val = round(anova_res['p-unc'].values[0], 4)
    robust_results.append({'Variable': etiqueta_var, 'F robusta (Welch)': f_val, 'p-valor': p_val})

tabla_f_robusta = pd.DataFrame(robust_results)
print("\nF robusta (Welch ANOVA):")
print(tabla_f_robusta)

# --- 10. Kruskal-Wallis (no paramétrico) ---
np_resultados = []
for var, etiqueta_var in zip(variables, etiquetas_vars):
    grupos = [df[df[agrupadora] == cat][var].dropna() for cat in df[agrupadora].cat.categories]
    stat, p_val = stats.kruskal(*grupos)
    np_resultados.append({'Variable': etiqueta_var, 'Kruskal-Wallis H': round(stat,3), 'p-valor': round(p_val,4)})

tabla_np = pd.DataFrame(np_resultados)
print("\nPruebas no paramétricas (Kruskal-Wallis):")
print(tabla_np)



# %%B.5. VAR. METRICA by NOM/ORD (+ de 2 grupos) Pruebas Post Hoc

# =============================================================================

# --- 2. Variables ---
variable = 'P3_1'
agrupadora = 'INGRESHOG_R3G'

# --- 3. Preparar datos ---
df[variable] = pd.to_numeric(df[variable], errors='coerce')
df[agrupadora] = df[agrupadora].astype(str)

# Eliminar filas con NaN en las dos variables
df_filtrado = df[[variable, agrupadora]].dropna()
df_filtrado[agrupadora] = df_filtrado[agrupadora].astype('category')
df_filtrado[variable] = pd.to_numeric(df_filtrado[variable], errors='coerce')

# Verificar que haya más de un grupo con datos
print(df_filtrado[agrupadora].value_counts())

# --- 4. Tukey HSD ---
if df_filtrado[agrupadora].nunique() > 1:
    tukey = pairwise_tukeyhsd(endog=df_filtrado[variable],
                              groups=df_filtrado[agrupadora],
                              alpha=0.05)

    # Convertir a DataFrame
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    print("\nResultados Tukey HSD para P3_1 según INGRESHOG_R3G:")
    print(tukey_df)



### Si ANOVA robusta (Welch) → se puede usar Games-Howell (pingouin.pairwise_gameshowell), que es robusta a heterocedasticidad y tamaños desiguales.

# --- 1. Variables ---
variable = 'P3_1'
agrupadora = 'INGRESHOG_R3G'

df_filtrado = df[[variable, agrupadora]].dropna()
df_filtrado[agrupadora] = df_filtrado[agrupadora].astype('category')
df_filtrado[variable] = pd.to_numeric(df_filtrado[variable], errors='coerce')

# --- 2. Preparar datos ---
df[agrupadora] = df[agrupadora].astype('category')
df[variable] = pd.to_numeric(df[variable], errors='coerce')

# Etiquetas SPSS
value_labels = meta.variable_value_labels.get(agrupadora, {})
etiquetas_grupos = [value_labels.get(val, str(val)) for val in df[agrupadora].cat.categories]

# --- 3. Games-Howell post hoc ---
gh = pg.pairwise_gameshowell(dv=variable, between=agrupadora, data=df)

# Reemplazar nombres de grupos por etiquetas SPSS
gh['A'] = gh['A'].map(lambda x: value_labels.get(x, str(x)))
gh['B'] = gh['B'].map(lambda x: value_labels.get(x, str(x)))

# Guardar para Variable Explorer
tabla_gh_final = gh

print("Pruebas post hoc Games-Howell para P3_1:")
print(tabla_gh_final)

# --- Variables ---
variable = 'P3_3'
agrupadora = 'INGRESHOG_R3G'

# --- Filtrar NaN 
df_filtrado = df[[variable, agrupadora]].dropna()
df_filtrado[agrupadora] = df_filtrado[agrupadora].astype('category')
df_filtrado[variable] = pd.to_numeric(df_filtrado[variable], errors='coerce')

# --- Etiquetas SPSS ---
value_labels = meta.variable_value_labels.get(agrupadora, {})
df_filtrado[agrupadora] = df_filtrado[agrupadora].cat.remove_unused_categories()
etiquetas_grupos = [value_labels.get(val, str(val)) for val in df_filtrado[agrupadora].cat.categories]

# --- Games-Howell post hoc ---
gh = pg.pairwise_gameshowell(dv=variable, between=agrupadora, data=df_filtrado)

# --- Reemplazar nombres por etiquetas legibles ---
gh['A'] = gh['A'].map(lambda x: value_labels.get(x, str(x)))
gh['B'] = gh['B'].map(lambda x: value_labels.get(x, str(x)))

# --- Mostrar resultados ---
print(f"\nPruebas post hoc Games-Howell para {variable}:")
print(gh)

# %%C.1. VAR. MÉTRICA by METRICA - Gráfico

# =============================================================================
# Recodificación por marca de clase: INGRESHOG → INGRESOS_RMARCA
# =============================================================================

# Diccionario de marcas de clase (valores medios aproximados por tramo)

marca_clase = {
    'Menos o igual a 300 €': 150,
    'De 301 a 600 €': 450,
    'De 601 a 900 €': 750,
    'De 901 a 1.200 €': 1050,
    'De 1.201 a 1.800 €': 1500,
    'De 1.801 a 2.400 €': 2100,
    'De 2.401 a 3.000 €': 2700,
    'De 3.001 a 4.500 €': 3750,
    'De 4.501 a 6.000 €': 5250,
    'Más de 6.000 €': 6500,
    'No tienen ingresos de ningún tipo': 0,
    'NS/NC': np.nan
}

# Aplicar recodificación
df['INGRESOS_RMARCA'] = df['INGRESHOG'].map(marca_clase)
df['INGRESOS_RMARCA'] = df['INGRESOS_RMARCA'].astype(float)

# Verificar distribución
print("Distribución de INGRESOS_RMARCA:")
print(df['INGRESOS_RMARCA'].describe())

# --- 1. Definir variables ---
x_var = 'INGRESOS_RMARCA'
y_vars = [f'P3_{i}' for i in range(1, 13)]

# --- 2. Obtener etiquetas SPSS ---
etiquetas = {meta.column_names[i]: meta.column_labels[i] for i in range(len(meta.column_names))}

# --- 3. Crear figura con subplots ---
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

# --- 4. Dibujar los gráficos ---
for i, var in enumerate(y_vars):
    if var in df.columns:
        data = df[[x_var, var]].dropna()
        sns.regplot(
            data=data,
            x=x_var, y=var,
            scatter_kws={'alpha':0.6, 's':35},
            line_kws={'color':'red'},
            ci=95,
            ax=axes[i]
        )
        etiqueta_y = etiquetas.get(var, var)
        axes[i].set_title(f"{etiqueta_y} vs INGRESOS_RMARCA", fontsize=11)
        axes[i].set_xlabel("Ingreso mensual estimado (€)")
        axes[i].set_ylabel(etiqueta_y)
        axes[i].grid(True, linestyle='--', alpha=0.4)
    else:
        axes[i].set_visible(False)

# --- 5. Ajustar espaciado ---
plt.suptitle("Relación entre ingresos y percepciones sobre avances tecnológicos (P3_1 a P3_12)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# =============================================================================
#%% C.2. VAR MÉTRICA by MÉTRICA – Tabla y test: INGRESOS_RMARCA by P3_?
# =============================================================================

# --- 1. Variables ---
x_var = 'INGRESOS_RMARCA'
y_vars = [f'P3_{i}' for i in range(1, 13)]
variables = [x_var] + y_vars

# --- 2. Asegurar que son numéricas ---
df[variables] = df[variables].apply(pd.to_numeric, errors='coerce')

# --- 3. Correlaciones de Pearson ---
resultados = []

for var in y_vars:
    datos = df[[x_var, var]].dropna()
    if len(datos) > 1:
        r, p = stats.pearsonr(datos[x_var], datos[var])
        resultados.append({
            'Variable': var,
            'r de Pearson': round(r, 3),
            'p-valor': round(p, 4),
            'N': len(datos)
        })
    else:
        resultados.append({'Variable': var, 'r de Pearson': None, 'p-valor': None, 'N': len(datos)})

tabla_corr = pd.DataFrame(resultados)

# --- Reemplazar nombres por etiquetas SPSS (con corrección) ---
etiquetas = [
    meta.column_labels[meta.column_names.index(v)] if v in meta.column_names else v
    for v in tabla_corr['Variable']
]
tabla_corr['Etiqueta (SPSS)'] = etiquetas
tabla_corr = tabla_corr[['Etiqueta (SPSS)', 'r de Pearson', 'p-valor', 'N']]

print("Tabla de correlaciones entre INGRESOS_RMARCA y variables P3_1 a P3_12:")
print(tabla_corr)

# =============================================================================
# Pruebas de normalidad (Kolmogorov-Smirnov)
# =============================================================================

resultados_norm = []

for var in variables:
    datos = df[var].dropna()
    if len(datos) >= 2:
        z_datos = (datos - datos.mean()) / datos.std(ddof=0)
        stat, p_val = stats.kstest(z_datos, 'norm')
        resultados_norm.append({
            'Variable': var,
            'Estadístico KS': round(stat, 3),
            'p-valor': round(p_val, 4),
            'N': len(datos)
        })
    else:
        resultados_norm.append({'Variable': var, 'Estadístico KS': None, 'p-valor': None, 'N': len(datos)})

tabla_normalidad = pd.DataFrame(resultados_norm)

# --- Reemplazar nombres por etiquetas SPSS (con corrección) ---
etiquetas_norm = [
    meta.column_labels[meta.column_names.index(v)] if v in meta.column_names else v
    for v in tabla_normalidad['Variable']
]
tabla_normalidad['Etiqueta (SPSS)'] = etiquetas_norm
tabla_normalidad = tabla_normalidad[['Etiqueta (SPSS)', 'Estadístico KS', 'p-valor', 'N']]

print("Pruebas de normalidad (Kolmogorov-Smirnov):")
print(tabla_normalidad)

# =============================================================================
# Gráficos Q-Q
# =============================================================================

num_vars = len(variables)
cols = 5
rows = (num_vars // cols) + (num_vars % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
axes = axes.flatten()

for i, (var, etiqueta) in enumerate(zip(variables, etiquetas_norm)):
    datos = df[var].dropna()
    if len(datos) >= 3:
        sm.qqplot(datos, line='s', ax=axes[i])
        axes[i].set_title(etiqueta, fontsize=9)
        axes[i].grid(True, linestyle='--', alpha=0.5)
    else:
        axes[i].text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center')
        axes[i].set_title(etiqueta, fontsize=9)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# =============================================================================
# Correlaciones no paramétricas (Spearman y Kendall Tau)
# =============================================================================

resultados_np = []

for var in y_vars:
    datos = df[[x_var, var]].dropna()
    if len(datos) > 1:
        rho, p_rho = stats.spearmanr(datos[x_var], datos[var])
        tau, p_tau = stats.kendalltau(datos[x_var], datos[var])
        resultados_np.append({
            'Variable 1': x_var,
            'Variable 2': var,
            'Spearman ρ': round(rho, 3),
            'p-valor ρ': round(p_rho, 4),
            'Kendall τ': round(tau, 3),
            'p-valor τ': round(p_tau, 4),
            'N': len(datos)
        })
    else:
        resultados_np.append({
            'Variable 1': x_var,
            'Variable 2': var,
            'Spearman ρ': None,
            'p-valor ρ': None,
            'Kendall τ': None,
            'p-valor τ': None,
            'N': len(datos)
        })

tabla_corr_np = pd.DataFrame(resultados_np)

# --- Reemplazar nombres por etiquetas SPSS (con corrección) ---
etiquetas_map = {
    v: meta.column_labels[meta.column_names.index(v)] if v in meta.column_names else v
    for v in variables
}
tabla_corr_np['Variable 1'] = tabla_corr_np['Variable 1'].map(etiquetas_map)
tabla_corr_np['Variable 2'] = tabla_corr_np['Variable 2'].map(etiquetas_map)

print("Correlaciones no paramétricas (Spearman y Kendall Tau) con INGRESOS_RMARCA:")
print(tabla_corr_np)

# =============================================================================
#%% C.3. INGRESOS_RMARCA by P3_1 a P3_12 – Test para muestras emparejadas
# =============================================================================

# --- 1. Variables ---
variables = [f'P3_{i}' for i in range(1, 13)]
df[variables] = df[variables].apply(pd.to_numeric, errors='coerce')

# --- 2. Etiquetas SPSS con protección ---
etiquetas = {
    v: meta.column_labels[meta.column_names.index(v)] if v in meta.column_names else v
    for v in variables
}

# --- 3. Correlaciones de Spearman + gráficos ---
from itertools import combinations

resultados_spearman = []

for var1, var2 in combinations(variables, 2):
    datos = df[[var1, var2]].dropna()
    rho, p_val = stats.spearmanr(datos[var1], datos[var2])
    
    resultados_spearman.append({
        'Variable 1': etiquetas[var1],
        'Variable 2': etiquetas[var2],
        'Spearman ρ': round(rho, 3),
        'p-valor': round(p_val, 4),
        'N': len(datos)
    })

tabla_spearman = pd.DataFrame(resultados_spearman)
print("Correlaciones de Spearman entre variables P3_1 a P3_12:")
print(tabla_spearman)

# --- 4. Tabla de medias y desviaciones ---
medias = df[variables].mean()
desv = df[variables].std()
ns = df[variables].count()
etiquetas_lista = [etiquetas[v] for v in variables]

tabla_descriptiva = pd.DataFrame({
    'Variable': etiquetas_lista,
    'Media': medias.values,
    'Desviación típica': desv.values
}).round(2)

print("Tabla de medias y desviaciones:")
print(tabla_descriptiva)

# --- 5. Gráfico de medias con IC 95% ---
plt.figure(figsize=(10,6))
conf_int = 1.96 * desv.values / np.sqrt(ns.values)
plt.errorbar(x=etiquetas_lista, y=medias.values, yerr=conf_int, fmt='o', capsize=5, color='blue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Media ± IC 95%')
plt.title('Medias de variables P3_1 a P3_12')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- 6. Test t para muestras emparejadas ---
pares = [(variables[i], variables[j]) for i in range(len(variables)) for j in range(i+1, len(variables))]
resultados_t = []

for v1, v2 in pares:
    datos = df[[v1, v2]].dropna()
    t_stat, p_val = stats.ttest_rel(datos[v1], datos[v2])
    resultados_t.append({
        'Variable 1': etiquetas[v1],
        'Variable 2': etiquetas[v2],
        't': round(t_stat, 3),
        'p-valor': round(p_val, 4),
        'N': len(datos)
    })

tabla_t = pd.DataFrame(resultados_t)
print("\nTest t para muestras emparejadas:")
print(tabla_t)

# --- 7. Test de Wilcoxon (no paramétrico) ---
resultados_w = []

for v1, v2 in pares:
    datos = df[[v1, v2]].dropna()
    stat, p_val = stats.wilcoxon(datos[v1], datos[v2])
    resultados_w.append({
        'Variable 1': etiquetas[v1],
        'Variable 2': etiquetas[v2],
        'Wilcoxon W': round(stat, 3),
        'p-valor': round(p_val, 4),
        'N': len(datos)
    })

tabla_w = pd.DataFrame(resultados_w)
print("\nTest de Wilcoxon (muestras emparejadas, no paramétrico):")
print(tabla_w)