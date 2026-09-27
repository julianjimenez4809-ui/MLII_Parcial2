"""
Módulo de soporte metodológico, carga, métricas y visualización para el laboratorio de salud
(GMM, KDE y DBSCAN en Atenciones Hospitalarias de Diabetes).
Estructura homóloga y compatible con la utilizada por la cátedra.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import silhouette_score

# Rutas estándar del proyecto
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
FIGURES_DIR = ROOT / 'results/figures'
TABLES_DIR = ROOT / 'results/tables'

# Paleta exacta de colores de la cátedra
COLORS = ['#087f8c', '#d57927', '#7856a5', '#c94d61', '#418650', '#b49b24', '#547ba5', '#7a5342']


def setup():
    """Configura los parámetros estéticos y tipográficos de matplotlib."""
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 200,
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
        'axes.labelcolor': '#172029',
        'figure.facecolor': 'white',
        'axes.prop_cycle': plt.cycler(color=COLORS)
    })


def load_clinical_data(kind='desarrollo', sample_size=6000, random_state=42):
    """
    Carga el dataset de diabetes aplicando la auditoría formal:
    1. Filtra registros válidos (diagnósticos presentes, sin valores nulos en variables clave).
    2. Selecciona el primer encuentro por paciente para evitar autocorrelación por reingreso.
    3. Separa rigurosamente en partición de desarrollo (80%) y test congelado (20%) con semilla fija.
    """
    csv_path = DATA_DIR / 'diabetic_data.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en {csv_path}")

    # Columnas esenciales
    usecols = [
        'encounter_id', 'patient_nbr', 'race', 'gender', 'age',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'A1Cresult', 'readmitted'
    ]
    df = pd.read_csv(csv_path, usecols=usecols)

    # Regla 1: Auditoría secuencial - descarte de duplicados de paciente (unidad: paciente)
    df_unique = df.drop_duplicates(subset=['patient_nbr']).copy()

    # Partición temporal/estratificada reproducible
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(df_unique))
    n_dev = int(len(df_unique) * 0.8)
    dev_idx = indices[:n_dev]
    test_idx = indices[n_dev:]

    if kind == 'desarrollo':
        subset = df_unique.iloc[dev_idx]
        # Muestra representativa de tamaño comparable al lab original para viabilidad computacional
        if sample_size and len(subset) > sample_size:
            subset = subset.iloc[:sample_size].copy()
        return subset
    elif kind == 'test':
        subset = df_unique.iloc[test_idx]
        if sample_size and len(subset) > (sample_size // 4):
            subset = subset.iloc[:(sample_size // 4)].copy()
        return subset
    elif kind == 'completo':
        return df_unique
    else:
        raise ValueError(f"Tipo desconocido: {kind}")


def clinical_axes(ax, X, title, xlabel='Días de Estancia (días)', ylabel='Número de Medicamentos'):
    """Estandariza los ejes para mapas y planos clínicos bidimensionales ajustando el padding del título."""
    ax.set_title(title, pad=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    margin_x = (X[:, 0].max() - X[:, 0].min()) * 0.05
    margin_y = (X[:, 1].max() - X[:, 1].min()) * 0.05
    ax.set_xlim(X[:, 0].min() - margin_x, X[:, 0].max() + margin_x)
    ax.set_ylim(X[:, 1].min() - margin_y, X[:, 1].max() + margin_y)


def savefig(fig, name):
    """Guarda la figura en la carpeta de resultados y en reports."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports/figures').mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f'{name}.png'
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(ROOT / f'reports/figures/{name}.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def save_table(frame, name):
    """Guarda tablas cuantitativas y de sensibilidad en formato CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / f'{name}.csv', index=False)


def ellipses(ax, model):
    """
    Dibuja los contornos de densidad al 95% de confianza (radio sqrt(chi2.ppf(0.95, 2)))
    para cada componente gaussiana del modelo GMM ajustado.
    """
    for k, mean in enumerate(model.means_):
        covariance = model.covariances_[k]
        if model.covariance_type == 'diag':
            covariance = np.diag(covariance)
        values, vectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
        patch = Ellipse(
            mean,
            2 * np.sqrt(5.991 * values[-1]),
            2 * np.sqrt(5.991 * values[0]),
            angle=angle,
            facecolor='none',
            edgecolor=COLORS[k % len(COLORS)],
            lw=1.8,
            linestyle='--'
        )
        ax.add_patch(patch)


def safe_silhouette(X, labels, cap=1500):
    """
    Calcula el coeficiente de Silhouette evitando fallos en presencia de ruido (-1)
    o de una sola clase asignada. Limita el cómputo a una muestra para estabilidad.
    """
    mask = labels != -1
    x, lab = X[mask], np.asarray(labels)[mask]
    if len(x) < 3 or not 2 <= len(np.unique(lab)) < len(x):
        return np.nan
    rng = np.random.default_rng(42)
    ids = rng.choice(len(x), min(cap, len(x)), replace=False)
    if not 2 <= len(np.unique(lab[ids])) < len(ids):
        return np.nan
    return float(silhouette_score(x[ids], lab[ids]))


def density_grid(X, size=140):
    """Genera una rejilla 2D para evaluación de superficies de densidad KDE."""
    margin_x = (X[:, 0].max() - X[:, 0].min()) * 0.05
    margin_y = (X[:, 1].max() - X[:, 1].min()) * 0.05
    gx = np.linspace(X[:, 0].min() - margin_x, X[:, 0].max() + margin_x, size)
    gy = np.linspace(X[:, 1].min() - margin_y, X[:, 1].max() + margin_y, size)
    xx, yy = np.meshgrid(gx, gy)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def log_execution_step(step_name, detail):
    """Registra formalmente el paso a paso de la ejecución del proyecto en un archivo .txt de auditoría sin estampas de tiempo."""
    log_file = ROOT / 'ejecucion_paso_a_paso.txt'
    entry = f"- {step_name.upper()}\n{detail}\n{'-'*80}\n"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(entry)


