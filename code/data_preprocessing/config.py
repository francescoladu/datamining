from pathlib import Path


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = Path("phishing_clean.csv")

TARGET_COLUMN = "Result"

# Colonne che possono essere state salvate accidentalmente
# durante la creazione del CSV.
INDEX_COLUMNS = [
    "index",
    "Unnamed: 0",
]


# ============================================================
# CLASSI
# ============================================================

PHISHING_LABEL = -1
LEGITIMATE_LABEL = 1


# ============================================================
# RIPRODUCIBILITÀ
# ============================================================

RANDOM_STATE = 42


# ============================================================
# FILE DI OUTPUT
# ============================================================

CORRELATION_MATRIX_PATH = Path(
    "pearson_correlation_matrix.csv"
)

CORRELATION_HEATMAP_PDF_PATH = Path(
    "pearson_correlation_heatmap.pdf"
)

CORRELATION_HEATMAP_PNG_PATH = Path(
    "pearson_correlation_heatmap.png"
)