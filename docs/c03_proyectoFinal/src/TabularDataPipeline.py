import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline

class TabularDataPipeline:
    """
    Un pipeline completo para preprocesar, balancear y submuestrear datos tabulares.

    Pasos:
    1. Preprocesamiento:
        - Estandariza columnas numéricas (StandardScaler).
        - Codifica como enteros columnas categóricas (OrdinalEncoder), necesario 
          para modelos con capas de Embedding.
        - Deja intactas las columnas binarias.
    2. Balanceo:
        - Sobremuestreo de la clase minoritaria con SMOTE.
        - Limpieza de enlaces Tomek para eliminar ruido.
    3. Submuestreo:
        - Extrae un subconjunto final de 16,000 observaciones (8k por clase).
    """
    def __init__(self, 
                 numeric_cols: list, 
                 categorical_cols: list, 
                 binary_cols: list,
                 random_state: int = 42):
        """
        Inicializa el pipeline con las listas de columnas.

        Args:
            numeric_cols (list): Lista de nombres de columnas numéricas.
            categorical_cols (list): Lista de nombres de columnas categóricas.
            binary_cols (list): Lista de nombres de columnas binarias.
            random_state (int): Semilla para reproducibilidad.
        """
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.random_state = random_state
        self.target_col = None
        
        # El preprocesador se ajustará a los datos de entrada
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_cols),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.categorical_cols),
                ('bin', 'passthrough', self.binary_cols)
            ],
            remainder='drop'
        )

    def process(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Aplica el pipeline completo al DataFrame de entrada.

        Args:
            df (pd.DataFrame): El DataFrame inicial.
            target_col (str): El nombre de la columna objetivo.

        Returns:
            pd.DataFrame: Un nuevo DataFrame limpio, balanceado y submuestreado.
        """
        self.target_col = target_col
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # --- 1. Preprocesamiento ---
        print("Paso 1: Aplicando preprocesamiento (Scaler y OrdinalEncoder)...")
        X_processed = self.preprocessor.fit_transform(X)
        
        # Reconstruir el DataFrame con nombres de columna correctos
        processed_cols = self.numeric_cols + self.categorical_cols + self.binary_cols
        X_processed = pd.DataFrame(X_processed, columns=processed_cols)

        # --- 2. Balanceo con SMOTE + Tomek Links ---
        print("Paso 2: Aplicando SMOTE y Tomek Links para balanceo...")
        smote = SMOTE(random_state=self.random_state, sampling_strategy='auto')
        tomek = TomekLinks(sampling_strategy='auto')
        
        # imblearn requiere np.array, no DataFrames
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)
        
        print(f"Tamaño después de SMOTE+Tomek: {X_resampled.shape[0]} observaciones.")
        
        resampled_df = pd.DataFrame(X_resampled, columns=processed_cols)
        resampled_df[target_col] = y_resampled

        # --- 3. Submuestreo a 16k (8k por clase) ---
        print("Paso 3: Realizando submuestreo estratificado a 16k (8k por clase)...")
        
        class_0 = resampled_df[resampled_df[target_col] == 0]
        class_1 = resampled_df[resampled_df[target_col] == 1]
        
        # Verificar si hay suficientes muestras
        if len(class_0) < 8000 or len(class_1) < 8000:
            raise ValueError(f"No hay suficientes muestras para el submuestreo. "
                             f"Clase 0: {len(class_0)}, Clase 1: {len(class_1)}. "
                             f"Se necesitan al menos 8000 por clase.")

        sample_0 = class_0.sample(n=8000, random_state=self.random_state)
        sample_1 = class_1.sample(n=8000, random_state=self.random_state)
        
        final_df = pd.concat([sample_0, sample_1]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print("¡Pipeline completado!")
        print(f"Tamaño final del DataFrame: {final_df.shape}")
        print("Distribución de clases final:")
        print(final_df[target_col].value_counts())
        
        return final_df