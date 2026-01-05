import pandas as pd
from .core import MiloBrain

@pd.api.extensions.register_dataframe_accessor("milo")
class MiloAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj
        self._validate()

    def _validate(self):
        if self._df.empty:
            raise AttributeError("¡MiLo no puede aprender de un plato vacío! (DataFrame vacío)")

    def serve(self, target_col, n_trials=10):
        """
        Punto de entrada principal.
        
        Args:
            target_col (str): Nombre de la columna que quieres predecir.
            n_trials (int): Cuántos modelos diferentes probar.
        
        Returns:
            dict: Reporte con el mejor modelo, score y el historial completo.
        """
        # 1. Validar que la columna exista
        if target_col not in self._df.columns:
            raise ValueError(f"⚠️ La columna '{target_col}' no existe en tus datos.")
            
        # 2. Separar Features (X) y Target (y)
        X = self._df.drop(columns=[target_col])
        y = self._df[target_col]
        
        # 3. Despertar al cerebro
        # (Aquí asumimos clasificación por defecto por ahora)
        brain = MiloBrain(X, y, task="classification")
        
        # 4. Iniciar la búsqueda y devolver resultados
        return brain.think(n_trials=n_trials)