import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
from .kennel import BREEDS  # Importamos nuestro catálogo de modelos

class MiloBrain:
    def __init__(self, X, y, task="classification"):
        self.X = X
        self.y = y
        self.task = task
        self.history = []  # Aquí guardaremos lo que aprendamos
        self.best_score = -np.inf
        self.best_model = None
        self.best_params = {}
        self.best_algo_name = ""

    def _get_random_params(self, breed_name):
        """
        Elige aleatoriamente una configuración del catálogo para una raza específica.
        """
        breed_info = BREEDS[breed_name]
        raw_params = breed_info['params']
        
        selected_params = {}
        for param, options in raw_params.items():
            # De la lista de opciones [10, 20, 30], elige una al azar
            selected_params[param] = random.choice(options)
            
        return selected_params

    def think(self, n_trials=10):
        """
        El ciclo principal de pensamiento.
        Prueba 'n_trials' modelos diferentes y encuentra el mejor.
        """
        print(f"🧠 MiLo está pensando... Probando {n_trials} estrategias.")
        
        available_breeds = list(BREEDS.keys())

        for i in range(n_trials):
            try:
                # 1. Elegir un algoritmo al azar (ej. 'random_forest')
                algo_name = random.choice(available_breeds)
                
                # 2. Elegir hiperparámetros al azar para ese algoritmo
                params = self._get_random_params(algo_name)
                
                # 3. Instanciar el modelo real
                model_class = BREEDS[algo_name]['model']
                model = model_class(**params)
                
                # 4. Evaluar (Usamos Cross Validation de 3 pliegues)
                # 'accuracy' es el default para clasificación
                cv_scores = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy')
                mean_score = cv_scores.mean()
                
                # 5. Guardar en la bitácora
                self.history.append({
                    "trial_id": i,
                    "algorithm": algo_name,
                    "score": mean_score,
                    "params": params
                })

                # 6. ¿Es este el mejor hasta ahora?
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model
                    self.best_params = params
                    self.best_algo_name = algo_name
                    print(f"   🌟 ¡Nuevo récord! {algo_name} -> Acc: {mean_score:.4f}")
            
            except Exception as e:
                print(f"   💥 El intento {i} falló: {e}")

        print("🏁 Pensamiento terminado.")
        
        return self._generate_report()

    def _generate_report(self):
        """Empaqueta los resultados en un formato bonito"""
        return {
            "best_model": self.best_model,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "history": pd.DataFrame(self.history).sort_values(by="score", ascending=False)
        }