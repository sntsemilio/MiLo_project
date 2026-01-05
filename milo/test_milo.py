import pandas as pd
from sklearn.datasets import load_iris, load_wine
import milo  # Al importar esto, se activa el accessor 'df.milo'

def test_con_iris():
    print("\n" + "="*50)
    print("🌸 TEST 1: Dataset IRIS (Clasificación de Flores)")
    print("="*50)
    
    # 1. Cargar datos de prueba
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target  # Esta es la columna a predecir
    
    print(f"📊 Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. LLAMAR A MILO (La Magia)
    # Le pedimos que haga 15 intentos para encontrar el mejor modelo
    resultado = df.milo.serve(target_col="species", n_trials=15)
    
    # 3. Mostrar resultados
    print("\n🏆 GANADOR DEL TORNEO:")
    print(f"   Modelo: {resultado['best_algo_name'].upper()}")
    print(f"   Accuracy: {resultado['best_score']:.2%}")
    print(f"   Parámetros: {resultado['best_params']}")
    
    print("\n📜 Top 3 Intentos:")
    print(resultado['history'][['algorithm', 'score', 'params']].head(3))

def test_con_wine():
    print("\n" + "="*50)
    print("🍷 TEST 2: Dataset WINE (Clasificación de Vinos)")
    print("="*50)
    
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['quality'] = data.target
    
    # Prueba rápida con solo 5 intentos
    resultado = df.milo.serve("quality", n_trials=5)
    
    print(f"\n🏆 Mejor Vino-Modelo: {resultado['best_algo_name']} ({resultado['best_score']:.2%})")

if __name__ == "__main__":
    # Ejecutar los tests
    try:
        test_con_iris()
        test_con_wine()
        print("\n✅ ¡TODO FUNCIONÓ PERFECTAMENTE! MiLo está vivo.")
    except Exception as e:
        print(f"\n❌ Algo salió mal: {e}")