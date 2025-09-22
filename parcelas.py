import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# =============================
# Función de fitness
# =============================
def calcular_fitness(individuo, valores, pesos, C, W):
    valor_total = np.sum(individuo * valores)
    peso_total = np.sum(individuo * pesos)
    if peso_total > W or valor_total > C:
        return 0
    return valor_total

# =============================
# Algoritmo Genético
# =============================
def algoritmo_genetico(valores, pesos, C, W, num_generaciones=50, tam_poblacion=30, prob_mutacion=0.1):
    N = len(valores)
    poblacion = [np.random.randint(0, 2, size=N) for _ in range(tam_poblacion)]
    historico_mejor = []
    for gen in range(num_generaciones):
        fitness = [calcular_fitness(ind, valores, pesos, C, W) for ind in poblacion]
        historico_mejor.append(max(fitness))
        padres = []
        for _ in range(tam_poblacion):
            i, j = random.sample(range(tam_poblacion), 2)
            padres.append(poblacion[i] if fitness[i] > fitness[j] else poblacion[j])
        hijos = []
        for i in range(0, tam_poblacion, 2):
            padre1, padre2 = padres[i], padres[i+1]
            punto = random.randint(1, N-1)
            hijo1 = np.concatenate((padre1[:punto], padre2[punto:]))
            hijo2 = np.concatenate((padre2[:punto], padre1[punto:]))
            hijos.extend([hijo1, hijo2])
        for hijo in hijos:
            if random.random() < prob_mutacion:
                pos = random.randint(0, N-1)
                hijo[pos] = 1 - hijo[pos]
        poblacion = hijos
    return historico_mejor

# =============================
# Recocido Simulado
# =============================
def recocido_simulado(valores, pesos, C, W, temp_inicial=100, temp_final=1, alpha=0.95, max_iter=500):
    N = len(valores)
    sol_actual = np.random.randint(0, 2, size=N)
    fit_actual = calcular_fitness(sol_actual, valores, pesos, C, W)
    mejor = fit_actual
    historico = [fit_actual]
    temp = temp_inicial
    while temp > temp_final:
        for _ in range(max_iter):
            vecino = sol_actual.copy()
            pos = random.randint(0, N-1)
            vecino[pos] = 1 - vecino[pos]
            fit_vecino = calcular_fitness(vecino, valores, pesos, C, W)
            if fit_vecino > fit_actual or random.random() < np.exp((fit_vecino - fit_actual) / temp):
                sol_actual, fit_actual = vecino, fit_vecino
                mejor = max(mejor, fit_actual)
        historico.append(mejor)
        temp *= alpha
    return historico

# =============================
# Búsqueda Tabú
# =============================
def busqueda_tabu(valores, pesos, C, W, max_iter=200, tabu_tam=10):
    N = len(valores)
    sol_actual = np.random.randint(0, 2, size=N)
    fit_actual = calcular_fitness(sol_actual, valores, pesos, C, W)
    mejor = sol_actual.copy()
    mejor_fit = fit_actual
    tabu_list = []
    historico = [fit_actual]
    for _ in range(max_iter):
        vecinos = []
        for i in range(N):
            vecino = sol_actual.copy()
            vecino[i] = 1 - vecino[i]
            vecinos.append(vecino)
        fitness_vecinos = [(v, calcular_fitness(v, valores, pesos, C, W)) for v in vecinos]
        fitness_vecinos.sort(key=lambda x: x[1], reverse=True)
        for v, f in fitness_vecinos:
            if list(v) not in tabu_list:
                sol_actual, fit_actual = v, f
                break
        if fit_actual > mejor_fit:
            mejor, mejor_fit = sol_actual.copy(), fit_actual
        tabu_list.append(list(sol_actual))
        if len(tabu_list) > tabu_tam:
            tabu_list.pop(0)
        historico.append(mejor_fit)
    return historico

# =============================
# Interfaz Streamlit
# =============================
st.title("Simulación de Cosecha de Caña de Azúcar")
st.write("Optimización con Metaheurísticas: Genético, Recocido Simulado y Búsqueda Tabú")

# Parámetros del problema
N = st.slider("Número de parcelas", 5, 20, 12)
num_trabajadores = st.slider("Número de trabajadores", 1, 10, 6)

# Generar datos
valores = np.random.randint(100, 1201, size=N)
pesos = np.round(np.random.uniform(1, 10, size=N), 1)
W = num_trabajadores * 40

df_parcelas = pd.DataFrame({
    "Parcela": range(1, N+1),
    "Valor (kg)": valores,
    "Peso (horas)": pesos
})

st.subheader("Parcelas iniciales")
st.dataframe(df_parcelas)
st.write(f"**Capacidad de trabajo total (horas):** {W}")

# Selección de algoritmo
algoritmo = st.selectbox("Seleccione el algoritmo", ["Genético", "Recocido Simulado", "Búsqueda Tabú"])

if st.button("Ejecutar optimización"):
    # 🔹 Aquí se genera C en cada ejecución
    C = random.randint(1000, 2500)
    st.write(f"**Capacidad vehículo (generada esta corrida):** {C}")

    if algoritmo == "Genético":
        historico = algoritmo_genetico(valores, pesos, C, W)
    elif algoritmo == "Recocido Simulado":
        historico = recocido_simulado(valores, pesos, C, W)
    else:
        historico = busqueda_tabu(valores, pesos, C, W)

    st.subheader("Evolución del Fitness")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(historico, label=algoritmo, color="blue")
    ax.set_xlabel("Iteraciones / Generaciones")
    ax.set_ylabel("Mejor Fitness")
    ax.set_title(f"Evolución del Fitness - {algoritmo}")
    ax.legend()
    st.pyplot(fig)
