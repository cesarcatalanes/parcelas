import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# Funciones de inicialización
def inicializar_problema(N, trabajadores,
                         min_kg, max_kg,
                         min_h, max_h,
                         min_C, max_C):
    V = np.random.randint(min_kg, max_kg + 1, N)
    P = np.round(np.random.uniform(min_h, max_h, N), 2)
    C = np.random.randint(min_C, max_C + 1)
    W = trabajadores * 40  # horas totales
    return V, P, C, W

def fitness(X, V, P, C, W):
    total_v = np.sum(V * X)
    total_p = np.sum(P * X)
    if total_p > W or total_v > C:
        return 0
    return total_v


# Algoritmos
def genetico(V, P, C, W, pop_size=50, generations=200, mutation_rate=0.05):
    N = len(V)
    poblacion = np.random.randint(0, 2, (pop_size, N))
    mejor = None
    historial = []

    for _ in range(generations):
        fit = np.array([fitness(ind, V, P, C, W) for ind in poblacion])
        historial.append(fit.max())
        padres = poblacion[np.argsort(fit)[-pop_size//2:]]
        hijos = []
        for _ in range(pop_size):
            p1, p2 = padres[np.random.randint(len(padres), size=2)]
            corte = np.random.randint(1, N-1)
            child = np.concatenate([p1[:corte], p2[corte:]])
            mask = np.random.rand(N) < mutation_rate
            child[mask] = 1 - child[mask]
            hijos.append(child)
        poblacion = np.array(hijos)
        if mejor is None or fit.max() > fitness(mejor, V, P, C, W):
            mejor = poblacion[fit.argmax()]
    return mejor, historial

def recocido_simulado(V, P, C, W, T=1000, alpha=0.95, max_iter=2000):
    N = len(V)
    actual = np.random.randint(0, 2, N)
    mejor = np.copy(actual)
    historial = []

    for _ in range(max_iter):
        vecino = np.copy(actual)
        idx = np.random.randint(N)
        vecino[idx] = 1 - vecino[idx]
        delta = fitness(vecino, V, P, C, W) - fitness(actual, V, P, C, W)
        if delta > 0 or np.random.rand() < math.exp(delta / T):
            actual = vecino
            if fitness(actual, V, P, C, W) > fitness(mejor, V, P, C, W):
                mejor = np.copy(actual)
        historial.append(fitness(mejor, V, P, C, W))
        T *= alpha
        if T < 1e-3:
            break
    return mejor, historial

def busqueda_tabu(V, P, C, W, max_iter=200, tabu_tam=10):
    N = len(V)
    actual = np.random.randint(0, 2, N)
    mejor = np.copy(actual)
    historial = []
    tabu = []

    for _ in range(max_iter):
        vecinos = []
        for j in range(N):
            vecino = np.copy(actual)
            vecino[j] = 1 - vecino[j]
            if not any((vecino == t).all() for t in tabu):
                vecinos.append(vecino)
        if not vecinos:
            break
        vecinos_fit = [fitness(v, V, P, C, W) for v in vecinos]
        mejor_vecino = vecinos[np.argmax(vecinos_fit)]
        actual = mejor_vecino
        if fitness(actual, V, P, C, W) > fitness(mejor, V, P, C, W):
            mejor = np.copy(actual)
        historial.append(fitness(mejor, V, P, C, W))
        tabu.append(actual)
        if len(tabu) > tabu_tam:
            tabu.pop(0)
    return mejor, historial


# Interfaz Streamlit
st.title(" Programa de Cosecha - Metaheurísticas")

st.sidebar.header("Parámetros del Problema")
N = st.sidebar.slider("Número de parcelas (N)", 5, 40, 20)
trabajadores = st.sidebar.slider("Trabajadores", 1, 20, 8)
min_kg = st.sidebar.slider("Kg mínimos por parcela", 50, 800, 100)
max_kg = st.sidebar.slider("Kg máximos por parcela", 200, 1200, 600)
min_h  = st.sidebar.slider("Horas mínimas por parcela", 1, 5, 1)
max_h  = st.sidebar.slider("Horas máximas por parcela", 3, 12, 6)
min_C  = st.sidebar.slider("Capacidad mínima vehículo (kg)", 500, 2500, 1500)
max_C  = st.sidebar.slider("Capacidad máxima vehículo (kg)", 1000, 5000, 2500)

algoritmo = st.selectbox("Seleccione Algoritmo",
                         ["Genético", "Recocido Simulado", "Búsqueda Tabú"])

if st.button("Generar y Resolver"):
    V, P, C, W = inicializar_problema(N, trabajadores,
                                      min_kg, max_kg,
                                      min_h, max_h,
                                      min_C, max_C)

    st.subheader("📊 Parámetros Iniciales")
    st.write(f"Capacidad de trabajo (W): **{W} horas**")
    st.write(f"Capacidad vehículo (C): **{C} kg**")
    st.dataframe({"Parcela": range(1, N+1), "Caña (kg)": V, "Horas": P})

    # Resolver con algoritmo elegido
    if algoritmo == "Genético":
        mejor, historial = genetico(V, P, C, W)
    elif algoritmo == "Recocido Simulado":
        mejor, historial = recocido_simulado(V, P, C, W)
    else:
        mejor, historial = busqueda_tabu(V, P, C, W)

    mejor_fitness = fitness(mejor, V, P, C, W)
    st.success(f"✅ Mejor Fitness: **{mejor_fitness} kg**")

    # Parcelas seleccionadas
    idx = np.where(mejor == 1)[0]
    st.subheader("Parcelas Seleccionadas")
    if len(idx) > 0:
        st.dataframe({
            "Parcela": idx + 1,
            "Caña (kg)": V[idx],
            "Horas": P[idx]
        })
    else:
        st.warning("No se seleccionó ninguna parcela (revisa restricciones).")

    # Gráfica de evolución del fitness
    st.subheader("Evolución del Fitness")
    fig, ax = plt.subplots()
    ax.plot(historial, color="green")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fitness (kg de caña)")
    ax.set_title(f"Evolución del Fitness - {algoritmo}")
    st.pyplot(fig)
