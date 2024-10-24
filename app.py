import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def monte_carlo_simulation(S0, mu, sigma, time_unit, num_periods, num_simulations):
    # Definindo o intervalo de tempo (dt) com base na unidade de tempo escolhida
    if time_unit == 'Dia':
        dt = 1/252  # Aproximadamente 252 dias úteis em um ano
    elif time_unit == 'Semana':
        dt = 1/52  # 52 semanas em um ano
    elif time_unit == 'Mês':
        dt = 1/12  # 12 meses em um ano
    elif time_unit == 'Ano':
        dt = 1  # 1 ano

    # Simulação de Monte Carlo
    np.random.seed(42)  # Para reprodutibilidade
    prices = np.zeros((num_simulations, num_periods + 1))

    # Preço inicial para todas as simulações
    prices[:, 0] = S0

    for i in range(num_simulations):
        for t in range(1, num_periods + 1):
            epsilon = np.random.normal(0, 1)
            # Calculando a mudança de preço usando a fórmula discreta
            delta_S = (mu * dt * prices[i, t-1]) + (sigma * np.sqrt(dt) * prices[i, t-1] * epsilon)
            # Atualizando o preço da ação
            prices[i, t] = prices[i, t-1] + delta_S

    return prices

# Página principal do aplicativo
st.title("Simulação de Monte Carlo para Preços de Ações")

# Inicializa o estado do botão se não estiver presente
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = True

# Inputs do usuário com validações
S0 = st.number_input("Preço inicial (S0):", value=100.0, step=1.0, min_value=1.0, max_value=10000.0)
mu = st.number_input("Retorno anual (mu):", value=0.15, step=0.01, max_value=1.0)
sigma = st.number_input("Volatilidade (sigma):", value=0.3, step=0.01, max_value=1.0)
time_unit = st.selectbox("Unidade de tempo:", ['Dia', 'Semana', 'Mês', 'Ano'])
num_periods = st.number_input("Número de períodos:", value=10, min_value=1, max_value=10000)
num_simulations = st.number_input("Número de simulações:", value=100, min_value=1, max_value=100000)

# Verificação dos limites
max_periods = {
    'Dia': 252 * 5,
    'Semana': 52 * 5,
    'Mês': 12 * 5,
    'Ano': 5
}

# Botão de execução da simulação
execute_button = st.button("Executar Simulação")

if execute_button:
    if num_periods > max_periods[time_unit] or S0>10000 or num_simulations>100000 or sigma>1 or mu>1:
        pass
    # Executando a simulação
    prices = monte_carlo_simulation(S0, mu, sigma, time_unit, int(num_periods), int(num_simulations))
    
    # Coletando os preços finais após os períodos simulados
    final_prices = prices[:, -1]

    # Calculando as estatísticas
    mean_price = np.mean(final_prices)
    std_dev_price = np.std(final_prices)
    min_price = np.min(final_prices)
    max_price = np.max(final_prices)
    median_price = np.median(final_prices)

        # Criando colunas para exibir gráficos lado a lado
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução do preço da ação - Primeira Simulação")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(num_periods + 1), prices[0], lw=2, color='blue')
        ax.set_title('Evolução do preço da ação - Primeira Simulação')
        ax.set_xlabel(f'{time_unit}(s)')
        ax.set_ylabel('Preço da ação')
        st.pyplot(fig)

    with col2:
        st.subheader("Evolução do preço da ação ao longo dos períodos")
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(num_simulations):
            ax.plot(range(num_periods + 1), prices[i], lw=0.8, alpha=0.6)
        ax.set_title('Evolução do preço da ação')
        ax.set_xlabel(f'{time_unit}(s)')
        ax.set_ylabel('Preço da ação')
        st.pyplot(fig)

    st.subheader("Distribuição dos preços finais após os períodos simulados")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(final_prices, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('Distribuição dos preços finais')
    ax.set_xlabel('Preço da ação')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)


    # Estatísticas descritivas
    st.subheader("Estatísticas Descritivas dos Preços Finais")
    df = pd.DataFrame(final_prices, columns=["Preços Finais"])
    descriptive_stats = df.describe()
    st.write(descriptive_stats)

    # Observação final
    st.markdown("""
    **Observação:**
    Prezad@ usuári@, as informações apresentadas não devem ser interpretadas como recomendação de qualquer natureza, para quaisquer tipos de investimentos, estratégias ou transações comerciais. A única finalidade é a didática, ou seja, apoiar o processo de ensino-aprendizagem-avaliação de tópicos de análise de investimentos.""")
