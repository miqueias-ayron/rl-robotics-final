<img width="664" height="564" alt="gymnasium_taxidriver" src="https://github.com/user-attachments/assets/6eec50e4-e0c7-4a26-a754-12d031e0142f" />

## Team
- Brenda Lima
- Miqueias Ayron
- Pedro Pereira
  
# Q-Learning x SARSA no Taxi-v4

Projeto de introdução a Reinforcement Learning: agentes treinados com **Q-Learning** e **SARSA** no ambiente `Taxi-v4` do Gymnasium, nas versões determinística e não determinística (`is_rainy=True`, `rainy_probability=0.6`).

**O relatório com os resultados e a discussão está em [`notebooks/RELATORIO.ipynb`](notebooks/RELATORIO.ipynb).**

## Estrutura

```
algorithms/
  QLearning.py, Sarsa.py           # implementações tabulares dos dois algoritmos
notebooks/
  taxidriver_deterministic/        # treino e avaliação no ambiente determinístico
    QLearningTaxiDriver.ipynb
    SarsaTaxiDriver.ipynb
  taxidriver_stochastic/           # treino e avaliação no ambiente com chuva
    QLearningTaxiDriver.ipynb
    SarsaTaxiDriver.ipynb
  RELATORIO.ipynb                  # relatório: curvas, tabelas, discussão e conclusões
  NUMEROS.ipynb                    # calcula os números citados no texto do relatório
  report_utils.py                  # funções de análise usadas pelos dois notebooks acima
data/
  rewards/                         # recompensa por episódio dos 5 treinos de cada experimento
  q_tables/                        # Q-table final de cada um dos 5 treinos
  deterministic/, stochastic/      # Q-table da última execução, usada pelos notebooks de treino
```

## Como executar

Requisitos: Python 3.13 com `gymnasium` (1.4), `numpy`, `pandas`, `matplotlib`, `pygame` (renderização) e `jupyter`.

```bash
pip install "gymnasium[toy-text]" numpy pandas matplotlib jupyter
```

1. **(Opcional) Treinar os agentes:** execute os quatro notebooks de `notebooks/taxidriver_deterministic/` e `notebooks/taxidriver_stochastic/`. Cada um treina o agente 5 vezes (10.000 episódios cada) e salva os resultados em `data/`.
2. **Gerar o relatório:** execute `notebooks/RELATORIO.ipynb`. Os dados dos treinos já estão versionados em `data/`, então o relatório roda direto, sem treinar de novo.
3. **(Opcional) Conferir os números do texto:** execute `notebooks/NUMEROS.ipynb`. Cada célula calcula, a partir dos mesmos dados, um número citado no relatório e indica em que seção ele é usado.

## Hiperparâmetros

Os mesmos para os dois algoritmos e os dois ambientes: α = 0.5, γ = 0.99, ε de 0.2 a 0.05 (decaimento de 0.99 por episódio), 10.000 episódios, 5 treinos por configuração.

## Principais resultados

A política final de cada um dos 5 treinos foi testada sem exploração em 100 episódios:

| Configuração | Timesteps médios (± desvio) | Taxa de sucesso |
|---|---|---|
| Q-Learning, determinístico | 13.4 ± 2.9 | 100% |
| SARSA, determinístico | 13.9 ± 3.3 | 81.8% |
| Q-Learning, com chuva | 27.1 ± 11.4 | 83.8% |
| SARSA, com chuva | 30.0 ± 15.5 | 83.0% |
| Q-Learning treinado sem chuva, testado com chuva | 26.6 ± 10.6 | 97.8% |

- O **Q-Learning** aprende mais rápido e, no ambiente determinístico, tem uma política mais confiável que a do SARSA.
- A **chuva** dobra o número de passos por entrega e aumenta a variação dos resultados para os dois algoritmos.
- A política aprendida **sem chuva continua válida com chuva**, porque a melhor rota é a mesma.

A discussão completa está no [relatório](notebooks/RELATORIO.ipynb).
