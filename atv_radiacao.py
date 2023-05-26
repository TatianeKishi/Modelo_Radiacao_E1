#!/usr/bin/env python
# coding: utf-8

# ###### Importando bibliotecas 

# In[ ]:


import math 
import matplotlib.pyplot as plt
import scipy.constants as const
from datetime import datetime, date
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ###### Declarando as constantes 

# In[ ]:


# Constante de Planck (h)
h = const.h  # em Joules por segundo
# Constante de Boltzmann (k)
k = const.k  # em Joules por Kelvin
# Velocidade da luz (c)
c = const.c  # em metros por segundo
# Pi
pi = const.pi
# Constante de Boltzman
sigma = 5.667*(10**-8)
#Constante de Wien
wien = 2897.756


# # Atividade Pratica 1 - Radiação 

# ### MODELO 1: Com base na Lei de Planck, desenvolva um modelo computacional para calcular a temperatura de brilho de um corpo negro, a partir de radiâncias monocromáticas observadas em diferentes bandas do espectro.  
# 

# Exemplo 1:
# 
#     Valor da radiância espectral (B): 1e-18 W/m^2/sr/µm
#     Escolha da variável independente: 1 (comprimento de onda)
#     Valor do comprimento de onda mínimo: 400 nm (4e-7 m)
#     Valor do comprimento de onda máximo: 700 nm (7e-7 m)
# 
# Exemplo 2:
# 
#     Valor da radiância espectral (B): 1e-16 W/m^2/sr/µm
#     Escolha da variável independente: 2 (número de onda)
#     Valor do número de onda mínimo: 1e3 1/m
#     Valor do número de onda máximo: 1e5 1/m
# 
# Exemplo 3:
# 
#     Valor da radiância espectral (B): 1e-15 W/m^2/sr/µm
#     Escolha da variável independente: 3 (frequência)
#     Valor da frequência mínima: 1e12 Hz
#     Valor da frequência máxima: 1e14 Hz
# 
# Você pode usar esses exemplos como entrada para testar o código e verificar os gráficos resultantes.

# In[ ]:


# Função para calcular a radiância espectral em função do comprimento de onda e temperatura
def radiance_espectral(lambda_, T):
    termo_exponencial = np.exp((h * c) / (lambda_ * k * T))
    return (2 * h * c**2) / (lambda_**5 * (termo_exponencial - 1))

# Função para calcular a temperatura de brilho em função do comprimento de onda e radiância
def temperatura_de_brilho_lambda(lambda_, radiance):
    T = (h * c) / (lambda_ * k * np.log((2 * h * c**2) / (lambda_**5 * radiance + 1)))
    return T

# Função para calcular a temperatura de brilho em função do número de onda e radiância
def temperatura_de_brilho_nu(nu, radiance):
    lambda_ = c / nu
    T = (h * nu) / (k * np.log((2 * h * nu**3) / (c**2 * radiance + 1)))
    return T

# Função para calcular a temperatura de brilho em função da frequência e radiância
def temperatura_de_brilho_f(f, radiance):
    lambda_ = c / f
    T = (h * f) / (k * lambda_ * np.log((2 * h * f**3) / (c**2 * radiance + 1)))
    return T

# Definindo valor mínimo de radiância espectral (B) (evitar divisão por zero):
B_min = 1e-20

# Entrada do usuário para a radiância espectral (B)
B = float(input("Entre com o valor da radiância B(λ,T) em W/m^2/sr/µm: "))

# Verifica se o valor de B é menor do que o valor mínimo
if B < B_min:
    print("Valor de radiância é menor do que o valor mínimo permitido. Encerrando o programa.")
else:
    valor_B = "Usuário = {}".format(B)

    # Entrada do usuário para a escolha da variável independente
    var_independente = int(input("Escolha a entrada (1 - Comprimento de onda, 2 - Número de onda, 3 - Frequência): "))

    if var_independente == 1:
        # Entrada do usuário para o intervalo de comprimento de onda
        lambda_min_option = input("Definir valor mínimo do comprimento de onda manualmente? (s/n): ")
        if lambda_min_option.lower() == 's':
            lambda_min = float(input("Entre com o valor do comprimento de onda mínimo em metros: "))
        else:
            lambda_min = 1e-9

        lambda_max_option = input("Definir valor máximo do comprimento de onda manualmente? (s/n): ")
        if lambda_max_option.lower() == 's':
            lambda_max = float(input("Entre com o valor do comprimento de onda máximo em metros: "))
        else:
            lambda_max = 2.5e-6

        # Vetor de comprimentos de onda em micrômetros
        lambdas = np.linspace(lambda_min, lambda_max, 10000)

    elif var_independente == 2:
        # Entrada do usuário para os valores mínimos e máximos do número de onda
        nu_min_option = input("Definir valor mínimo do número de onda manualmente? (s/n): ")
        if nu_min_option.lower() == 's':
            nu_min = float(input("Entre com o valor do número de onda mínimo em 1/metro: "))
        else:
            nu_min = 1e12

        nu_max_option = input("Definir valor máximo do número de onda manualmente? (s/n): ")
        if nu_max_option.lower() == 's':
            nu_max = float(input("Entre com o valor do número de onda máximo em 1/metro: "))
        else:
            nu_max = 1e15

        # Verifica se o valor máximo é maior do que o valor mínimo
        if nu_max <= nu_min:
            print("O valor máximo do número de onda deve ser maior do que o valor mínimo. Encerrando o programa.")
            exit()

        # Vetor de números de onda em 1/micrômetro
        nus = np.linspace(nu_min, nu_max, 10000)

        # Cálculo do comprimento de onda a partir do número de onda
        lambdas = 1 / nus

    elif var_independente == 3:
        # Entrada do usuário para os valores mínimos e máximos da frequência
        f_min_option = input("Definir valor mínimo da frequência manualmente? (s/n): ")
        if f_min_option.lower() == 's':
            f_min = float(input("Entre com o valor da frequência mínima em Hz: "))
        else:
            f_min = c / (2.5e-6)

        f_max_option = input("Definir valor máximo da frequência manualmente? (s/n): ")
        if f_max_option.lower() == 's':
            f_max = float(input("Entre com o valor da frequência máxima em Hz: "))
        else:
            f_max = c / (1e-9)

        # Vetor de frequências
        fs = np.linspace(f_min, f_max, 10000)

        # Cálculo do comprimento de onda a partir da frequência
        lambdas = c / fs

    else:
        print("Opção inválida. Encerrando o programa.")
        exit()

    # Cálculo da temperatura em função do comprimento de onda
    T = h * c / (k * lambdas * np.log(2 * h * c ** 2 / (B * 1e6 * lambdas ** 5) + 1))

    # Curvas adicionais
    T_sol = temperatura_de_brilho_lambda(lambdas, radiance_espectral(lambdas, 5778))
    T_ferro = temperatura_de_brilho_lambda(lambdas, radiance_espectral(lambdas, 800))
    T_lampada = temperatura_de_brilho_lambda(lambdas, radiance_espectral(lambdas, 3000))
    T_corpo_humano = temperatura_de_brilho_lambda(lambdas, radiance_espectral(lambdas, 310))

    # Plot da temperatura em função do comprimento de onda
    plt.figure(figsize=(12, 6))
    plt.plot(lambdas * 1e6, T, label=valor_B)
    #plt.plot(lambdas * 1e6, T_sol, label="Sol")
    #plt.plot(lambdas * 1e6, T_ferro, label="Ferro Incandescente")
    #plt.plot(lambdas * 1e6, T_lampada, label="Lâmpada")
    #plt.plot(lambdas * 1e6, T_corpo_humano, label="Corpo Humano")
    plt.xlabel('Comprimento de Onda (µm)')
    plt.ylabel('Temperatura de Brilho (K)')
    plt.title('Temperatura de Brilho em Função do Comprimento de Onda')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Imprimir temperatura de brilho do usuário
    print("Temperatura de brilho do usuário (B={} W/m^2/sr/µm):".format(B))

    # Encontrar a temperatura máxima
    temperatura_maxima = np.max(T)

    # Imprimir apenas a temperatura máxima
    print("Temperatura máxima de brilho: {:.2f} K".format(temperatura_maxima))


# ### MODELO 2: Com base na Lei de Stefan-Boltzmann, desenvolva um modelo computacional para calcular a emitância radiante de um corpo negro total ou para diferentes bandas do espectro (isto é, entre dois comprimentos de onda definidos).
# 
# Exemplo 1:
# 
#     Emissividade: 1 (corpo negro)
#     Temperatura: 1000
#     Unidade de temperatura: 3 (Kelvin)
#     Comprimento de onda mínimo: 1
#     Comprimento de onda máximo: 5
# 
# Exemplo 2:
# 
#     Emissividade: 0.8
#     Temperatura: 200
#     Unidade de temperatura: 1 (graus Celsius)
#     Comprimento de onda mínimo: 3
#     Comprimento de onda máximo: 8
# 
# Exemplo 3:
# 
#     Emissividade: 0.9
#     Temperatura: 68
#     Unidade de temperatura: 2 (graus Fahrenheit)
#     Comprimento de onda mínimo: 2
#     Comprimento de onda máximo: 4
# 
# Exemplo 4:
# 
#     Emissividade: 0.5
#     Temperatura: 300
#     Unidade de temperatura: 3 (Kelvin)
#     Comprimento de onda mínimo: 1
#     Comprimento de onda máximo: 3
#     
# Você pode usar esses exemplos como entrada para testar o código.

# In[ ]:


def print_line():
    print("=" * 50)
print_line()
print("Para o cálculo da emitância de um corpo é necessário saber primeiro sua emissividade.")
print("Qual é a emissividade do corpo estudado?")
print("Lembrando que, caso o estudo seja para um corpo negro, a emissividade será igual a 1.")
emi = float(input("Emissividade: "))
print_line()

print("\nDigite o valor da temperatura da superfície do corpo estudado:")
temp = float(input("Temperatura: "))
print_line()

print("\nQual é a unidade da sua temperatura?")
print("Digite:")
print("1 - Temperatura em graus Celsius (°C)")
print("2 - Temperatura em graus Fahrenheit (°F)")
print("3 - Temperatura em graus Kelvin (K)")
temp1 = int(input("Unidade: "))
print_line()

if temp1 == 1:
    temp0 = temp + 273.15

elif temp1 == 2:
    temp2 = (temp - 32) / 1.8
    temp0 = temp2 + 273.15

elif temp1 == 3:
    temp0 = temp

else:
    print("ERRO: seu valor não corresponde a nenhuma das opções recomendadas.")
    

lambda_min = float(input("Digite o comprimento de onda mínimo em µm: "))
lambda_max = float(input("Digite o comprimento de onda máximo em µm: "))

# Calcular a emitância radiante na banda do espectro especificada pelo usuário
emitancia = 0.0
for wavelength in range(int(lambda_min), int(lambda_max) + 1):
    lambda_meters = wavelength * 1e-6  # Converter comprimento de onda para metros
    if lambda_meters == 0:
        continue
    termo_exponencial = math.exp(sigma / (lambda_meters * temp0))
    emitancia += sigma / (lambda_meters ** 5 * (termo_exponencial - 1))

emitancia *= emi

print_line()
print("RESULTADO:")
print_line()
print(f"\nEmitância radiante no espectro entre {lambda_min} µm e {lambda_max} µm: {emitancia:.3e} W/m²")
print_line()


# ### MODELO 3: Desenvolva um modelo computacional para calcular, dada a temperatura de um corpo negro, o valor do comprimento de onda, para o qual a radiância espectral emitida é máxima.
# Exemplo 1:
# 
#     Temperatura: 500
#     Unidade de temperatura: 1 (graus Celsius)
# 
# Exemplo 2:
# 
#     Temperatura: 1000
#     Unidade de temperatura: 2 (graus Fahrenheit)
# 
# Exemplo 3:
# 
#     Temperatura: 300
#     Unidade de temperatura: 3 (Kelvin)
#     
# Você pode usar esses exemplos como entrada para testar o código.

# In[ ]:


print("\n")
print("Digite o valor da temperatura da superfície do corpo negro estudado:")
temp = float(input())
print("\n")
print("Qual é a unidade da sua temperatura?\nDigite:\n1 - Temperatura em graus Celsius (°C)\n2 - Temperatura em graus Fahrenheit (°F))\n3 - Temperatura em graus Kelvin (K) ")
temp1 = int(input())
print("\n")

if temp1 == 1:
    temp0 = temp + 273.15
    unidade_temp = "°C"
elif temp1 == 2:
    temp2 = (temp - 32) / 1.8
    temp0 = temp2 + 273.15
    unidade_temp = "°F"
elif temp1 == 3:
    temp0 = temp
    unidade_temp = "K"
else:
    print('ERROR: Seu valor não corresponde a nenhum dos recomendados.')

# Lei de Wien
maxi = wien / temp0
print("====================================================")
print("RESULTADO:")
print("====================================================")
print("Temperatura da superfície do corpo negro: {:.2f} {}".format(temp, unidade_temp))
print("Comprimento de onda para o qual a radiância espectral é máxima: {:.2f} micrômetros".format(maxi))
print("====================================================")


# ### MODELO 4: Desenvolva um modelo computacional para calcular a energia total emitida por um emissor circular, infinitamente fino e que age como um corpo negro, a diferentes temperaturas. Esse modelo também deverá calcular a irradiância detectada sobre um alvo que pode estar a diferentes distâncias do centro desse emissor.
# Exemplo 1:
# 
#     Unidade de medida para o diâmetro do emissor circular: 1 (metros)
#     Valor do diâmetro do emissor: 0.5
#     Unidade de medida para a temperatura do emissor: 1 (Kelvin)
#     Valor da temperatura do emissor: 1000
#     Unidade de medida para a distância até o alvo puntual: 1 (metros)
#     Valor da distância até o alvo puntual: 10
# 
# Exemplo 2:
# 
#     Unidade de medida para o diâmetro do emissor circular: 2 (centímetros)
#     Valor do diâmetro do emissor: 10
#     Unidade de medida para a temperatura do emissor: 2 (Celsius)
#     Valor da temperatura do emissor: 25
#     Unidade de medida para a distância até o alvo puntual: 2 (centímetros)
#     Valor da distância até o alvo puntual: 50
# 
# Exemplo 3:
# 
#     Unidade de medida para o diâmetro do emissor circular: 3 (quilômetros)
#     Valor do diâmetro do emissor: 1
#     Unidade de medida para a temperatura do emissor: 1 (Kelvin)
#     Valor da temperatura do emissor: 5000
#     Unidade de medida para a distância até o alvo puntual: 3 (quilômetros)
#     Valor da distância até o alvo puntual: 5
#     
# Você pode usar esses exemplos como entrada para testar o código.

# In[ ]:


# Função para imprimir uma linha visual
def print_line():
    print("=" * 50)

# Exibe as opções de unidade de medida para o diâmetro do emissor circular
print_line()
print("Escolha a unidade de medida para o diâmetro do emissor circular:")
print("1 - Metros")
print("2 - Centímetros")
print("3 - Quilômetros")
unidade_d_emissor = int(input("Digite o número correspondente à opção escolhida: "))

# Solicita ao usuário o valor do diâmetro do emissor circular
d_emissor = float(input("Digite o valor do diâmetro do emissor: "))

# Converte o diâmetro para metros, se necessário
if unidade_d_emissor == 2:
    d_emissor /= 100
elif unidade_d_emissor == 3:
    d_emissor *= 1000

# Exibe as opções de unidade de medida para a temperatura do emissor
print_line()
print("Escolha a unidade de medida para a temperatura do emissor:")
print("1 - Kelvin")
print("2 - Celsius")
unidade_T_emissor = int(input("Digite o número correspondente à opção escolhida: "))

# Solicita ao usuário o valor da temperatura do emissor
T_emissor = float(input("Digite o valor da temperatura do emissor: "))

# Converte a temperatura para Kelvin, se necessário
if unidade_T_emissor == 2:
    T_emissor += 273.15

# Exibe as opções de unidade de medida para a distância até o alvo puntual
print_line()
print("Escolha a unidade de medida para a distância até o alvo puntual:")
print("1 - Metros")
print("2 - Centímetros")
print("3 - Quilômetros")
unidade_dist_alvo = int(input("Digite o número correspondente à opção escolhida: "))

# Solicita ao usuário o valor da distância até o alvo puntual
dist_alvo = float(input("Digite o valor da distância até o alvo puntual: "))

# Converte a distância para metros, se necessário
if unidade_dist_alvo == 2:
    dist_alvo /= 100
elif unidade_dist_alvo == 3:
    dist_alvo *= 1000

# Calcula a área do emissor circular
area_emissor = math.pi * pow((d_emissor / 2), 2)

# Calcula a energia total emitida pelo emissor
Stefan_Boltzmann = 5.670374419e-8
energia_total = Stefan_Boltzmann * pow(T_emissor, 4)

# Calcula a área lateral do cone
g = math.sqrt(pow((d_emissor / 2), 2) + pow(dist_alvo, 2))
area_lateral = math.pi * g * (d_emissor / 2)

# Calcula a irradiância detectada pelo alvo puntual
irradiancia = energia_total / area_lateral

# Exibe os resultados ao usuário
print_line()
print("RESULTADO:")
print_line()
print(f"Área do emissor circular: {area_emissor:.2f} m²")
print(f"Energia total emitida pelo emissor: {energia_total:.2e} W")
print(f"Irradiância detectada pelo alvo puntual: {irradiancia:.2e} W/m²")
print_line()


# ### MODELO 5: Desenvolva um modelo computacional para converter densidades espectrais, isto é, radiâncias ponderadas por diferentes feixes monocromáticas nas unidades de comprimento de onda, número de onda e frequência.
# Exemplo 1:
# 
#     A radiância espectral está em: 1 (Comprimento de onda)
#     Deseja converter para: 1 (Número de onda)
#     Valor da radiância espectral: 5e-6
#     Valor do comprimento de onda: 500e-9
# 
# Exemplo 2:
# 
#     A radiância espectral está em: 2 (Número de onda)
#     Deseja converter para: 2 (Frequência)
#     Valor da radiância espectral: 1e-3
#     Valor do comprimento de onda: 600e-9
# 
# Exemplo 3:
# 
#     A radiância espectral está em: 3 (Frequência)
#     Deseja converter para: 1 (Comprimento de onda)
#     Valor da radiância espectral: 2e13
#     Valor do comprimento de onda: 700e-9
# 
# Você pode usar esses exemplos como entrada para testar o código.

# In[ ]:


# Função para imprimir uma linha visual
def print_line():
    print("=" * 50)

# Pede ao usuário para escolher qual transformação deseja realizar
print_line()
print("A radiância espectral está em?")
print("1 - Comprimento de onda?")
print("2 - Número de onda?")
print("3 - Frequência?")
opcao = int(input("Selecione a radiância (1 a 3): \n"))

#Comprimento de onda
if opcao == 1:
    print_line()
    print("Deseja converter para:")
    print("1 - Número de onda")
    print("2 - Frequência")
    opcao_2 = int(input("Digite a opção desejada: "))
    print_line()
    # Realiza a transformação selecionada pelo usuário
    if opcao_2 == 1:
        # transformação Comprimento de onda para Numero de onda
        l = float(input("Informe o valor da radiância espectral (em W m^-2 sr^-1 µm^-1): ")) * 1e6
        comp = float(input("Digite o valor do comprimento de onda (em m): "))
        l_nu = l * (comp ** 2)
        print(f"A radiância espectral em número de onda é de {l_nu:.4e} W m^-2 sr^-1 m^-1")
    elif opcao_2 == 2:
        # transformação comprimento de onda para frequência
        l = float(input("Digite o valor da radiância espectral em comprimento de onda (em W m^-2 sr^-1 µm^-1): ")) * 1e6
        comp = float(input("Digite o valor do comprimento de onda (em m): "))
        l_f = l * (comp ** 2) / c
        print(f"A radiância espectral em frequência é de {l_f:.4e} W m^-2 sr^-1 Hz^-1")

#Número de onda
elif opcao == 2:
    print_line()
    print("Deseja converter para:")
    print("1 - Comprimento de onda")
    print("2 - Frequência")
    opcao_2 = int(input("Digite a opção desejada: "))
    print_line()
    # Realiza a transformação selecionada pelo usuário
    if opcao_2 == 1:
        #Transformação Número de onda para Comprimento de onda
        l = float(input("Digite o valor da radiância espectral em número de onda (em W m^-2 sr^-1 cm^-1): ")) * 100
        comp = float(input("Digite o valor do comprimento de onda (em m): "))
        l_comp = l * c / (comp ** 2)
        print(f"A radiância espectral em comprimento de onda é de {l_comp:.4e} W m^-2 sr^-1 µm^-1")
    elif opcao_2 == 2:
        # transformação número de onda para frequência
        l = float(input("Digite o valor da radiância espectral em número de onda (em W m^-2 sr^-1 cm^-1): ")) * 100
        comp = float(input("Digite o valor do comprimento de onda (em m): "))
        l_f = l * c / comp
        print(f"A radiância espectral em frequência é de {l_f:.4e} W m^-2 sr^-1 Hz^-1")

#Frequência
elif opcao == 3:
    print_line()
    print("Deseja converter para:")
    print("1 - Comprimento de Onda")
    print("2 - Número de onda")
    opcao_2 = int(input("Digite a opção desejada: "))
    print_line()
    if opcao_2 == 1:
        # transformação frequência para comprimento de onda
        l = float(input("Digite o valor da radiância espectral em frequência (em Hz): "))
        comp = float(input("Digite o valor do comprimento de onda (em m): "))
        l_f = l * c / (comp ** 2)
        print(f"A radiância espectral em comprimento de onda é de {l_f:.4e} W m^-2 sr^-1 nm^-1")
    elif opcao_2 == 2:
        # transformação frequência para número de onda
        l = float(input("Digite o valor da radiância espectral em frequência (em Hz): "))
        comp = float(input("Digite o valor do comprimento de onda (em m): "))
        l_n = l * comp / c
        print(f"A radiância espectral em número de onda é de {l_n:.4e} W m^-2 sr^-1 m^-1")

print_line()


# ### MODELO 6: Desenvolva um modelo computacional para utilização por um usuário leigo. O aplicativo deve ter as seguintes características:
# Exemplo 1:
# 
#     Data: 25/04/2023
#     Latitude: -22.90
#     Longitude: -43.1729
#     Hora: 12:30:00
# 
# Exemplo 2:
# 
#     Data: 08/09/2023
#     Latitude: 40.71
#     Longitude: -74.0060
#     Hora: 18:45:00
# 
# Exemplo 3:
# 
#     Data: 02/11/2023
#     Latitude: -33.86
#     Longitude: 151.20
#     Hora: 09:15:00
#     
# Você pode usar esses exemplos como entrada para testar o código.

# In[ ]:


# Função para imprimir uma linha visual
def print_line():
    print("=" * 50)
    
#Entradas do Usuário----------------------------------------------------------
print_line()
# Solicita ao usuário que insira a data
data_str = input("Digite a data no formato DD/MM/AAAA: ")

# Converte a string de data para um objeto datetime
data = datetime.strptime(data_str, "%d/%m/%Y")

# Extrai o dia e o mês da data
dia = data.day
mes = data.month
ano = data.year

dn = date(ano,mes,dia).timetuple().tm_yday

#Solicita ao usuário que insira as lat e lon
print_line()
print("Digite a latitude em que você se encontra:")
lat = float(input())
print_line()
print("Digite a Longitude em que você se encontra:")
lon = float(input())
print_line()

# Solicita ao usuário que insira a hora
hora_str = input("Digite a hora no formato HH:MM:SS: ")
# Converte a string de hora para um objeto datetime
hora = datetime.strptime(hora_str, "%H:%M:%S")

# Extrai a hora e os minutos da hora
hor = hora.hour
mi = hora.minute
secs = hora.second

print_line()

#Calculos--------------------------------------------------------------------

#Hora de Greenwich--------------------------------------------------
if lon>0:
    var = int(lon)/15
    
    aux = mi/60
    UT = (hor - var) + aux

elif lon<0:
    var = int(lon)/15
    
    var = var*-1
    
    aux = mi/60
    UT = hor + var + aux

#Eq. do tempo----------------------------------------------------------------
# Cálculo do Gama (L):
L0 = 2*pi*(dn-1)
L = L0/365

# Equação do tempo (et):
et1 = 0.000075 + (0.001868*(math.cos(L)))
et2 = -(0.0320077*(math.sin(L)))
et3 = -(0.014615*(math.cos((2*L))))
et4 = -(0.040849*(math.sin((2*L))))
et5 = et1+et2+et3+et4
et = et5*(1440/(2*pi))
et = et/60

#Declinação Solar------------------------------------------------------------

deta1 = 0.006918 - (0.399912*(math.cos(L)))
deta2 = 0.070257*(math.sin(L))
deta3 = -0.006758*(math.cos((2*L)))
deta4 = 0.000907*(math.sin((2*L)))
deta5 = -(0.002697*(math.cos((3*L))))
deta6 = 0.00148*(math.sin((3*L)))
deta = deta1+deta2+deta3+deta4+deta5+deta6
deta = deta*(180/pi)


#Instante da passagem meridiana do Sol----------------------------------------------
H = - (lon/15) + (12-et)
H = (H * (360/24))/15

#Hora do nascer e do por do sol------------------------------------------------------------
Ho1 = -((math.tan(lat*(pi/180))))
Ho2 = ((math.tan(deta*(pi/180))))
Ho3 = Ho1*Ho2
Ho0 = (math.acos(Ho3))#Ho em rad
Ho = (math.acos(Ho3))*(180/pi) #Ho em graus
Hoh = Ho/15 #Ho em horas
if Hoh<0:
    HOH = 12-Hoh
    u = 12+Hoh
elif Hoh>0:
    HOH = 12+Hoh
    u = 12-Hoh
    
#Elevação do Sol na hora local inserida pelo usuário-----------------------------------------------

Z1 = math.sin(deta*(pi/180))
Z2 = math.sin(lat*(pi/180))
Z3 = math.cos(lat*(pi/180))
Z4 = math.cos(deta*(pi/180))
Z5 = math.cos(Ho*(pi/180))
Z6 = Z1*Z2
Z7 = Z3*Z4*Z5
Z8 = Z6 + Z7
Z = math.acos(Z8)*(180/pi)
#Azimute do Sol na hora local inserida pelo usuário----------------------------------------------------


#Azimute para seno 
fi1 = math.cos(deta*(pi/180)) 
fi2 = (math.sin(Ho*(pi/180)))
fi3 = math.sin(Z*(pi/180))

fi4 = -(fi1*fi2)
fi5 = fi4/fi3
fi = math.asin(fi5)*(180/pi)
#print(fi)

#Azimute para cosseno
fl1 = math.sin(deta*(pi/180))
fl2 = math.cos(lat*(pi/180))
fl3 = math.cos(deta*(pi/180))
fl4 = math.sin(lat*(pi/180))
fl5 = math.cos(Ho*(pi/180))
fl6 = math.sin(Z*(pi/180))

fl7 = fl1*fl2
fl8 = - (fl3*fl4*fl5)
fl9 = (fl7+fl8)/fl6
fl = math.acos(fl9)*(180/pi)


#Saídas----------------------------------------------------------------------------------

#Mostra os quadrantes ------------------------------------------------------------------

fol = math.acos(fl9)*(180/pi)
if fi>0:
    if fol>0:
        fl = 1
    elif fol<0:
        fol = 2
elif fi<0:
    if fol<0:
        fol = 3
    elif fol>0:
        fol = 4

#Duração do dia solar -------------------------------------------------------------------

DS = abs(Hoh)
DS = 2*DS

#TOP-----------------------------------------------------------------------------------------

#Distância terra sol
dis1 = 1.000110
dis2 = 0.034221*(math.cos(L))
dis3 = 0.001280*(math.sin(L))
dis4 = 0.000719*(math.cos((2*L)))
dis5 = 0.000077*(math.sin((2*L)))

dis = dis1 + dis2 + dis3 + dis4 + dis5
Eo = 1360

DO1 = dis*(Eo*3600)*2
DO2 = math.sin(lat*(pi/180))
DO3 = math.sin(deta*(pi/180))
DO4 = math.cos(lat*(pi/180))
DO5 = math.cos(deta*(pi/180))
DO6 = math.sin(Ho*(pi/180))
DO7 = (Ho/15)*(DO2*DO3)
DO8 = DO4*DO5*DO6
DO9 = (DO8*(180/pi)*(1/15))

DO10 = DO7+DO9
DOSE = DO1*DO10


# Armazenar as saídas em um dicionário
output = {
    'Dia no ano': [dn],
    'Hora de Greenwich': [f'{UT:.1f}'],
    'Eq. do Tempo': [f'{et:.3f}'],
    'Declinação Solar': [f'{deta:.2f}'],
    'Passagem Meridiana': [f'{H:.1f}'],
    'Hora do Nascer do Sol': [f'{u:.1f}'],
    'Hora do Pôr do Sol': [f'{HOH:.1f}'],
    'Elevação do Sol': [f'{Z:.2f}'],
    'Azimute do Sol (Seno)': [f'{fi:.2f}'],
    'Azimute do Sol (Cosseno)': [f'{fl:.2f}'],
    'Quadrante': [fol],
    'Duração do Dia Solar': [f'{DS:.1f}'],
    'Dose de Radiação Solar': [f'{DOSE:.2e}']
}

# Criar um DataFrame a partir do dicionário
df = pd.DataFrame(output)
def align_text(text):
    return 'text-align: center'
styled_df = df.style.applymap(align_text)

styled_df


# In[ ]:





# In[ ]:




