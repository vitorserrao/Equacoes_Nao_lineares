import numpy as np
import sympy as sp


#============= configuração do sistema ================

tolerancia = 1e-6                                                              # tolerancia do erro
k = 1                                                                         # numero de iterações
kmax = 20                                                                      # numero de iterações maximo
s = list()                                                                     # lista de soluções
ci = sp.Matrix([[0.2, 0.5, 0.85, 0.93, 0.22]])                                 # condição inicial


A = sp.Matrix([[-3.933, 0.107, 0.126, 0, -9.99, 0, -45.83, -7.64],             # Matriz A
              [0, -0.987, 0, -22.95, 0, -28.37, 0, 0],
              [0.002, 0, -0.235, 0, 5.67, 0, -0.921, -6.51],
              [ 0, 1.0, 0, -1.0, 0, -0.168, 0, 0],
              [0, 0, -1.0, 0, -0.196, 0, -0.0071, 0]])

#============= Definição das funções ================


x1 = sp.Symbol("x1")                                            # devinindo x1 como simbolo
x2 = sp.Symbol("x2")
x3 = sp.Symbol("x3")
x4 = sp.Symbol("x4")
x5 = sp.Symbol("x5")
var_x = sp.Matrix([[x1],[x2],[x3],[x4],[x5]])                   # matriz com as variáveis

f1 = -0.727*x2*x3 + 8.39*x3*x4 - 684.4*x4*x5 + 63.5*x4*x2       #devinindo f1 como função
f2 = 0.949*x1*x3 + 0.173*x1*x5
f3 = -0.716*x1*x2 - 1.578*x1*x4 + 1.132*x4*x2
f4 = -x1*x5
f5 = x1*x4


F = sp.Matrix([[f1],[f2],[f3],[f4],[f5]])                       # matriz com as funções
Fx = []
#================== Operações =====================

x = sp.Matrix([x1, x2, x3, x4, x5, 1, 1, 1])                    # matriz x
sistema = (A*x) + F                                             # matiz do sistema
s.append(ci)                                                    # add condição inicial na lista de soluções

# subistituindo x1,x2,x3,x4,x5 nas funções ou seja achando F(x0):
fxo = sistema.subs({x1: s[-1][0], x2: s[-1][1], x3: s[-1][2], x4: s[-1][3], x5: s[-1][4]})
Fx.append(fxo)
#  condição valor max do módulo de fxo



while np.max(np.abs(fxo)) > tolerancia and k < kmax:

    jacobiano = sistema.jacobian(var_x)        # Calculando Jacobiano

    # Substituindo  os valores de  x1,x2,x3,x4,x5 no  Jacobiano:
    jfxo = jacobiano.subs({x1:s[-1][0], x2:s[-1][1], x3:s[-1][2], x4:s[-1][3], x5:s[-1][4]})

    #Regra de atualização do x definido por x = xo + ((jfxo^(-1) * (-fxo)):
    novo_x = np.array(s[-1]) - np.array((jfxo.inv()*(fxo))).transpose()

    # add o novo valor de x na lista de soluções
    s.append([novo_x[0][0], novo_x[0][1], novo_x[0][2], novo_x[0][3],novo_x[0][4]])

    fxo = sistema.subs({x1: s[-1][0], x2: s[-1][1], x3: s[-1][2], x4: s[-1][3], x5: s[-1][4]})
    k += 1
    Fx.append(fxo)

    Fxpipe = np.array(Fx).astype(np.float64)
    #plt.plot(k, Fx, color='blue', marker='o', markersize=4)
    #plt.text(k + 0.05, Fx + 0.005, round(Fx, 6), ha='left')


rx = [[],[],[],[],[]]

print(s[0][0])
for i in range(10):
    rx[0].append(np.abs(s[i][0]) - s[-1][0])
    rx[1].append(np.abs(s[i][1]) - s[-1][1])
    rx[2].append(np.abs(s[i][2]) - s[-1][2])
    rx[3].append(np.abs(s[i][3]) - s[-1][3])
    rx[4].append(np.abs(s[i][4]) - s[-1][4])
print(f'test: {rx[0]}')
    #print(f'o Valor de x1 : {s[-1][0]}, valor de x na iteração: {s[i][0]} e residuo:{r[i]}')
print(f'Número de iterações: {k}')
print(f'o Valor de x1: {s[-1][0]}')
print(f'o Valor de x2: {s[-1][1]}')
print(f'o Valor de x3: {s[-1][2]}')
print(f'o Valor de x4: {s[-1][3]}')
print(f'o Valor de x5: {s[-1][4]}')
