import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from scipy.optimize import minimize, rosen, rosen_der
from re import M

def calcola_p(t1, t2, t3):
    # Calcolare p00, p10, p01, p11
    p00 = (1 + t1 - t2 + t3) / 4
    p10 = (1 - t1 + t2 + t3) / 4
    p01 = (1 + t1 + t2 - t3) / 4
    p11 = (1 - t1 - t2 - t3) / 4

    if p00<0 or p10<0 or p01<0 or p11<0:
      print("Stato non fisico")
      return False
    else:
      # Restituire i valori calcolati come una lista o una tupla
      return p00, p10, p01, p11

def calcola_t(p00, p10, p01, p11):
    # Verifica che le probabilità siano valide
    if any(p < 0 for p in [p00, p10, p01, p11]):
        print("Probabilità non fisiche (valore negativo).")
        return False
    if abs(p00 + p10 + p01 + p11 - 1) > 1e-9:
        print("Le probabilità non sommano a 1.")
        return False

    t1 = 2 * (p00 + p01) - 1
    t2 = 2 * (p10 + p01) - 1
    t3 = 2 * (p00 + p10) - 1

    return t1, t2, t3

def calcola_t_vec(p00, p10, p01):
    t1 = 2 * (p00 + p01) - 1
    t2 = 2 * (p10 + p01) - 1
    t3 = 2 * (p00 + p10) - 1
    return t1, t2, t3



##### Hilbert-Schmidt ######

def M_hs_nonloc(t_1, t_2, t_3): #Definizione della funzione
  a = 1/(2)**(1/2); #Range di variazoine dei parametri
  M_Bell_hs = (9-6*(2)**(1/2))/2; #Normalizzazione (calcolata in modo tale che gli stati di Bell abbiano misura di nonlocality pari a 1)
  fun = lambda x: (t_1-x[0])**2+(t_2-x[1])**2+(t_3-x[2])**2; #Funzione da minimizzare rispetto al vettore x = (x[0], x[1], x[2]) che mettendo i constraints si troverà sulla superficie di separazione.

  #Abbiamo diverse superfici di separazione disgiunte (tra local e nonlocal), dato uno stato input,
  #il minimo sarà calcolato da ogni superficie di separazione e poi verrà preso il valore più piccolo (vedi sotto).
  #Il tipo di constraint (type) è una disuguaglianza (fun>0) se 'ineq' ed è un'uguaglianza (fun=0) se 'eq'.

  #Superface 1
  constr1 =  ({'type': 'ineq', 'fun': lambda x: (-x[0]**2-x[1]**2+1).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (3-(x[0]**2+x[1]**2+x[2]**2)).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (abs(x[0])-abs(x[2])).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (abs(x[1])-abs(x[2])).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (1+x[0]-x[1]+x[2]).astype(float)}, #Constraint sulle probabilità <1
              {'type': 'ineq', 'fun': lambda x: (1+x[0]+x[1]-x[2]).astype(float)}, #Constraint sulle probabilità <1
              {'type': 'ineq', 'fun': lambda x: (1-x[0]+x[1]+x[2]).astype(float)}, #Constraint sulle probabilità <1
              {'type': 'ineq', 'fun': lambda x: (1-x[0]-x[1]-x[2]).astype(float)}); #Constraint sulle probabilità <1
  bnds = ((-1.,1.),(-1.,1.),(-1.,1.)); #Intervalli in cui giacciono le x[i]
  res1 = minimize(fun, (a, a, -a), method = None, bounds = bnds, constraints = constr1); #Funzione di shipy che fa il minimo
  M_hs1 = res1.fun/M_Bell_hs; #Carico quì il valore della misura (normalizzato) ottenuto dal minimo rispetto alla prima superficie

  #Superface 2
  constr2 = ({'type': 'ineq', 'fun': lambda x: (-x[0]**2-x[2]**2+1).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (3-(x[0]**2+x[1]**2+x[2]**2)).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (abs(x[0])-abs(x[1])).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (abs(x[2])-abs(x[1])).astype(float)},
              {'type': 'ineq', 'fun': lambda x: (1+x[0]-x[1]+x[2]).astype(float)}, #Constraint sulle probabilità <1
              {'type': 'ineq', 'fun': lambda x: (1+x[0]+x[1]-x[2]).astype(float)}, #Constraint sulle probabilità <1
              {'type': 'ineq', 'fun': lambda x: (1-x[0]+x[1]+x[2]).astype(float)}, #Constraint sulle probabilità <1
              {'type': 'ineq', 'fun': lambda x: (1-x[0]-x[1]-x[2]).astype(float)}); #Constraint sulle probabilità <1
  bnds = ((-1.,1.),(-1.,1.),(-1.,1.));
  res2 = minimize(fun, (a, -a, a), method = None, bounds = bnds, constraints = constr2);
  M_hs2 = res2.fun/M_Bell_hs;

  #Superface 3
  constr3 = ({'type': 'ineq', 'fun': lambda x: (-x[1]**2-x[2]**2+1).astype(float)},
                {'type': 'ineq', 'fun': lambda x: (3-(x[0]**2+x[1]**2+x[2]**2)).astype(float)},
                {'type': 'ineq', 'fun': lambda x: (abs(x[1])-abs(x[0])).astype(float)},
                {'type': 'ineq', 'fun': lambda x: (abs(x[2])-abs(x[0])).astype(float)},
                {'type': 'ineq', 'fun': lambda x: (1+x[0]-x[1]+x[2]).astype(float)}, #Constraint probabilities <1
                {'type': 'ineq', 'fun': lambda x: (1+x[0]+x[1]-x[2]).astype(float)},
                {'type': 'ineq', 'fun': lambda x: (1-x[0]+x[1]+x[2]).astype(float)},
                {'type': 'ineq', 'fun': lambda x: (1-x[0]-x[1]-x[2]).astype(float)});
  bnds = ((-1.,1.),(-1.,1.),(-1.,1.));
  res3 = minimize(fun, (-a, a, a), method = None, bounds = bnds, constraints = constr3);
  M_hs3 = res3.fun/M_Bell_hs;

  M_hs = M_hs1;
  if M_hs2 < M_hs:
    M_hs = M_hs2;
  if M_hs3 < M_hs:
     M_hs = M_hs3;
  return(math.sqrt(M_hs));

###### Hellinger #######

def M_hellinger_nonloc(p):
    M_Bell_He = 2 - (1 + 3/(2**0.5))**0.5
    constr = (
        {'type': 'eq', 'fun': lambda x: (x[0]-x[1])**2 + (x[2]-x[3])**2 - 0.5},
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1},
        {'type': 'ineq', 'fun': lambda x: abs(x[0]-x[1]+x[2]-x[3]) - abs(x[0]+x[1]-x[2]-x[3])},
        {'type': 'ineq', 'fun': lambda x: abs(-x[0]+x[1]+x[2]-x[3]) - abs(x[0]+x[1]-x[2]-x[3])}
    )
    bnds = ((0,1),(0,1),(0,1),(0,1))
    fun = lambda x: (((1-p)**0.5 - x[0]**0.5)**2
                     + (p**0.5 - x[2]**0.5)**2
                     + x[1] + x[3])
    res = minimize(fun, (0.5, 0., 0.5, 0.), method=None, bounds=bnds, constraints=constr)
    return fun(res.x)/M_Bell_He


###### Trace #######
def M_trace_nonloc(p):
    M_Bell_Tr = 1.5*(1 - 1/(2**0.5))
    constr = (
        {'type': 'eq', 'fun': lambda x: (x[0]-x[1])**2 + (x[2]-x[3])**2 - 0.5},
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1},
        {'type': 'ineq', 'fun': lambda x: abs(x[0]-x[1]+x[2]-x[3]) - abs(x[0]+x[1]-x[2]-x[3])},
        {'type': 'ineq', 'fun': lambda x: abs(-x[0]+x[1]+x[2]-x[3]) - abs(x[0]+x[1]-x[2]-x[3])}
    )
    bnds = ((0,1),(0,1),(0,1),(0,1))
    fun = lambda x: abs((1-p) - x[0]) + abs(p - x[2]) + x[1] + x[3]
    res = minimize(fun, (0.5, 0., 0.5, 0.), method=None, bounds=bnds, constraints=constr)
    return fun(res.x)/M_Bell_Tr

###### Relative entropy #######
def M_relent_nonloc(p):
    M_Bell_Re = 2 - np.log2(1 + 3/(2**0.5))
    constr = (
        {'type': 'eq', 'fun': lambda x: (x[0]-x[1])**2 + (x[2]-x[3])**2 - 0.5},
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1},
        {'type': 'ineq', 'fun': lambda x: abs(x[0]-x[1]+x[2]-x[3]) - abs(x[0]+x[1]-x[2]-x[3])},
        {'type': 'ineq', 'fun': lambda x: abs(-x[0]+x[1]+x[2]-x[3]) - abs(x[0]+x[1]-x[2]-x[3])}
    )
    bnds = ((0,1),(0,1),(0,1),(0,1))
    fun = lambda x: (p*np.log2(p) - p*np.log2(x[0])
                     + (1-p)*np.log2(1-p) - (1-p)*np.log2(x[2]))
    res = minimize(fun, (0.5, 0., 0.5, 0.), method=None, bounds=bnds, constraints=constr)
    val = fun(res.x)/M_Bell_Re
    if val > 1.0:   # stesso check del vecchio codice
         val = 1.0
    return val
