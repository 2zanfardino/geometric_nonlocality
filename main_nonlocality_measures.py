import matplotlib.pyplot as plt
import numpy as np
from measures import calcola_t, calcola_t_vec, M_hs_nonloc, M_hellinger_nonloc, M_trace_nonloc, M_relent_nonloc


###### Werner states (use analyitical formula) #######

epsilon = 0.001
w_min = 1/(2**(1/2))+epsilon;
w_max = 1;
w_x = np.linspace(w_min, w_max, 10000)
Max_HS = (9-6*(2)**(1/2))/2
Max_He = 2-(1+3/(2**(1/2)))**(1/2)
Max_Tr = (3/2)*(1-1/((2)**(1/2)))
Max_Re = 2-np.log2(1+3/(2**(1/2)))


MHS_f = lambda w: np.sqrt(3*((1/(2**(1/2))-w)**2)/Max_HS)
M_He_f = lambda w: (2-(1/2)*(3*((1-w)*(1-1/(2**(1/2))))**(1/2)+((1+3*w)*(1+3/(2**(1/2))))**(1/2)))/(Max_He);
M_Tr_f = lambda w: (3/2)*(w-(1/(2)**(1/2)))/(Max_Tr)
M_Re_f = lambda w: ( (3/4)*(1-w)*np.log2((1/4)*(1-w)) + (1/4)*(1+3*w)*np.log2((1/4)*(1+3*w)) - (3/4)*(1-w)*np.log2((1/4)*(1-1/((2)**(1/2)))) - (1/4)*(1+3*w)*np.log2((1/4)*(1+3/((2)**(1/2)))))/(Max_Re)

M_HS = MHS_f(w_x)
M_He = M_He_f(w_x)
M_Tr = M_Tr_f(w_x)
M_Re = M_Re_f(w_x)


plt.plot(w_x, M_HS, color="k", linestyle="--", label=r'$\widetilde{\mathcal{M}}_{\mathrm{HS}} = \widetilde{\mathcal{M}}_{\mathrm{Tr}}$')
plt.plot(w_x, M_He, color="k", linestyle="-.", label=r'$\widetilde{\mathcal{M}}_{\mathrm{He}} = \widetilde{\mathcal{M}}_{\mathrm{Bu}}$')
plt.plot(w_x, M_Re, color="k", linestyle="-", label=r'$\widetilde{\mathcal{M}}_{\mathrm{Re}}$')

plt.xlabel(r'$\omega$', size=16)
plt.ylabel(r'$\widetilde{\mathcal{M}}$', size=16)

# modifica della dimensione dei numeri (tick labels)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.legend()
plt.show()


p_min = 0.5
p_max = 1.0
n = 50   # abbastanza punti per un plot liscio
p_vec = np.linspace(p_min, p_max, n)

M_hs_vec = np.zeros(n)
M_he_vec = np.zeros(n)
M_tr_vec = np.zeros(n)
M_re_vec = np.zeros(n)

for i in range(n):
    t1, t2, t3 = calcola_t(p_vec[i], 1-p_vec[i], 0, 0)
    M_hs_vec[i] = M_hs_nonloc(t1, t2, t3)

    M_he_vec[i] = M_hellinger_nonloc(p_vec[i])
    M_tr_vec[i] = M_trace_nonloc(p_vec[i])
    M_re_vec[i] = M_relent_nonloc(p_vec[i])

# --- Plot ---
plt.figure(figsize=(7,5))

jitter = 0.002

plt.scatter(p_vec, M_hs_vec, color=str(0.0), s=15, marker="o", label=r'$\widetilde{\mathcal{M}}_{\mathrm{HS}}$')
plt.scatter(p_vec + jitter, M_tr_vec, color=str(0.3), s=15, marker="^", label=r'$\widetilde{\mathcal{M}}_{\mathrm{Tr}}$')
plt.scatter(p_vec - jitter, M_re_vec, color=str(0.6), s=15, marker="x", label=r'$\widetilde{\mathcal{M}}_{\mathrm{RE}}$')
plt.scatter(p_vec, M_he_vec + jitter, color=str(0.8), edgecolor='k', s=15, marker="s", label=r'$\widetilde{\mathcal{M}}_{\mathrm{He}}$')


# plt.grid(alpha=0.4)
plt.xlabel(r"$p$")
plt.ylabel(r"$\widetilde{\mathcal{M}}$")
plt.legend()
plt.tight_layout()

plt.xlabel(r'$p$', size=16)
plt.ylabel(r'$\widetilde{\mathcal{M}}$', size=16)

plt.tick_params(axis='both', which='major', labelsize=13)

plt.legend()
plt.show()


# -------------------------------
# 1️⃣ Funzione vectorizzata calcola_t (p11 fissato a 0)
# -------------------------------



n = 10
p1_vec = np.linspace(0, 1, n)
p2_vec = np.linspace(0, 1, n)

P1, P2 = np.meshgrid(p1_vec, p2_vec)
P3 = 1 - P1 - P2


mask = (P3 >= 0) & (P3 <= 1)


T1 = np.zeros_like(P1)
T2 = np.zeros_like(P2)
T3 = np.zeros_like(P3)

valid_P1 = P1[mask]
valid_P2 = P2[mask]
valid_P3 = P3[mask]

valid_T1, valid_T2, valid_T3 = calcola_t_vec(valid_P1, valid_P2, valid_P3)

T1[mask] = valid_T1
T2[mask] = valid_T2
T3[mask] = valid_T3


M_hs = np.full_like(P1, np.nan)  # inizializza con NaN
valid_indices = np.where(mask)

for i, j in zip(*valid_indices):
    M_hs[i, j] = M_hs_nonloc(T1[i,j], T2[i,j], T3[i,j])

plt.figure(figsize=(8,6))

levels = np.linspace(0, 1, 51)  # 50 intervalli da 0 a 1 inclusi
cp = plt.contourf(P1, P2, M_hs, levels=levels, cmap="viridis")

cbar = plt.colorbar(cp)
cbar.set_label(r'$\widetilde{\mathcal{M}}_{\mathrm{HS}}$', fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.xlabel(r'$e_{1}$', fontsize=16)
plt.ylabel(r'$e_{2}$', fontsize=16)

# Aumentare la dimensione dei numeri degli assi
plt.tick_params(axis='both', which='major', labelsize=14)  # ad esempio 14

plt.show()
