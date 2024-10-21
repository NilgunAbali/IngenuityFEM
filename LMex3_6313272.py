
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from permetrics.regression import RegressionMetric

a = 13 
b = 0.13 
BC = np.array([[1],[0]])
E = 70e3
L = 500 
A = 120 

def StiffnessMLoc(E,A,L,Nelem):
    l = L/Nelem
    Kb = E*A/(3*l) * np.array([[7, -8, 1],[-8, 16, -8], [1, -8, 7]])
    return Kb

def ForceVectorLoc(a,b,L,Nelem,e):
    n = Nelem
    c1 = (b*(e-1)*L+a*n)*L/(6*n**2)
    c2 = 2*L*(b*L*(e-1/2)+a*n)/(3*n**2)
    c3 = L*(b*L*e + a*n)/(6*n**2)
    Fe = np.array([c1,c2,c3])
    return Fe

def EquilibriumEq(E, A, L, Nelem, a, b):
    Nnodes = 2*Nelem + 1 
    Kb = StiffnessMLoc(E,A,L,Nelem)
    Kg = np.zeros((Nnodes,Nnodes))
    Fg = np.zeros(Nnodes)

    for i in range(0,Nnodes-1,2):
        e = int((i+2)/2)
        Fe = ForceVectorLoc(a,b,L,Nelem,e)
        Fg[i:(i+3)] += Fe
        Kg[i : (i+3), i : (i+3)] += Kb 
    return Kg, Fg

def applyBCs(Kg, Fg, BC): 
    BC_nodes = np.array(BC[0,:] - 1, dtype = 'int64')
    BC_disp = BC[1,:]
    F_reduced = np.copy(Fg).astype(np.float64)
    for i,node in enumerate(BC_nodes):
        F_reduced -= Kg[:,node] * BC_disp[i].astype(np.float64) 
    F_reduced = np.delete(F_reduced, BC_nodes, axis = 0)
  
    Kg_reduced = np.delete(np.delete(Kg, BC_nodes, axis = 0), BC_nodes, axis = 1) 
    return Kg_reduced, F_reduced 

def DisplacementsforG(Kg, Fg, BC, Nelem): 
    Nnodes = 2 * Nelem + 1
    BC_nodes = np.array(BC[0, :] - 1, dtype='int64')  
    BC_disp = BC[1, :]
    
    Kg_reduced, F_reduced = applyBCs(Kg, Fg, BC)
    u_reduced = np.linalg.solve(Kg_reduced, F_reduced)
    
    Ug = np.zeros(Nnodes)
    
    Ug[BC_nodes] = BC_disp
    
    nonBC_nodes = np.setdiff1d(np.arange(Nnodes), BC_nodes)

    Ug[nonBC_nodes] = u_reduced
    
    return Ug


def StrainCalc(Ug, L):
    Nnode = len(Ug)
    Nelem = int((Nnode-1)/2)
    l = L/Nelem
    CoStrain = np.zeros((Nelem,2))
    for i in range(0,Nnode-1, 2):
        e = int(i/2)
        u_i = Ug[i]
        u_j = Ug[i+1]
        u_k = Ug[i+2]
        z_i = l*e 
        z_j = l*(e+1/2)
        z_k = l*(e+1)
       
        CoStrain[e][0] = 4/l**2*(u_i - 2*u_j + u_k)
        CoStrain[e][1] = -2/l**2*((z_j+z_k)*u_i - 2*(z_i+z_k)*u_j + (z_i+z_j)*u_k)
    return CoStrain

def StressCalc(Ug,L,E):
    SC = StrainCalc(Ug,L)
    return SC*E

def PlotU(ax,E, A, L, a, b, BC, Nelem, color='b'):
    sizing = 1000 
    Nnode = 2 * Nelem + 1 
    zR = np.linspace(0, L, sizing)
    Kg, Fg = EquilibriumEq(E, A, L, Nelem, a, b)
    Ug = DisplacementsforG(Kg, Fg, BC, Nelem)
    z_fem = np.linspace(0, L, Nnode)
    label = str(Nelem) + " elements"
    for e in range(Nelem):
        Z = np.array([z_fem[2 * e], z_fem[2 * e + 1], z_fem[2 * e + 2]])
        U = np.array([Ug[2 * e], Ug[2 * e + 1], Ug[2 * e + 2]])
        f = interpolate.interp1d(Z, U, kind='quadratic')
        x = np.array([t for t in zR if z_fem[2 * e] <= t <= z_fem[2 * e + 2]])
        ax.plot(x, f(x), color=color, label=label if e == 0 else "")

def plotStrain(ax,Ug,L,color='b'):
    StrainCo = StrainCalc(Ug,L)
    Nnodes = len(Ug)
    Nelem = int((Nnodes-1)/2)
    l = L/Nelem
    z_array = []
    strain_array = []
    for i in range(0, Nnodes-1, 2):
        e= int(i/2)
        z_i = l*e
        z_k = l*(e+1)
        z_array.append(z_i)
        z_array.append(z_k)
        strain_i = StrainCo[e][0]*z_i+StrainCo[e][1]
        strain_k = StrainCo[e][0]*z_k+StrainCo[e][1]
        strain_array.append(strain_i)
        strain_array.append(strain_k)
    z_array = np.array(z_array)
    strain_array = np.array(strain_array)
    label = str(Nelem) + " elements"
    ax.plot(z_array, strain_array, linestyle='-', color=color, label = label)

def plotStress(ax, Ug, L, E, color='b'):
    StressCo = StressCalc(Ug, L, E)
    Nnodes = len(Ug)
    Nelem = (Nnodes - 1) // 2
    l = L / Nelem
    z_array = np.linspace(0, L, Nelem + 1) 
    stress_array = np.zeros(Nelem + 1)
    
    for e in range(Nelem):
        stress_array[e] = StressCo[e][0] * z_array[e] + StressCo[e][1]
    
    stress_array[-1] = StressCo[-1][0] * z_array[-1] + StressCo[-1][1]

    label = f"{Nelem} elements"
    ax.plot(z_array, stress_array, linestyle='-', color=color, label=label)

    
def plotUG(ax,E,A,L, a, b, BC, CompNelem, colorOfGraphs):
    sizing = 1000
    zR = np.linspace(0, L, sizing)
    for i, Nelem in enumerate(CompNelem):
        PlotU(ax,E, A, L, a, b, BC, Nelem, color=colorOfGraphs[i%len(colorOfGraphs)])
    u_real = 1 / (E * A) * ((a * L + b * L**2 / 2) * zR 
                            - a / 2 * zR**2 
                            - b / 6 * zR**3)
    
    ax.plot(zR, u_real, label="Analytical Solution", color='k', linestyle='--')
    ax.set_xlabel('Wall distance z (mm)')
    ax.set_ylabel('Displacement U (mm)')
    ax.set_title('Fem and Analytical Comparison of Displacement')
    ax.legend(loc='best')
    ax.grid(True)
    
def PlotAllStress(ax, E, A, L, a, b, BC, CompNelem, colorOfGraphs):
    sizing = 700
    zR = np.linspace(0, L, sizing)
    stress_real = 1 / A * (a * (L - zR) + (b / 2) * (L**2 - zR**2))
    ax.plot(zR, stress_real, label="Analytical Displacement", color='k', linestyle='--')

    for i, Nelem in enumerate(CompNelem):
        Kg, Fg = EquilibriumEq(E, A, L, Nelem, a, b)
        Ug = DisplacementsforG(Kg, Fg, BC, Nelem)
        plotStress(ax, Ug, L, E, color=colorOfGraphs[i % len(colorOfGraphs)])

    ax.set_xlabel('Wall distance z (mm)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('FEM and Analytical Comparison of Stress')
    ax.legend(loc='best')
    ax.grid(True)

def PlotAllStrains(ax,E,A,L,a,b,BC,CompNelem, colorOfGraphs):
    sizing = 700
    zR = np.linspace(0, L, sizing)
    for i, Nelem in enumerate(CompNelem):
        Kg, Fg = EquilibriumEq(E, A, L, Nelem, a, b)
        Ug = DisplacementsforG(Kg,Fg,BC,Nelem)
        plotStrain(ax,Ug,L,color = colorOfGraphs[i%len(colorOfGraphs)])
    strainR = 1/(E*A)*(a*(L-zR) + b/2*(L**2 - zR**2))
    ax.plot(zR, strainR,label="Analytical Strain", color='k', linestyle='--')

    ax.set_xlabel('Wall distance z (mm)')
    ax.set_ylabel('Strain')
    ax.set_title('Fem and Analytical Comparison of Strain')
    ax.legend(loc='best')
    ax.grid(True)

def FEM(E,A,L,Nelem,a,b,BC,z):
    Kg,Fg = EquilibriumEq(E,A,L,Nelem,a,b)
    Ug = DisplacementsforG(Kg,Fg,BC,Nelem)
    StrainC = StrainCalc(Ug,L)
    StressC = StressCalc(Ug,L,E)
    l = L/Nelem
    e = -1
    if(z!=L):
        e = int(z*Nelem/L)
        z_to_interpolate = np.array([l*e, l*(e+1/2), l*(e+1)])
        u_to_interpolate = np.array([Ug[2*e], Ug[2*e + 1], Ug[2*e + 2]])
        f = interpolate.interp1d(z_to_interpolate,u_to_interpolate,kind='quadratic')
        u = f(z)
    else:
        u = Ug[-1]
    strain = StrainC[e][0]*z + StrainC[e][1]
    stress = StressC[e][0]*z + StressC[e][1]
    return np.array([u, strain, stress])

def ANALYTICAL(E,A,L,a,b,z_array):
    u =  1 / (E * A) * ((a * L + b * L**2 / 2) * z_array - a / 2 * z_array**2 - b / 6 * z_array**3)
    strain = 1/(E*A)*(a*(L-z_array) + b/2*(L**2 - z_array**2))
    stress = 1/A*(a*(L-z_array) + b/2*(L**2 - z_array**2))
    return np.array([u,strain, stress])

def convergence(ax,E, A, L, a, b, BC, CompNelem):

    zR = np.linspace(0,L,600)
    solution_analytical = ANALYTICAL(E,A,L,a,b,zR).T 
    NRMSE = np.zeros((len(CompNelem),3)) 
    for (n,Nelem) in enumerate(CompNelem):
        solution_fem = np.zeros((600,3))
        for(i,z) in enumerate(zR):
            solution_fem[i,:] = FEM(E,A,L,Nelem,a,b,BC,z)
        evaluator = RegressionMetric(solution_fem,solution_analytical)
        NRMSE[n,:] = evaluator.NRMSE() 
    ax.plot(CompNelem, NRMSE[:, 0], 'o-', color=GraphPalette[0], label='Displacement', markersize=8, linewidth=2)
    ax.plot(CompNelem, NRMSE[:, 1], 's--', color=GraphPalette[1], label='Stress', markersize=8, linewidth=2)  
    ax.plot(CompNelem, NRMSE[:, 2], 'd:', color=GraphPalette[2], label='Strain', markersize=8, linewidth=2)  

    for i in range(len(CompNelem)):
        ax.text(CompNelem[i], NRMSE[i,0], f'({CompNelem[i]},{NRMSE[i,0]*100:.2f}%)', fontsize=9, ha='left')
        ax.text(CompNelem[i], NRMSE[i,1], f'({CompNelem[i]},{NRMSE[i,1]*100:.2f}%)', fontsize=9, ha='left')
    ax.set_xlabel('Element Number')
    ax.set_ylabel('NRMSE')
    ax.set_title('NRMSE of Analytical and FEM')
    ax.legend(loc='best')
    ax.grid(True)


CompNelem = np.array([1,2,4,6,8])
GraphPalette =['#FF5733', '#33FF57', '#3357FF', '#FFC300', '#DAF7A6', '#581845', '#900C3F']


fig, axs = plt.subplots(2, 2, figsize=(75, 75))
convergence(axs[1][1],E, A, L, a, b, BC, CompNelem) 
plotUG(axs[1][0],E,A,L,a,b,BC,CompNelem,GraphPalette) 
PlotAllStrains(axs[0][1],E,A,L,a,b,BC,CompNelem,GraphPalette) 
PlotAllStress(axs[0][0],E,A,L,a,b,BC,CompNelem,GraphPalette) 
plt.show()
