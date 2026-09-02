import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

m = 1 # kg, keep it SI
k = 100 # N/m, also keeping it SI

omega_guess = np.sqrt(k/m)
p0 = [1.0, omega_guess, 0.0]

solve_array = np.array(np.linspace(0,2*np.pi,1000))
#print(solve_array)

def ivp(Y, t):
    return [Y[1], -k/m * Y[0]]

def fitfunction(t, amplitude, omega,phi):
    return amplitude * np.cos(omega * t + phi)

displacement_solution = integrate.odeint(ivp, [0,1], solve_array)[:,[0]]
displacement_solution = np.array(displacement_solution).reshape(-1)
#print(displacement_solution)

params = curve_fit(fitfunction, solve_array, displacement_solution, p0 = p0)[0]

amplitude = params[0]
omega = params[1]
frequency = omega / (2*np.pi)
phi = params[2]

print("Numerical frequeny: ", frequency, "Hz")
print("Analytical frequency:", np.sqrt(k/m)/(2*np.pi), "Hz")

plt.plot(solve_array, fitfunction(solve_array, amplitude, omega, phi))
plt.grid()
plt.show()





