# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 10:15:23 2026

@author: james
"""

import numpy as np
import matplotlib.pyplot as plt


# Grid parameters
# -----------------------------

N = 150 # Size of square grid, constant
L = 1.0 # Size of one grid square, constant

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)


# Epidemiological parameters (EDITABLE PARAMETERS)
'''
So these are the editable parameters, as described in the photos
of my notes on the main group chat. beta0 is the infectious term, 
turning it up will increase the rate of new infection. gamma is
related to the recovery term, represents how many people are
recovering over time. D is the diffusion coefficient, i.e the
mobility, will increase the rate of spread with increasing D.

Default values I chose after some tinkering (so not at all
particularly scientific):
beta0 = 0.5; gamma = 1/14; D = 0.02

Feel free to edit these as you see fit.
'''
# -----------------------------

beta0 = 0.5 # infection coeff, don't increase too much otherwise it blows up
gamma = 1 / 14 # 1 / recovery
D = 0.02 # diff coeff, larger = faster spread, i.e proxy for mobility


dt = 0.1 # timestep (days)
days = 100 # length of simulation (days)


# ----------------------------- RUNNING CODE ---------------------------
'''
I have used guassians to try to model the population density of
Greater Manchester - I went on plumplot, had a look at some general
values and then tried to create guassians with different intensities
to represent each town. amp = 1 is the baseline of the relative scale,
and I've set this as what you'd expect in a suburb. As such, density of
central manchester is approximately double this, i.e amp = 2, and then
I've adjusted this accordingly for each town, and rural areas will have
amp < 1.

This may be updated to a model more realistic, based on actual grid
square data for Manchester so this is a good approximation meanwhile.

Also tried to add 'corridors' between towns as Manchester is basically
one connected metropolitan area and so town aren't really separated by
much.
'''
# ------------------------------------------------

density = 0.35*np.ones((N,N))

def gaussian(x0,y0,amp,sx,sy=None):
    if sy is None:
        sy = sx
    return amp*np.exp(-(((X-x0)**2)/(2*sx**2) + ((Y-y0)**2)/(2*sy**2)))


# Central Manchester–Salford–Trafford 'urban plateau'
# ------------------------------------------------

density += gaussian(0.48,0.55,2.0,0.28,0.20)


# Core density peaks
# ------------------------------------------------

density += gaussian(0.50,0.55,1.8,0.06)   # Manchester centre
density += gaussian(0.45,0.56,1.2,0.05)   # Salford
density += gaussian(0.48,0.48,1.0,0.06)   # Trafford


# Surrounding towns
# ------------------------------------------------

density += gaussian(0.60,0.38,1.4,0.06)   # Stockport
density += gaussian(0.36,0.72,1.2,0.07)   # Bolton
density += gaussian(0.66,0.70,1.2,0.06)   # Oldham
density += gaussian(0.55,0.86,1.1,0.06)   # Rochdale
density += gaussian(0.45,0.78,1.0,0.05)   # Bury
density += gaussian(0.20,0.60,0.9,0.09)   # Wigan
density += gaussian(0.40,0.37,1.1,0.05)   # Altrincham


# Development corridors between towns
# ------------------------------------------------

density += gaussian(0.50,0.47,0.7,0.22,0.04)   # Manchester–Stockport
density += gaussian(0.44,0.66,0.6,0.22,0.05)   # Manchester–Bolton
density += gaussian(0.56,0.70,0.6,0.20,0.05)   # Manchester–Oldham
density += gaussian(0.50,0.78,0.5,0.18,0.04)   # Oldham–Rochdale


# Pennines effect (east side less dense so need to
# try to model that)
# ------------------------------------------------

east_drop = 1 - 0.5*(X**1.5)
density *= east_drop


# Small irregularity i.e a bit of randomness to
# try to simulate the real city
# ------------------------------------------------

density *= (1 + 0.02*np.random.randn(N,N))
density = np.clip(density,0,None)

# Define full beta now that density is defined

beta = beta0 * density



# Initialise variables S, R and I
# -----------------------------

S = density.copy()
I = np.zeros((N, N))
R = np.zeros((N, N))

# single infection centre, centred approximately on
# Fallowfield for shits and gigs
I[int(N*0.47):int(N*0.49), int(N*0.49):int(N*0.51)] = 0.01
S -= I


# Laplacian
# -----------------------------

def laplacian(Z):

    L = np.zeros_like(Z)

    L[1:-1,1:-1] = (
        Z[2:,1:-1] +
        Z[:-2,1:-1] +
        Z[1:-1,2:] +
        Z[1:-1,:-2] -
        4 * Z[1:-1,1:-1]
    )

    return L


# Plot the density map at beginning
# -----------------------------

plt.figure(figsize=(6,5))
plt.imshow(density, cmap="viridis", origin = 'lower')
plt.colorbar(label="Population density")
plt.title("City Population Density Map")
plt.show()


# SIMULATION
# -----------------------------

total_infected = []
times = []

for step in range(int(days / dt)):

    diffusion = D * laplacian(I)

    infection = beta * S * I
    recovery = gamma * I

    S += -infection * dt
    I += (diffusion + infection - recovery) * dt
    R += recovery * dt

    total_infected.append(np.sum(I))
    times.append(step * dt)

    # plot every 5 days
    if step % int(2 / dt) == 0:

        plt.figure(figsize=(6,5))

        # infection heatmap
        im = plt.imshow(I, cmap="inferno", origin = 'lower')
        
        # density contours
        plt.contour(density, colors="white", linewidths=0.7)
        
        # attach colorbar to infection heatmap
        plt.colorbar(im, label="Infected density")
        
        # ---- town labels ----
        plt.text(N*0.50, N*0.52, "Manchester", color="white", ha="center")
        plt.text(N*0.44, N*0.56, "Salford", color="white", ha="center")
        plt.text(N*0.48, N*0.48, "University", color="white", ha="center")
        
        plt.text(N*0.60, N*0.38, "Stockport", color="white", ha="center")
        plt.text(N*0.36, N*0.72, "Bolton", color="white", ha="center")
        plt.text(N*0.66, N*0.70, "Oldham", color="white", ha="center")
        plt.text(N*0.55, N*0.86, "Rochdale", color="white", ha="center")
        plt.text(N*0.45, N*0.78, "Bury", color="white", ha="center")
        plt.text(N*0.20, N*0.60, "Wigan", color="white", ha="center")
        plt.text(N*0.40, N*0.37, "Altrincham", color="white", ha="center")
        
        plt.title(f"Infection spread after {step*dt:.1f} days")
        
        plt.xlabel("x position")
        plt.ylabel("y position")
        
# This part I used to get the 6 images I used to make the figure,
# un-comment it if you want to save the frames at the times
# specified
       
#        valid_times = [2, 6, 16, 28, 40, 60, 90]
#        for i in valid_times:
#            if step*dt == i:
#                plt.savefig(f'time{step*dt}infection.png', 
#                            bbox_inches = 'tight', dpi = 600)

        
        plt.show()


# Epidemic curve at the end
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(times, total_infected, color="red", linewidth=2)

plt.xlabel("Time (days)")
plt.ylabel("Total infected")
plt.title("Epidemic Curve")

# major grid
plt.grid(which='major', color='grey', linestyle='-', linewidth=0.6)

# minor grid
plt.minorticks_on()
plt.grid(which='minor', color='lightgrey', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.show()