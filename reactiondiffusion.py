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
beta0 = 1.1; gamma = 1/14; D = 0.15

Feel free to edit these as you see fit.
'''
# -----------------------------

beta0 = 0.48 # infection coeff, don't increase too much otherwise it blows up
gamma = 1 / 7 # 1 / recovery
D = 0.5 # diff coeff, larger = faster spread, i.e proxy for mobility


dt = 0.05 # timestep (days)
days = 150 # length of simulation (days)


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

# Real densities (people per km^2)
densities_real = {
    "Manchester": 4773,
    "Salford": 2777,
    "Trafford": 2217,
    "Stockport": 2338,
    "Bolton": 2117,
    "Bury": 1949,
    "Tameside": 2240,
    "Oldham": 1701,
    "Rochdale": 1416,
    "Wigan": 1750
}

# Normalise (relative to Manchester)
max_density = max(densities_real.values())
densities = {k: v / max_density for k, v in densities_real.items()}

density = 0.2*np.ones((N,N))  # low baseline (rural-ish)

def gaussian(x0,y0,amp,sx,sy=None):
    if sy is None:
        sy = sx
    return amp*np.exp(-(((X-x0)**2)/(2*sx**2) + ((Y-y0)**2)/(2*sy**2)))


# Core towns 
# ------------------------------------------------

density += gaussian(0.50,0.55,densities["Manchester"],0.04)
density += gaussian(0.44,0.565,densities["Salford"],0.05)
density += gaussian(0.48,0.48,densities["Trafford"],0.06)

density += gaussian(0.60,0.38,densities["Stockport"],0.06)
density += gaussian(0.36,0.72,densities["Bolton"],0.07)
density += gaussian(0.66,0.70,densities["Oldham"],0.06)
density += gaussian(0.55,0.86,densities["Rochdale"],0.06)
density += gaussian(0.45,0.78,densities["Bury"],0.05)
density += gaussian(0.20,0.60,densities["Wigan"],0.09)
density += gaussian(0.62,0.57,densities["Tameside"],0.07)


# corridors, manchester is continuous
# ------------------------------------------------

density += gaussian(0.55,0.47,0.5,0.22,0.04)
density += gaussian(0.49,0.66,0.4,0.22,0.05)
density += gaussian(0.61,0.70,0.4,0.20,0.05)
density += gaussian(0.55,0.78,0.3,0.18,0.04)


# edges less dense, reduce edge density
# ------------------------------------------------

# distance from Manchester centre
r2 = (X - 0.5)**2 + (Y - 0.55)**2

# smooth radial decay
edge_drop = np.exp(-r2 / 0.25)

density *= edge_drop

# a bit of randomness to increase realism to city sprawl
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
I[int(N*0.51):int(N*0.52), int(N*0.53):int(N*0.54)] = 0.02
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
snapshot_times = [4, 8, 12, 18, 26, 36, 50, 70]
snapshots = []

total_infected = []
times = []

for step in range(int(days / dt)):

    diffusion = D * laplacian(I)

    infection = beta * S * I
    recovery = gamma * I

    S += -infection * dt
    I += (diffusion + infection - recovery) * dt
    R += recovery * dt

    total_infected.append(np.sum(I * max_density * 0.8))
    times.append(step * dt)
    
    # store snapshots
    for t_snap in snapshot_times:
        if abs((step*dt) - t_snap) < dt/2:
            snapshots.append(((step*dt), I.copy()))

    # plot every 2 days
    if step % int(2 / dt) == 0:

        plt.figure(figsize=(6,5))

        # infection heatmap
        im = plt.imshow(I, cmap="inferno", origin = 'lower')
        
        # density contours
        plt.contour(density, colors="white", linewidths=0.7)
        
        # attach colorbar to infection heatmap
        plt.colorbar(im, label="Infected density")
        
        # ---- town labels ----
        plt.text(N*0.50, N*0.55, "Manchester", color="white", ha="center")
        plt.text(N*0.44, N*0.565, "Salford", color="white", ha="center")
        plt.text(N*0.38, N*0.48, "Trafford", color="white", ha="center")
        
        plt.text(N*0.55, N*0.38, "Stockport", color="white", ha="center")
        plt.text(N*0.36, N*0.72, "Bolton", color="white", ha="center")
        plt.text(N*0.66, N*0.70, "Oldham", color="white", ha="center")
        plt.text(N*0.55, N*0.86, "Rochdale", color="white", ha="center")
        plt.text(N*0.45, N*0.78, "Bury", color="white", ha="center")
        plt.text(N*0.20, N*0.60, "Wigan", color="white", ha="center")
        plt.text(N*0.62, N*0.58, "Tameside", color="white", ha="center")
                
        plt.title(f"Infection spread after {step*dt:.1f} days")
        
        plt.xlabel("x position")
        plt.ylabel("y position")
             
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


# Figure creation
# --------------------------------
fig, axes = plt.subplots(2, 4, figsize=(14, 6))

axes = axes.flatten()

vmax = max(np.max(snap[1]) for snap in snapshots)

for i, (t, I_snap) in enumerate(snapshots):
    ax = axes[i]
    
    im = ax.imshow(I_snap, cmap="inferno", origin='lower', vmin=0, vmax=vmax)
    
    # density contours
    ax.contour(density, colors="white", linewidths=0.5)
    
    if i == 0:
        ax.text(N*0.50, N*0.55, "Manchester", color="white", ha="center")
        ax.text(N*0.44, N*0.565, "Salford", color="white", ha="center")
        ax.text(N*0.38, N*0.48, "Trafford", color="white", ha="center")
        
        ax.text(N*0.55, N*0.38, "Stockport", color="white", ha="center")
        ax.text(N*0.36, N*0.72, "Bolton", color="white", ha="center")
        ax.text(N*0.66, N*0.70, "Oldham", color="white", ha="center")
        ax.text(N*0.55, N*0.86, "Rochdale", color="white", ha="center")
        ax.text(N*0.45, N*0.78, "Bury", color="white", ha="center")
        ax.text(N*0.20, N*0.60, "Wigan", color="white", ha="center")
        ax.text(N*0.62, N*0.58, "Tameside", color="white", ha="center")
    
    ax.set_title(f"t = {t:.0f} days")
    ax.set_xticks([])
    ax.set_yticks([])

# shared colorbar
# create space on the right
plt.tight_layout(rect=[0, 0, 0.88, 1])
fig.subplots_adjust(right=0.88)

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Infected density")
plt.savefig('Infection_density_figure.png', dpi = 1000)

print(density)