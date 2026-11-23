import matplotlib.pyplot as plt
import numpy as np
import vpython as vp

##############################################WAVE & LATTICE PARAMETERS######################################################

WAVE_TYPE = "longitudinal"  # Choose "longitudinal" (along x) or "transverse" (along y)
DRIVE_MODE = "pulse"        # Choose "pulse" (single wave packet) or "continuous" (standing waves)

Nx, Ny, Nz = 24, 4, 4       # Long bar geometry so wavefront propagation is clearly visible

# Realistic Copper (Cu) atomic parameters in SI units
m = 1.055e-25               # Atomic mass of Copper (kg)
L0 = 2.56e-10               # Equilibrium spacing (m)
L_diag = np.sqrt(2) * L0    # Diagonal rest length (m)

k = 28.0                    # Nearest-neighbor spring constant (N/m)
k_diag = 0.5 * k            # Diagonal spring constant for shear rigidity (N/m)

dt = 4.0e-15                # 4 femtoseconds per step
steps = 600                 # Total simulation steps (2.4 picoseconds total)
substeps = 4                # Physics steps per rendered frame

# Driving wave parameters
amp = 0.25 * L0             # Wave amplitude
freq = 1.8e12               # Driving frequency (1.8 THz; well below ~7.3 THz cutoff)
omega = 2 * np.pi * freq

##############################################SYSTEM INITIALIZATION#########################################################

# Equilibrium grid coordinates
eq_pos = np.zeros((Nx, Ny, Nz, 3))
for ix in range(Nx):
    for iy in range(Ny):
        for iz in range(Nz):
            eq_pos[ix, iy, iz] = [ix * L0, iy * L0, iz * L0]

lattice = eq_pos.copy()
velocities = np.zeros_like(lattice)

# 18-neighbor bond definitions: (dx, dy, dz, stiffness, rest_length)
neighbor_bonds = [
    (1, 0, 0, k, L0),   (-1, 0, 0, k, L0),
    (0, 1, 0, k, L0),   (0, -1, 0, k, L0),
    (0, 0, 1, k, L0),   (0, 0, -1, k, L0),
    (1, 1, 0, k_diag, L_diag),   (1, -1, 0, k_diag, L_diag),
    (-1, 1, 0, k_diag, L_diag),  (-1, -1, 0, k_diag, L_diag),
    (1, 0, 1, k_diag, L_diag),   (1, 0, -1, k_diag, L_diag),
    (-1, 0, 1, k_diag, L_diag),  (-1, 0, -1, k_diag, L_diag),
    (0, 1, 1, k_diag, L_diag),   (0, 1, -1, k_diag, L_diag),
    (0, -1, 1, k_diag, L_diag),  (0, -1, -1, k_diag, L_diag),
]


def get_forces(pos):
    """Vectorized 18-neighbor Hookean force calculation (~50x faster than Python loops)."""
    forces = np.zeros_like(pos)
    for dx, dy, dz, k_spring, rest_len in neighbor_bonds:
        s1 = (
            slice(max(0, -dx), Nx - max(0, dx)),
            slice(max(0, -dy), Ny - max(0, dy)),
            slice(max(0, -dz), Nz - max(0, dz)),
        )
        s2 = (
            slice(max(0, dx), Nx - max(0, -dx)),
            slice(max(0, dy), Ny - max(0, -dy)),
            slice(max(0, dz), Nz - max(0, -dz)),
        )
        diff = pos[s1] - pos[s2]
        d = np.linalg.norm(diff, axis=-1, keepdims=True)
        forces[s1] += -k_spring * (d - rest_len) * (diff / d)
    return forces


def apply_boundary_driver(pos, vel, t):
    """Drives the left wall (x = 0) to launch longitudinal or transverse waves."""
    axis = 0 if WAVE_TYPE == "longitudinal" else 1
    period = 1.0 / freq

    # Reset left wall to equilibrium before applying wave displacement
    pos[0, :, :, :] = eq_pos[0, :, :, :]
    vel[0, :, :, :] = 0.0

    if DRIVE_MODE == "continuous" or t <= period:
        # Smooth Hann-windowed pulse or continuous sine wave
        envelope = 0.5 * (1 - np.cos(2 * np.pi * t / period)) if DRIVE_MODE == "pulse" else 1.0
        disp = amp * envelope * np.sin(omega * t)
        pos[0, :, :, axis] += disp


def position_update(pos, vel, forces, t):
    accel = forces / m
    pos = pos + vel * dt + 0.5 * accel * (dt**2)
    apply_boundary_driver(pos, vel, t + dt)

    new_forces = get_forces(pos)
    new_accel = new_forces / m
    vel = vel + 0.5 * (accel + new_accel) * dt
    vel[0, :, :, :] = 0.0  # Driven boundary has prescribed position
    return pos, vel, new_forces


##############################################VISUALIZATION SETUP###########################################################

center = vp.vector((Nx - 1) * L0 / 2, (Ny - 1) * L0 / 2, (Nz - 1) * L0 / 2)
scene = vp.canvas(
    title=f"3D Copper Wave Propagation ({WAVE_TYPE.capitalize()} - {DRIVE_MODE})",
    width=1100,
    height=600,
    center=center,
)

balls = [[[None for _ in range(Nz)] for _ in range(Ny)] for _ in range(Nx)]
for x in range(Nx):
    for y in range(Ny):
        for z in range(Nz):
            p = lattice[x, y, z]
            balls[x][y][z] = vp.sphere(
                pos=vp.vector(p[0], p[1], p[2]),
                radius=0.18 * L0,
                color=vp.vector(0.2, 0.5, 0.9),
            )

springs = []
primary_dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
for x in range(Nx):
    for y in range(Ny):
        for z in range(Nz):
            for dx, dy, dz in primary_dirs:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nx < Nx and ny < Ny and nz < Nz:
                    p1 = balls[x][y][z].pos
                    p2 = balls[nx][ny][nz].pos
                    bond = vp.cylinder(
                        pos=p1,
                        axis=p2 - p1,
                        radius=0.035 * L0,
                        color=vp.color.white,
                    )
                    springs.append(((x, y, z), (nx, ny, nz), bond))

##############################################SIMULATION & DATA LOGGING#####################################################

forces = get_forces(lattice)
axis_idx = 0 if WAVE_TYPE == "longitudinal" else 1

time_log = []
left_wall_disp = []
mid_bar_disp = []
right_wall_disp = []

for step in range(steps):
    t = step * dt
    lattice, velocities, forces = position_update(lattice, velocities, forces, t)

    # Record displacement at x=0 (source), x=Nx//2 (middle), and x=Nx-1 (far end)
    time_log.append((t + dt) * 1e12)  # Convert seconds to picoseconds
    left_wall_disp.append((lattice[0, 1, 1, axis_idx] - eq_pos[0, 1, 1, axis_idx]) / L0)
    mid_bar_disp.append((lattice[Nx // 2, 1, 1, axis_idx] - eq_pos[Nx // 2, 1, 1, axis_idx]) / L0)
    right_wall_disp.append((lattice[-1, 1, 1, axis_idx] - eq_pos[-1, 1, 1, axis_idx]) / L0)

    if step % substeps == 0:
        vp.rate(60)
        disp_mag = np.linalg.norm(lattice - eq_pos, axis=-1)

        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    p = lattice[x, y, z]
                    balls[x][y][z].pos = vp.vector(p[0], p[1], p[2])
                    # Shift color from blue (at rest) to bright yellow/red (high displacement)
                    intensity = min(1.0, disp_mag[x, y, z] / (0.8 * amp))
                    balls[x][y][z].color = vp.vector(0.2 + 0.8 * intensity, 0.5 + 0.4 * intensity, 0.9 - 0.8 * intensity)

        for (x1, y1, z1), (x2, y2, z2), bond in springs:
            p1 = balls[x1][y1][z1].pos
            p2 = balls[x2][y2][z2].pos
            bond.pos = p1
            bond.axis = p2 - p1

# Plot wave arrival times so you can measure wave speed (v = distance / delta_t)
plt.figure(figsize=(9, 4.5))
plt.plot(time_log, left_wall_disp, label="Source (x = 0)")
plt.plot(time_log, mid_bar_disp, label=f"Middle (x = {Nx // 2})")
plt.plot(time_log, right_wall_disp, label=f"Far End (x = {Nx - 1})")
plt.title(f"{WAVE_TYPE.capitalize()} Wave Propagation Through Copper Lattice")
plt.xlabel("Time (picoseconds)")
plt.ylabel("Normalized Displacement (Δx / L0)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

while True:
    vp.rate(30)
