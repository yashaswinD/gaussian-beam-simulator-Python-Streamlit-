import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Physics backend
# =========================

def gaussian_beam_xy(w0, wavelength, z, grid_size, extent):
    zR = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z / zR)**2)

    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    I = np.exp(-2 * (X**2 + Y**2) / wz**2)
    return x, y, I


def gaussian_beam_xz(w0, wavelength, x_extent, z_extent, grid_size):
    x = np.linspace(-x_extent, x_extent, grid_size)
    z = np.linspace(-z_extent, z_extent, grid_size)
    X, Z = np.meshgrid(x, z)

    zR = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (Z / zR)**2)

    I = np.exp(-2 * X**2 / wz**2)
    return x, z, I


# =========================
# Streamlit UI
# =========================

st.set_page_config(layout="wide")
st.title("Gaussian Beam Simulator")

st.sidebar.header("Beam Parameters")

w0 = st.sidebar.number_input("Beam waist w₀ (μm)", value=1.0)
wavelength = st.sidebar.number_input("Wavelength λ (nm)", value=1064.0)
z_pos = st.sidebar.number_input("Propagation distance z (μm)", value=10.0)

grid_size = st.sidebar.number_input("Grid points", value=300, step=50)
extent = st.sidebar.number_input("Spatial extent (μm)", value=5.0)

update = st.sidebar.button("Update Simulation")

# =========================
# Unit conversion (SI)
# =========================

w0 *= 1e-6
wavelength *= 1e-9
z_pos *= 1e-6
extent *= 1e-6

# =========================
# Derived quantities
# =========================

zR = np.pi * w0**2 / wavelength
wz = w0 * np.sqrt(1 + (z_pos / zR)**2)
I_peak = 1.0  # normalized

# =========================
# Run simulation
# =========================

if update:
    col1, col2 = st.columns(2)

    # ---------- Front profile (x–y) ----------
    x, y, Ixy = gaussian_beam_xy(w0, wavelength, z_pos, grid_size, extent)

    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(
        Ixy,
        extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
        origin="lower"
    )
    ax1.set_title("Front Profile (x–y)")
    ax1.set_xlabel("x (μm)")
    ax1.set_ylabel("y (μm)")
    ax1.grid(color="white", linestyle="--", linewidth=0.3)
    plt.colorbar(im1, ax=ax1)

    col1.pyplot(fig1)

    # ---------- Side profile (x–z) ----------
    x2, z2, Ixz = gaussian_beam_xz(w0, wavelength, extent, 3*extent, grid_size)

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(
        Ixz,
        extent=[x2[0]*1e6, x2[-1]*1e6, z2[0]*1e6, z2[-1]*1e6],
        origin="lower",
        aspect="auto"
    )
    ax2.set_title("Side Profile (x–z)")
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("z (μm)")
    ax2.grid(color="white", linestyle="--", linewidth=0.3)
    plt.colorbar(im2, ax=ax2)

    col2.pyplot(fig2)

    # ---------- Exact numerical results ----------
    st.subheader("Computed Beam Parameters")

    colA, colB, colC = st.columns(3)

    colA.metric("Rayleigh length zR (μm)", f"{zR*1e6:.3f}")
    colB.metric("Beam waist w(z) (μm)", f"{wz*1e6:.3f}")
    colC.metric("Peak intensity I(0,0)", f"{I_peak:.3f}")

    st.caption(
        "Gaussian beam modeled under the paraxial approximation with normalized intensity."
    )