from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import streamlit as st
from jaxtyping import Array

jax.config.update("jax_platform_name", "cpu")


class LinearAffineOperator(eqx.Module):
    linear_affine_effect: Callable[
        [
            Array,
        ],
        Array,
    ]
    shape: tuple

    def __init__(
        self,
        linear_affine_effect: Callable[
            [
                Array,
            ],
            Array,
        ],
        input_structure: Array,
    ):
        """
        linear_affine_effect is promised to be linear affine
        """
        self.linear_affine_effect = linear_affine_effect
        self.shape = input_structure.shape

    @property
    def zero_state(self):
        return jnp.zeros(self.shape)

    @property
    def leftover(self):
        return self.linear_affine_effect(self.zero_state)

    @property
    def rhs(self):
        return -self.leftover

    def linear_effect(self, u):
        return self.linear_affine_effect(u) - self.leftover

    def get_lin_op(self):
        return lx.FunctionLinearOperator(self.linear_effect, self.zero_state)

    def materialize_matrix(self):
        return self.get_lin_op().as_matrix()

    def solve_jax(self, *, method="gmres", **kwargs):
        if method == "gmres":
            return jax.scipy.sparse.linalg.gmres(
                self.linear_effect, self.rhs, **kwargs
            )[0]
        else:
            raise ValueError(f"Method {method} not supported")


class Helmholtz1D(eqx.Module):
    """
    Helmholtz operator in 1D
    """

    dx: float
    wavenumber: float
    left_boundary: float
    right_boundary: float

    def __call__(self, force: Array) -> LinearAffineOperator:
        def helmholtz_effect(u: Array) -> Array:
            u_padded = jnp.concatenate(
                [jnp.array([self.left_boundary]), u, jnp.array([self.right_boundary])]
            )
            u_second_derivative = (
                u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]
            ) / self.dx**2
            effect_on_u = u_second_derivative + self.wavenumber**2 * u - force
            return effect_on_u

        return LinearAffineOperator(helmholtz_effect, force)


with st.sidebar:
    domain_size = st.slider("Domain size", 0.1, 10.0, 1.0, 0.1)
    num_points = st.slider("Number of points", 10, 100, 20)
    wavenumber = st.slider("Wavenumber", -10.0, 10.0, 3.0, 0.1)
    left_boundary = st.number_input("Left boundary", value=0.0)
    right_boundary = st.number_input("Right boundary", value=0.0)

    blob_center = st.slider("Blob center", 0.0, 10.0, 0.5, 0.1)
    blob_width = st.slider("Blob width", 0.1, 1.0, 0.1, 0.1)

dof_grid = jnp.linspace(0, domain_size, num_points + 2)[1:-1]
gaussian_blob = jnp.exp(-1 / blob_width**2 * (dof_grid - blob_center) ** 2)
# first_sine_mode = jnp.sin(6*jnp.pi/domain_size * dof_grid)
force = gaussian_blob
helmholtz = Helmholtz1D(
    dx=dof_grid[2] - dof_grid[1],
    wavenumber=wavenumber,
    left_boundary=left_boundary,
    right_boundary=right_boundary,
)

lao = helmholtz(force)
mat = lao.materialize_matrix()
rhs = lao.rhs
sol = jnp.linalg.solve(mat, rhs)
res = jnp.linalg.norm(mat @ sol - rhs)
rel_res = res / jnp.linalg.norm(rhs)

st.write(f"Residual: {res:.3e}")
st.write(f"Relative residual: {rel_res:.3e}")

fig, ax = plt.subplots()
ax.plot(dof_grid, force, label="Force")
ax.plot(dof_grid, sol, label="Solution")
ax.legend()
st.pyplot(fig)
