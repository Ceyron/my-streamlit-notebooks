import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

with st.sidebar:
    pop_mean = st.slider("pop_mean", -5.0, 5.0, 0.0, 0.1)
    sample_mean = st.slider("sample_mean", -5.0, 5.0, 0.0, 0.1)
    sample_std = st.slider("sample_std", 0.1, 5.0, 1.0, 0.1)
    num_samples = st.slider("num_samples", 2, 100, 10, 1)
    p_value = st.slider("p_value", 0.001, 0.1, 0.05, 0.001)

mean_diff = sample_mean - pop_mean
standard_error = sample_std / np.sqrt(num_samples)
test_statistic = mean_diff / standard_error
dof = num_samples - 1

t_dist = stats.t(dof)

stat_values = np.linspace(-15, 15, 1000)

log_cdf_values = t_dist.logcdf(stat_values)
log_sf_values = t_dist.logsf(stat_values)

fig, ax = plt.subplots()
ax.plot(stat_values, log_cdf_values, label="T-Dist CDF")
ax.plot(stat_values, log_sf_values, label="T-Dist SF")
ax.hlines(np.log(p_value), -16, 16, color="red", label="p-value")
ax.vlines(test_statistic, -10.5, 1.5, color="gray", label="test statistic")
ax.legend()
ax.set_ylim(-10.5, 1.5)
ax.set_xlim(-15, 15)
ax.grid()

st.write(f"Degrees of Freedom: {dof}")
st.write(f"Test Statistic: {test_statistic:.4f}")
cdf_value = t_dist.cdf(test_statistic)
sf_value = t_dist.sf(test_statistic)
st.write(f"CDF Value: {cdf_value:.4f}")
st.write(f"SF Value: {sf_value:.4f}")

st.pyplot(fig)
