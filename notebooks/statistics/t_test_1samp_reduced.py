import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

stat_range = np.linspace(-5, 0, 100)

NUM_SAMPLES_APPROX = 10
log_slope, log_offset = np.polyfit(
    stat_range, stats.t(NUM_SAMPLES_APPROX - 1).logcdf(stat_range), 1
)
log_approx_fn = lambda x: log_slope * x + log_offset

with st.sidebar:
    exact = st.checkbox("Exact", value=False)
    mean_diff = st.slider("mean_diff", -5.0, 0.0, -1.0, 0.1)
    sample_std = st.slider("sample_std", 0.1, 5.0, 1.0, 0.1)
    num_samples = st.slider("num_samples", 2, 100, 10, 1)

    what_to_vary = st.radio("What to vary?", ["mean_diff", "sample_std", "num_samples"])

    p_value = st.slider("p_value", 0.001, 0.1, 0.05, 0.001)

if what_to_vary == "mean_diff":
    mean_diff = np.linspace(-5, 0, 100)
    x_range = mean_diff
elif what_to_vary == "sample_std":
    sample_std = np.linspace(0.1, 5, 100)
    x_range = sample_std
elif what_to_vary == "num_samples":
    num_samples = np.linspace(2, 100, 100)
    x_range = num_samples

standard_error = sample_std / np.sqrt(num_samples)
test_statistic = mean_diff / standard_error

if exact:
    log_cdf_fn = lambda x: stats.t(num_samples - 1).logcdf(x)
else:
    log_cdf_fn = log_approx_fn

log_cdf_values = log_cdf_fn(test_statistic)

fig, ax = plt.subplots()
ax.plot(x_range, log_cdf_values, label="T-Dist CDF")
ax.hlines(
    np.log(p_value), np.min(x_range), np.max(x_range), color="red", label="p-value"
)
ax.set_xlabel(what_to_vary)
ax.set_ylabel("Log CDF Value")
ax.set_ylim(-10.5, 1.5)
ax.grid()

ax.legend()

st.pyplot(fig)
