import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

with st.sidebar:
    num_samples = st.slider("num_samples", 2, 100, 10, 1)
    mean_diff = st.slider("mean_diff", -5.0, 0.0, -2.0, 0.1)
    pooled_std = st.slider("pooled_std", 0.1, 5.0, 1.0, 0.1)

    p_value = st.slider("p_value", 0.0, 0.1, 0.05, 0.001)

df = 2 * num_samples - 2
test_statistic = mean_diff / (pooled_std * np.sqrt(2 / num_samples))

t_dist = stats.t(df)

stat_values = np.linspace(-10, 1, 1000)

log_cdf_values = t_dist.logcdf(stat_values)

fig, ax = plt.subplots()
ax.plot(stat_values, log_cdf_values, label="T-Dist CDF")
ax.hlines(np.log(p_value), -16, 1, color="red", label="p-value")
ax.vlines(test_statistic, -10.5, 1.5, color="gray", label="test statistic")
ax.legend()
ax.set_ylim(-10.5, 1.5)
ax.set_xlim(-15, 1.5)
ax.grid()

st.write(f"Degrees of Freedom: {df}")
st.write(f"Test Statistic: {test_statistic}")

found_prob = t_dist.cdf(test_statistic)
st.write(found_prob)
res = stats.ttest_ind_from_stats(
    0.0, pooled_std, num_samples, mean_diff, pooled_std, num_samples, alternative="less"
)
st.write(res)

st.pyplot(fig)
