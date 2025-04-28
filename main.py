from pathlib import Path

from git import Repo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

REPO = Path(Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel"))

DATA_DIR = REPO / 'data'
OUT_DIR = REPO / 'output'

def correlation_plot(data, subject_name, grade_num):
    fig, ax = plt.subplots()
    ax.scatter(data['absences'].to_numpy(), data[f'G{grade_num}'].to_numpy(), s=20, color='dodgerblue', zorder=3)
    ax.set_xlabel("Number of Absences", fontsize='large')
    ax.set_ylabel(f"Grade (G{grade_num})", fontsize='large')
    ax.set_title(f"Grade G{grade_num} vs. Number of Absences, {subject_name.title()}", fontsize='large')
    ax.grid(True, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{subject_name[:4]}_g{grade_num}.svg', bbox_inches='tight')
    plt.close(fig)

def regression_plot(x, y, t, L, M, U):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=20, color='dodgerblue', zorder=3, label='Data')
    ax.plot(t, L, color='hotpink', zorder=5)
    ax.plot(t, M, linewidth=2, color='magenta', zorder=5, label='Regression Line')
    ax.plot(t, U, color='hotpink', zorder=5)
    ax.fill_between(t, L, U, color='hotpink', alpha=0.25, label='95% Prediction Interval')
    ax.set_xlabel("Number of Absences", fontsize='large')
    ax.set_ylabel(f"Grade (G{grade_num})", fontsize='large')
    ax.set_title(f"Grade G{grade_num} vs. Number of Absences, Portuguese", fontsize='large')
    ax.grid(True, zorder=0)
    ax.legend(fontsize='large')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'port_g{grade_num}_reg.svg', bbox_inches='tight')
    plt.close(fig)

def pearsonr_wrapper(data, subject_name, grade_num):
    x = data['absences'].to_numpy()
    y = data[f'G{grade_num}'].to_numpy()
    print(f"{subject_name.title()}, G{grade_num}")

    rho, p = pearsonr(x, y)
    print(f"Correlation coefficient: {rho:.3g}")
    print(f"p-value: {p:.3g}")

    ci = pearsonr(x, y, alternative='two-sided').confidence_interval()
    print(f"Confidence interval: [{ci.low:.3g}, {ci.high:.3g}]")

    print()

def regression(data, subject_name, grade_num):
    x = data['absences'].to_numpy()
    y = data[f'G{grade_num}'].to_numpy()

    x_0 = x
    y_0 = y
    print(f"{subject_name.title()}, G{grade_num}")

    t = np.linspace(np.min(x), np.max(x), 1001)

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    c_0 = model.params[0]
    c_1 = model.params[1]
    cis = model.conf_int()

    print(f"Intercept: {c_0:.3g}")
    print(f"Intercept CI: [{cis[0, 0]:.3g}, {cis[0, 1]:.3g}]")
    print(f"Slope: {c_1:.3g}")
    print(f"Slope CI: [{cis[1, 0]:.3g}, {cis[1, 1]:.3g}]")

    M = c_0 + c_1*t
    t_1 = sm.add_constant(t)
    pred = model.get_prediction(t_1).summary_frame(alpha=0.05)
    L = pred['obs_ci_lower'].to_numpy()
    U = pred['obs_ci_upper'].to_numpy()

    regression_plot(x_0, y_0, t, L, M, U)

    print()

if __name__ == '__main__':
    math_data = pd.read_excel(DATA_DIR / 'student-mat.xlsx')
    port_data = pd.read_excel(DATA_DIR / 'student-por.xlsx')
    
    # Correlation test
    for grade_num in range(1, 4):
        correlation_plot(math_data, 'math', grade_num)
        pearsonr_wrapper(math_data, 'math', grade_num)

        correlation_plot(port_data, 'portuguese', grade_num)
        pearsonr_wrapper(port_data, 'portuguese', grade_num)

    # Regression
    for grade_num in range(1, 4):
        regression(port_data, 'portuguese', grade_num)
