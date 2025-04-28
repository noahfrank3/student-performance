from pathlib import Path

from git import Repo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel"))

DATA_DIR = REPO / 'data'
OUT_DIR = REPO / 'output'

def correlation_plot(data, subject_name, grade_num):
    fig, ax = plt.subplots()
    ax.scatter(data['absences'], data[f'G{grade_num}'], s=20, color='dodgerblue', zorder=3)
    ax.set_xlabel("Number of Absences", fontsize='large')
    ax.set_ylabel(f"Grade (G{grade_num})", fontsize='large')
    ax.set_title(f"Grade vs. Number of Absences, {subject_name.title()}", fontsize='large')
    ax.grid(True, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{subject_name[:4]}_g{grade_num}.svg', bbox_inches='tight')
    plt.close(fig)

rho_math = np.empty(3)
p_math = np.empty(3)
rho_port = np.empty(3)
p_port = np.empty(3)

if __name__ == '__main__':
    math_data = pd.read_excel(DATA_DIR / 'student-mat.xlsx')
    port_data = pd.read_excel(DATA_DIR / 'student-por.xlsx')
    
    for grade_num in range(1, 4):
        correlation_plot(math_data, 'math', grade_num)
        rho_math[grade_num - 1], p_math[grade_num - 1] = pearsonr(math_data['absences'], math_data[f'G{grade_num}'])

        correlation_plot(port_data, 'portuguese', grade_num)
        rho_port[grade_num - 1], p_port[grade_num - 1] = pearsonr(port_data['absences'], port_data[f'G{grade_num}'])

    corr_data = {
            'math': {
                'rho': rho_math,
                'p': p_math
            },
            'port': {
                'rho': rho_port,
                'p': p_port
            }
    }

    print(corr_data)
