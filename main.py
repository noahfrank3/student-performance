from pathlib import Path

from git import Repo
import numpy as np
import pandas as pd

def load_data()
    repo = Repo('.', search_parent_directories=True)
    repo = Path(repo.git.rev_parse("--show-toplevel"))

    data_dir = repo / 'data'
    data_file = data_dir / 'data.npz'

    try:
        data = np.load(data_file)
    except FileNotFoundError:
        def get_xlsx_data(filename):
            path = data_dir / filename
            df = pd.read_excel(path)

            absences = df['absences'].to_numpy()
            grades = df['G3'].to_numpy()

            return absences, grades

        math_absences, math_grades = get_xlsx_data('student-mat.xlsx')
        port_absences, port_grades = get_xlsx_data('student-por.xlsx')

        np.savez(
                data_file,
                math_absences=math_absences,
                math_grades=math_grades,
                port_absences=port_absences,
                port_grades=port_grades
        )
        data = np.load(data_file)

    return data

if __name__ == '__main__':
    data = load_data()
