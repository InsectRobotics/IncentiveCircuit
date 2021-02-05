import pandas as pd
import numpy as np
import csv
import re
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
__draft_data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "FruitflyMB", "draft"))
__dirs = {
    "MBON-γ1ped": {
        "A+": ["MBON-g1ped", "OCT+shock"],
        "B-": ["MBON-g1ped", "MCH+noshock"],
        "B+": ["MBON-g1ped", "MCH+shock"]
    },
    "MBON-γ2α'1": {
        "A+": ["MBON-g2a'1", "OCT+shock"],
        "B-": ["MBON-g2a'1", "MCH+noshock"],
        "B+": ["MBON-g2a'1", "MCH+shock"]
    },
    "MBON-γ5β'2a": {
        # "A": "",
        "B-": ["MBON-g5b'2a", "MCH+noshock"],
        "B+": ["MBON-g5b'2a", "MCH+shock"]
    },
    "PPL1-γ1ped": {
        "A+": ["PPL1-g1ped", "OCT+shock"],
        "B-": ["PPL1-g1ped", "MCH+noshock"],
        "B+": ["PPL1-g1ped", "MCH+shock"],
    },
    "PAM-β'2a": {
        "A-": ["PAM-b'2a", "OCT+noshock"],
        "A+": ["PAM-b'2a", "OCT+shock"],
        "B-": ["PAM-b'2a", "MCH+noshock"],
        "B+": ["PAM-b'2a", "MCH+shock"]
    },
    "PPL1-γ2α'1": {
        "B+": ["PPL1-g2a'1", "MCH+shock (1-9,1-8)"]
    }
}

_pattern_ = r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'


def load_draft_data():
    data = {}
    for genotype in __dirs:
        for experiment in __dirs[genotype]:
            if experiment not in data:
                data[experiment] = {genotype: []}
            data[experiment][genotype] = [[]] * 18
            for r, _, flist in os.walk(__draft_data_dir__):
                match = True
                for d in __dirs[genotype][experiment]:
                    if d not in r:
                        match = False
                if not match:
                    continue

                labels = re.findall(r'.*\(\d\-(\d),\d\-(\d)\).*', r)
                if len(labels) < 1:
                    print("Unknown directory pattern:", r)
                    continue
                nb_csm, nb_csp = labels[0]
                nb_csm, nb_csp = int(nb_csm), int(nb_csp)

                for filename in flist:
                    details = re.findall(_pattern_, filename)
                    if len(details) < 1:
                        print("Unknown file pattern:", os.path.join(r, filename))
                        continue
                    _, cs, trial = details[0]
                    trial = int(trial)

                    timepoint = None
                    with open(os.path.join(r, filename), 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                        for row in reader:
                            if timepoint is None:
                                timepoint = row
                            else:
                                timepoint = np.vstack([timepoint, row])  # {timepoint} x {datapoint}
                    a, b = "O" in cs, "M" in cs
                    csm = "B" in experiment and a or "A" in experiment and b
                    csp = "B" in experiment and b or "A" in experiment and a

                    if csp and nb_csp < 8:
                        trial += 1
                    if csm and nb_csp < 8:
                        trial += 1
                    if csm and nb_csp < 8 and 6 < trial < 9:
                        trial += 1

                    data[experiment][genotype][2 * (trial - 1) + int(csp)] = timepoint

            temp = []
            for t in range(len(data[experiment][genotype])):
                if len(data[experiment][genotype][t]) > 0:
                    temp = data[experiment][genotype][t]
                    break
            for t in range(len(data[experiment][genotype])):
                if len(data[experiment][genotype][t]) == 0:
                    data[experiment][genotype][t] = np.zeros_like(temp)
            data[experiment][genotype] = pd.DataFrame(np.concatenate(data[experiment][genotype]))

    return pd.DataFrame(data)
