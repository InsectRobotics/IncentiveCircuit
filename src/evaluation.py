from imaging import load_draft_data

import pandas as pd
import numpy as np


cases = ["acquisition (B)", "acquisition (A)",
         "reversal (B)", "reversal (A)",
         "extinction (B)", "extinction (A)"]
behaviour = {
    # [A-, B+, A+, B-, A-, B-]
    "MBON-γ1ped": [-1., +0., +0., -1., +0., +0.],
    "MBON-γ2α'1": [-1., +0., +1., +0., +0., +0.],
    "MBON-γ5β'2a": [+1., +0., -1., +1., -.1, +0.],
    "PPL1-γ1ped": [+1., +0., +0., +0., +0., +0.],
    "PPL1-γ2α'1": [+0., +0., +0., +0., +0., +0.],
    "PAM-β'2a": [-1., +0., +1., -.1, np.NaN, +0.]
}
behaviour = pd.DataFrame(behaviour, index=cases)


def evaluate(mb_model, experiment="B+", nids=None, reversal=True, extinction=True, tolerance=.1, percentage=False,
             cs_only=False, us_only=False, mbon_only=False, behav=behaviour, integration=np.mean):
    pred = {}
    models = []

    if reversal:
        # acquisition and reversal
        rev_model = mb_model.copy()
        models.append(rev_model)
        rev_model(reversal=reversal)
        rev = rev_model.as_dataframe(nids=nids, reconstruct=False)[experiment]
        for neuron in behav:
            pred[neuron] = {}
            for case in cases:
                trials = []
                if case == "acquisition (A)":
                    trials.extend([1, 2, 3, 4, 5, 6])
                    odour = "A"
                elif case == "acquisition (B)":
                    trials.extend([1, 2, 3, 4, 5, 6])
                    odour = "B"
                elif case == "reversal (A)":
                    trials.extend([7, 8, 9, 10, 11, 12, 13])
                    odour = "A"
                elif case == "reversal (B)":
                    trials.extend([6, 7, 8, 9, 10, 11, 12])
                    odour = "B"
                else:
                    continue
                for tr in trials[::-1]:
                    if tr >= rev[neuron].shape[0]:
                        trials.remove(tr)

                trs = []
                for trial in trials:
                    trs.append(rev[neuron][trial*2 + int(odour == "B")])
                pred[neuron][case], _ = _get_trend(trs, cs_only=cs_only, us_only=us_only, integration=integration)

    if extinction:
        ext_model = mb_model.copy()
        models.append(ext_model)
        ext_model(extinction=extinction)
        ext = ext_model.as_dataframe(nids=nids)[experiment]
        for neuron in behav:
            if not reversal:
                pred[neuron] = {}
            for case in cases:
                trials = []
                if case == "acquisition (A)" and not reversal:
                    trials.extend([1, 2, 3, 4, 5, 6])
                    odour = "A"
                elif case == "acquisition (B)" and not reversal:
                    trials.extend([1, 2, 3, 4, 5, 6])
                    odour = "B"
                elif case == "extinction (A)":
                    trials.extend([7, 8, 9, 10, 11, 12, 13])
                    odour = "A"
                elif case == "extinction (B)":
                    trials.extend([6, 7, 8, 9, 10, 11, 12])
                    odour = "B"
                else:
                    continue
                for tr in trials[::-1]:
                    if tr * 100 >= ext[neuron].shape[0]:
                        trials.remove(tr)
                trs = []
                for trial in trials:
                    trs.append(_get_trial(ext[neuron], trial, odour=odour))
                pred[neuron][case], _ = _get_trend(trs, cs_only=cs_only, us_only=us_only, integration=integration)

    pred = pd.DataFrame(pred)
    z_p = np.sqrt(np.nansum(np.square(np.array(pred))))
    pred /= z_p

    # calculate importance from behavioural values (zeros are important, NaNs are not important)
    target = behav.copy()
    z_t = np.sqrt(np.nansum(np.square(np.array(target))))
    targ = target / z_t

    # calculate accuracy = 1 - error
    err = pred @ targ
    total_err = np.sqrt(np.nansum(np.square(err)))
    print(total_err, np.sqrt(1 / np.array(err).size))
    acc = 1 - err
    total = pred.reshape((-1)) @ targ.reshape((-1))

    return total, acc, pred, models


def create_behaviour_map(cs_only=False, us_only=False, integration=np.nanmean):
    dff = load_draft_data()
    dfr = dff["B+"]
    dfe = dff["B-"]
    behav = {}
    for neuron in behaviour.columns:
        if not np.all(np.isnan(dfr[neuron])):
            trials_1, trials_2, trials_3, trials_4 = [], [], [], []
            # print(neuron, dfr[neuron].shape)
            for trial in [1, 2, 3, 4, 5, 6]:
                # B+
                trials_1.append(_get_trial(dfr[neuron], trial, odour="B"))
                # A-
                trials_2.append(_get_trial(dfr[neuron], trial, odour="A"))
            for trial in [6, 7, 8]:
                # B-
                trials_3.append(_get_trial(dfr[neuron], trial, odour="B"))
            for trial in [7, 8, 9]:
                # A+
                trials_4.append(_get_trial(dfr[neuron], trial, odour="A"))
            b1, s1 = _get_trend(trials_1, cs_only=cs_only, us_only=us_only, integration=integration)
            b2, s2 = _get_trend(trials_2, cs_only=cs_only, us_only=us_only, integration=integration)
            b3, s3 = _get_trend(trials_3, cs_only=cs_only, us_only=us_only, integration=integration)
            b4, s4 = _get_trend(trials_4, cs_only=cs_only, us_only=us_only, integration=integration)
        else:
            b1, s1 = np.NaN, np.NaN
            b2, s2 = np.NaN, np.NaN
            b3, s3 = np.NaN, np.NaN
            b4, s4 = np.NaN, np.NaN
        if not np.all(np.isnan(dfe[neuron])) and False:
            trials_5, trials_6 = [], []
            # print(neuron, dfe[neuron].shape)
            for trial in [6, 7, 8]:
                # B-
                trials_5.append(_get_trial(dfe[neuron], trial, odour="B"))
            for trial in [7, 8, 9]:
                # A-
                trials_6.append(_get_trial(dfe[neuron], trial, odour="A"))
            b5, s5 = _get_trend(trials_5, cs_only=cs_only, us_only=us_only, integration=integration)
            b6, s6 = _get_trend(trials_6, cs_only=cs_only, us_only=us_only, integration=integration)
        else:
            b5, s5 = np.NaN, np.NaN
            b6, s6 = np.NaN, np.NaN

        bs = np.array([b1/s1, b2/s2, b3/s3, b4/s4, b5/s5, b6/s6])

        behav[neuron] = bs

    behav = pd.DataFrame(behav, index=cases)

    for neuron in behaviour.columns:
        for i, b in enumerate(behav[neuron]):
            if np.isnan(b):
                behav[neuron][i] = behaviour[neuron][cases[i]]

    return behav


def _get_trend(trials, cs_only=False, us_only=False, integration=np.nanmean, axis=None):
    if np.array(trials).shape[1] == 2:
        dst = int(1 if us_only else 0)
        det = int(1 if cs_only else 2)
    else:
        dst = int(43 if us_only else 28)
        det = int(33 if cs_only else 48)
    trends = []
    for tr1, tr2 in zip(trials[:-1], trials[1:]):
        trends.append(tr2[dst:det] - tr1[dst:det])
        # trends.append((tr2[dst:det] - tr1[dst:det]).sum())
        # std.append((tr2[dst:det] - tr1[dst:det]).var())
    return integration(trends, axis=axis), np.std(trends, axis=axis)


def _get_trial(data, trial_no, odour="B"):
    b = int(odour == "B")
    return np.array(data[(2*(trial_no-1)+b)*100:(2*(trial_no-1)+b+1)*100])


if __name__ == '__main__':
    from imaging import plot_overlap

    pd.options.display.max_columns = 6
    pd.options.display.max_rows = 6
    pd.options.display.width = 1000

    bm = create_behaviour_map(cs_only=True)
    print("BEHAVIOUR FROM DESCRIPTION")
    print(behaviour.T)
    print("BEHAVIOUR FROM RAW DATA")
    print(bm.T)
    df = load_draft_data()
    plot_overlap(df, "B+", score=bm, individuals=False)
    # plot_overlap(df, "B-", phase2="extinction", score=bm, individuals=False)
    # plot_overlap(df, "B+", score=behaviour, zeros=True)
