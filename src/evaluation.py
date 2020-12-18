from imaging import load_draft_data

import pandas as pd
import numpy as np


cases = ["acquisition (B)", "acquisition (A)",
         "reversal (B)", "reversal (A)",
         "extinction (B)", "extinction (A)"]
behaviour = pd.DataFrame({
    # [A-, B+, A+, B-, A-, B-]
    "MBON-γ1ped": [-1., +0., +0., -1., +0., +0.],
    "MBON-γ2α'1": [-1., +0., +1., +0., +0., +0.],
    "MBON-γ5β'2a": [+1., +0., -1., +1., -.1, +0.],
    "PPL1-γ1ped": [+1., +0., +0., +0., +0., +0.],
    "PPL1-γ2α'1": [+0., +0., +0., +0., +0., +0.],
    "PAM-β'2a": [-1., +0., +1., -.1, np.NaN, +0.]
}, index=cases)
_behaviour_map = {}


def evaluate(mb_model, experiment="B+", nids=None, reversal=True, extinction=True, liyans_frames=False,
             cs_only=True, us_only=False, behav_mean=behaviour, behav_std=None, integration=np.nanmean):
    pred = {}
    models = []

    if reversal:
        # acquisition and reversal
        rev_model = mb_model.copy()
        models.append(rev_model)
        rev_model(reversal=reversal)
        rev = rev_model.as_dataframe(nids=nids, reconstruct=False)[experiment]
        for neuron in behav_mean:
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
                pred[neuron][case], _ = _get_trend(trs, cs_only=cs_only, us_only=us_only, integration=integration,
                                                   liyans_frames=liyans_frames)

    if extinction:
        ext_model = mb_model.copy()
        models.append(ext_model)
        ext_model(extinction=extinction)
        ext = ext_model.as_dataframe(nids=nids)[experiment]
        for neuron in behav_mean:
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
    target = behav_mean.copy()
    z_t = np.sqrt(np.nansum(np.square(np.array(target))))
    target = target / z_t
    if behav_std is not None:
        target_std = behav_std.copy()
    else:
        target_std = .1

    target_max = np.nanmax(np.absolute(np.array(target)))
    target_m = target / target_max
    target_s = target_std / np.square(target_max)
    print(target_s.T)
    t0 = np.exp(-np.square(target_m + 1)/(2 * target_s))
    t1 = np.exp(-np.square(target_m - 0)/(2 * target_s))
    t2 = np.exp(-np.square(target_m - 1)/(2 * target_s))
    t_all = t0 + t1 + t2

    pred_max = np.nanmax(np.absolute(np.array(pred)))
    pred_m = pred / pred_max
    p0 = np.exp(-np.square(pred_m + 1) / (2 * target_s))
    p1 = np.exp(-np.square(pred_m - 0) / (2 * target_s))
    p2 = np.exp(-np.square(pred_m - 1) / (2 * target_s))
    p_all = p0 + p1 + p2

    print(target_max, pred_max)
    # calculate accuracy = 1 - error
    target_id = np.argmax([np.array(t0), np.array(t1), np.array(t2)], axis=0)
    target_n = np.zeros_like(t0)
    target_n[target_id == 0] = np.array(t0 / t_all)[target_id == 0]
    target_n[target_id == 1] = np.array(t1 / t_all)[target_id == 1]
    target_n[target_id == 2] = np.array(t2 / t_all)[target_id == 2]
    target_n = pd.DataFrame(target_n, columns=t0.columns, index=t0.index)

    pred_n = np.zeros_like(p0)
    pred_n[target_id == 0] = np.array(p0 / p_all)[target_id == 0]
    pred_n[target_id == 1] = np.array(p1 / p_all)[target_id == 1]
    pred_n[target_id == 2] = np.array(p2 / p_all)[target_id == 2]
    pred_n = pd.DataFrame(pred_n, columns=p0.columns, index=p0.index)
    print(target_n.T)
    print(pred_n.T)

    error = p_all.copy()
    error[:] = np.absolute(target_n - pred_n)
    acc = 1 - error
    return np.nanmean(np.array(acc)), acc, pred, models


def update_gradients(cs_only=False, us_only=False, integration=np.nanmean):
    global _behaviour_map

    dff = load_draft_data()
    dfra = dff["A+"]  # reversal: MCH-, OCT+ and OCT-, MCH+
    dfrb = dff["B+"]  # reversal: OCT-, MCH+ and MCH-, OCT+
    dfcb = dff["B-"]  # control: OCT-, MCH- and MCH-, OCT-

    dfs = [dfra, dfrb, dfcb]
    o1s = ["A", "B", "B"]
    o2s = ["B", "A", "A"]
    o3s = ["A", "B", "B"]
    o4s = ["B", "A", "A"]
    m_names = [["Acq(A+)_mean_grad", "Acq(B-)_mean_grad", "Rev(A-)_mean_grad", "Rev(B+)_mean_grad"],
               ["Acq(B+)_mean_grad", "Acq(A-)_mean_grad", "Rev(B-)_mean_grad", "Rev(A+)_mean_grad"],
               ["Con(A-)_mean_grad", "Con(B-)_mean_grad", "Con_r(B-)_mean_grad", "Con_r(A+)_mean_grad"]]
    s_names = [["Acq(A+)_std_grad", "Acq(B-)_std_grad", "Rev(A-)_std_grad", "Rev(B+)_std_grad"],
               ["Acq(B+)_std_grad", "Acq(A-)_std_grad", "Rev(B-)_std_grad", "Rev(A+)_std_grad"],
               ["Con(A-)_std_grad", "Con(B-)_std_grad", "Con_r(B-)_std_grad", "Con_r(A+)_std_grad"]]

    for neuron in behaviour.columns:
        for df_, o1, o2, o3, o4, m_name, s_name in zip(dfs, o1s, o2s, o3s, o4s, m_names, s_names):
            b, s = [[], [], [], []], [[], [], [], []]

            if neuron not in _behaviour_map:
                _behaviour_map[neuron] = {}
            if not np.all(np.isnan(df_[neuron])):
                c_max, c_min = np.nanmax(np.array(df_[neuron])), np.nanmin(np.array(df_[neuron]))
                _behaviour_map[neuron]["max"] = np.maximum(_behaviour_map.get("max", c_max), c_max)
                _behaviour_map[neuron]["min"] = np.maximum(_behaviour_map.get("min", c_min), c_min)

                trials_1, trials_2, trials_3, trials_4 = [], [], [], []
                for trial in [1, 2, 3, 4, 5, 6]:
                    # A+ / B+
                    trials_1.append(_get_trial(df_[neuron], trial, odour=o1))
                    # B- / A-
                    trials_2.append(_get_trial(df_[neuron], trial, odour=o2))
                for trial in [6, 7, 8]:
                    # A- / B-
                    trials_3.append(_get_trial(df_[neuron], trial, odour=o3))
                for trial in [7, 8, 9]:
                    # B+ / A+
                    trials_4.append(_get_trial(df_[neuron], trial, odour=o4))
                b[0], s[0] = _get_gradient(trials_1, cs_only=cs_only, us_only=us_only, integration=integration)
                b[1], s[1] = _get_gradient(trials_2, cs_only=cs_only, us_only=us_only, integration=integration)
                b[2], s[2] = _get_gradient(trials_3, cs_only=cs_only, us_only=us_only, integration=integration)
                b[3], s[3] = _get_gradient(trials_4, cs_only=cs_only, us_only=us_only, integration=integration)
            else:
                b[0], s[0] = np.NaN, np.NaN
                b[1], s[1] = np.NaN, np.NaN
                b[2], s[2] = np.NaN, np.NaN
                b[3], s[3] = np.NaN, np.NaN

            for i in range(4):
                _behaviour_map[neuron][m_name[i]] = b[i]
                _behaviour_map[neuron][s_name[i]] = s[i]
        _behaviour_map[neuron]["Ext(B-)_mean_grad"] = behaviour[neuron][cases[4]]
        _behaviour_map[neuron]["Ext(A-)_mean_grad"] = behaviour[neuron][cases[5]]
        _behaviour_map[neuron]["Ext(B-)_std_grad"] = 0.1
        _behaviour_map[neuron]["Ext(A-)_std_grad"] = 0.1


def generate_behaviour_map(cs_only=False, us_only=False, integration=np.nanmean):
    dff = load_draft_data()
    dfr = dff["B+"]
    dfe = dff["B-"]
    behav, behav_s = {}, {}
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

        bs = np.array([b1, b2, b3, b4, b5, b6])
        ss = np.array([s1, s2, s3, s4, s5, s6])

        behav[neuron] = bs
        behav_s[neuron] = ss

    behav = pd.DataFrame(behav, index=cases)
    behav_s = pd.DataFrame(behav_s, index=cases)

    for neuron in behaviour.columns:
        for i, b in enumerate(behav[neuron]):
            if np.isnan(b):
                behav[neuron][i] = behaviour[neuron][cases[i]]
        for i, s in enumerate(behav_s[neuron]):
            if np.isnan(s):
                behav_s[neuron][i] = 0.1

    return behav, behav_s


def _get_gradient(trials, cs_only=True, us_only=False, integration=np.nanmean, liyans_frames=False):
    return _get_trend(trials, cs_only=cs_only, us_only=us_only, integration=integration, liyans_frames=liyans_frames,
                      axis=(-1, -2))


def _get_trend(trials, cs_only=False, us_only=False, integration=np.nanmean, liyans_frames=False, axis=None):
    if np.array(trials).shape[1] == 2:
        dst = int(1 if us_only else 0)
        det = int(1 if cs_only else 2)
    else:
        if liyans_frames:
            dst = int(45 if us_only else 25)
            det = int(44 if cs_only else 49)
        else:
            dst = int(43 if us_only else 28)
            det = int(33 if cs_only else 48)
    trends = []
    for tr1, tr2 in zip(trials[:-1], trials[1:]):
        trends.append(tr2[dst:det] - tr1[dst:det])
        # trends.append((tr2[dst:det] - tr1[dst:det]).sum())
        # std.append((tr2[dst:det] - tr1[dst:det]).var())
    return integration(trends, axis=axis), np.std(trends, axis=axis) / np.square(det - dst)


def _get_trial(data, trial_no, odour="B"):
    b = int(odour == "B")
    return np.array(data[(2*(trial_no-1)+b)*100:(2*(trial_no-1)+b+1)*100])


update_gradients(cs_only=True)


if __name__ == '__main__':
    from imaging import plot_overlap

    pd.options.display.max_columns = 6
    pd.options.display.max_rows = 6
    pd.options.display.width = 1000

    bm = generate_behaviour_map(cs_only=True)
    print("BEHAVIOUR FROM DESCRIPTION")
    print(behaviour.T)
    print("BEHAVIOUR FROM RAW DATA")
    print(bm.T)
    df = load_draft_data()
    plot_overlap(df, "B+", score=bm, individuals=False)
    # plot_overlap(df, "B-", phase2="extinction", score=bm, individuals=False)
    # plot_overlap(df, "B+", score=behaviour, zeros=True)
