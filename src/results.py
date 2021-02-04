def run_main_experiments(mb_model, reversal=True, unpaired=True, no_shock=True):
    models = []

    if reversal:
        # run acquisition and reversal phases
        rev_model = mb_model.copy()
        models.append(rev_model)
        rev_model(reversal=reversal)

    if unpaired:
        # run acquisition and  unpaired phases
        unp_model = mb_model.copy()
        models.append(unp_model)
        unp_model(unpaired=unpaired)

    if no_shock:
        # run acquisition and  no-shock phases
        nsk_model = mb_model.copy()
        models.append(nsk_model)
        nsk_model(no_shock=no_shock)

    return models


def run_custom_routine(mb_model, routine):
    new_model = mb_model.copy()
    new_model(routine=routine(new_model))
    return new_model

