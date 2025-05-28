def worker(args):
    crr_model, _ = args
    trajectory = crr_model._sample_trajectory()
    return crr_model.payoff(trajectory)