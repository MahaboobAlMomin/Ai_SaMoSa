"""Robustness evaluation stubs — used when no_robust=False (not needed for Phase 1)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def relative_robustness(robustness_result, robustness_key):
    if len(robustness_result) < 2:
        return 0.0
    return robustness_result[-1] - robustness_result[0]


def effective_robustness(robustness_result, robustness_key):
    if len(robustness_result) == 0:
        return 0.0
    return sum(robustness_result) / len(robustness_result)


def single_plot(robustness_result, robustness_key, xlabel='', ylabel='', fig_name='plot', method=''):
    plt.figure()
    plt.plot(robustness_result, marker='o', label=method)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{method} — {robustness_key}")
    plt.legend()
    plt.savefig(f"{fig_name}.png")
    plt.close()
