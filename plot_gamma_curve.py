import pickle
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def main(args):
    # Read gamma history file
    with open(args.file, 'rb') as f:
        gamma = np.asarray(pickle.load(f))

    # Parse into gamma_conflicts and gamma_history
    gamma_conflicts = gamma[:, 0]
    gamma_stories = gamma[:, 1]
    e = np.linspace(0, args.epochs, gamma.shape[0])
    
    # Plot gammas
    fig, ax = plt.subplots(nrows=2, figsize=(10, 4), sharex=True, tight_layout=True)
    ax[0].plot(e, gamma_conflicts, label=r'$\gamma_c$', linewidth=0.5)
    ax[1].plot(e, gamma_stories, label=r'$\gamma_s$', linewidth=0.5)

    # Figure annotations
    ax[0].set_xlabel(r'Epoch')
    ax[1].set_xlabel(r'Epoch')
    ax[0].set_ylabel(r'$\gamma_c$')
    ax[1].set_ylabel(r'$\gamma_s$')
    ax[0].set_title(rf"$\gamma_c$ Loss Weight Schedule")
    ax[1].set_title(rf"$\gamma_s$ Loss Weight Schedule")

    plt.savefig(f'figures/gamma_schedule_3.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot sigmoid weight loss schedule.")

    parser.add_argument("--file", type=str, default='gamma_history_1.pkl')
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    main(args)