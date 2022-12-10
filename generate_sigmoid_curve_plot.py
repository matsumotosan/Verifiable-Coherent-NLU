import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def main(args):
    # Enter baseline weights (precondition, effects, conflict, story)
    lam_base = np.array([0.4, 0.4, 0.2, 0.0])

    # Calculate lambda weight schedule (continuous)
    lam = np.zeros((4, 100))
    p = np.linspace(0, args.epochs, 100)
    for idx, (l, p_thi) in enumerate(zip(args.lambda_const, args.p_th)):    
        lam[idx, :] = l * sigmoid((p - p_thi) / args.alpha)

    # Plot baseline lambda weights
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.hlines(lam_base[0], 0, args.epochs, linestyle='dashed', colors='b', alpha=0.5)
    ax.hlines(lam_base[1], 0, args.epochs, linestyle='dashed', colors='g', alpha=0.5)
    ax.hlines(lam_base[2], 0, args.epochs, linestyle='dashed', colors='r', alpha=0.5)
    ax.hlines(lam_base[3], 0, args.epochs, linestyle='dashed', colors='c', alpha=0.5)
    
    # Plot weights (continuous)
    ax.plot(p, lam[0, :], linestyle='solid', color='b', label=f'$\lambda_p\ (\lambda_p^{{const}}={{{args.lambda_const[0]}}}, p_p^{{th}}={{{args.p_th[0]}}})$')
    ax.plot(p, lam[1, :], linestyle='solid', color='g', label=f'$\lambda_e\ (\lambda_e^{{const}}={{{args.lambda_const[1]}}}, p_e^{{th}}={{{args.p_th[1]}}})$')
    ax.plot(p, lam[2, :], linestyle='solid', color='r', label=f'$\lambda_c\ (\lambda_c^{{const}}={{{args.lambda_const[2]}}}, p_c^{{th}}={{{args.p_th[2]}}})$')
    ax.plot(p, lam[3, :], linestyle='solid', color='c', label=f'$\lambda_s\ (\lambda_s^{{const}}={{{args.lambda_const[3]}}}, p_s^{{th}}={{{args.p_th[3]}}})$')

    # Figure annotations
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    ax.set_xlabel(r'Epoch - $p$')
    ax.set_ylabel(r'Task Loss Weight - $\lambda_i(p)$')
    ax.set_title(rf"Sigmoid Loss Weight Schedule (${{\alpha={{{args.alpha}}}}}$)")
    plt.grid(True)
    plt.savefig(f'figures/sigmoid_{args.lambda_const}_{args.p_th}.png', dpi=200)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot sigmoid weight loss schedule.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--lambda_const", type=float, nargs=4, default=[0.4, 0.4, 0.2, 0.1])
    parser.add_argument("--p_th", type=float, nargs=4, default=[0.0, 0.0, 5.0, 8.0])

    args = parser.parse_args()

    main(args)