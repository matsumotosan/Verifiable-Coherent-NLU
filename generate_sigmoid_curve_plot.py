import numpy as np
import matplotlib.pyplot as plt


def main():
    # Set training parameters
    epochs = 10
    lam_const = np.array([0.4, 0.4, 0.2, 0.1])
    p_th = np.array([0.0, 0.0, 2.0, 8.0])
    alpha = 0.9

    # Enter baseline weights (precondition, effects, conflict, story)
    lam_base = np.array([0.4, 0.4, 0.2, 0.0])

    # Calculate lambda weight schedule (continuous)
    lam = np.zeros((4, 100))
    p = np.linspace(0, epochs, 100)
    for idx, (l, p_thi) in enumerate(zip(lam_const, p_th)):    
        lam[idx, :] = l * sigmoid((p - p_thi) / alpha)

    # Plot baseline lambda weights
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.hlines(lam_base[0], 0, epochs, linestyle='dashed', colors='b', alpha=0.5)
    ax.hlines(lam_base[1], 0, epochs, linestyle='dashed', colors='g', alpha=0.5)
    ax.hlines(lam_base[2], 0, epochs, linestyle='dashed', colors='r', alpha=0.5)
    ax.hlines(lam_base[3], 0, epochs, linestyle='dashed', colors='c', alpha=0.5)
    
    # Plot weights (continuous)
    ax.plot(p, lam[0, :], linestyle='dotted', color='b', label=r'$\lambda_p\ (\lambda_p^{const}=1.0, p_p^{th}=0.0)$')
    ax.plot(p, lam[1, :], linestyle='solid', color='g', label=r'$\lambda_e\ (\lambda_e^{const}=1.0, p_e^{th}=0.0)$')
    ax.plot(p, lam[2, :], linestyle='solid', color='r', label=r'$\lambda_c\ (\lambda_c^{const}=1.0, p_c^{th}=2.0)$')
    ax.plot(p, lam[3, :], linestyle='solid', color='c', label=r'$\lambda_s\ (\lambda_s^{const}=1.0, p_s^{th}=5.0)$')

    # Figure annotations
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    ax.set_xlabel(r'Epoch - $p$')
    ax.set_ylabel(r'Loss Weight - $\lambda(p)$')
    ax.set_title(r'Sigmoid Loss Weight Schedule ($\alpha=0.9$)')
    plt.grid(True)
    plt.savefig('figures/sigmoid_weight_schedule.png', dpi=200)
    plt.show()
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    main()