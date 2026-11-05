"""Plot saved Simulation 5 results. Place beside run_case_part_2.py."""
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / 'simulation_results' / 'sim5'
    result_path = output_dir / 'sim5_result.pkl'
    if not result_path.exists():
        raise FileNotFoundError(f'Could not find {result_path}. Place this script beside run_case_part_2.py.')
    with result_path.open('rb') as file:
        result = pickle.load(file)
    names = ('directions_deg', 'max_pos_dev_lf', 'max_pos_dev_total',
             'max_psi_dev_lf', 'max_psi_dev_total')
    for name in names:
        if getattr(result, name, None) is None:
            raise ValueError(f"Saved result is missing '{name}'. Check how sim5_result.pkl was saved.")
    direction = np.asarray(result.directions_deg, dtype=float)
    order = np.argsort(direction)
    direction = direction[order]
    pos_lf = np.asarray(result.max_pos_dev_lf)[order]
    pos_total = np.asarray(result.max_pos_dev_total)[order]
    heading_lf = np.rad2deg(np.asarray(result.max_psi_dev_lf)[order])
    heading_total = np.rad2deg(np.asarray(result.max_psi_dev_total)[order])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    panels = [(pos_lf, pos_total, 'Maximum position deviation [m]', '(a) Position', 'm'),
              (heading_lf, heading_total, 'Maximum heading deviation [deg]', '(b) Heading', 'deg')]
    for ax, (lf, total, ylabel, title, unit) in zip(axes, panels):
        ax.plot(direction, lf, 'o-', label='Low-frequency motion', markersize=4)
        ax.plot(direction, total, 's-', label='Total motion', markersize=4)
        ax.axhline(5, color='black', linestyle='--', label=f'LF limit: 5 {unit}')
        ax.axhline(3, color='red', linestyle=':', label=f'Total-motion limit: 3 {unit}')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 350)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(0, 351, 30))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    axes[1].set_xlabel("Common environmental 'from' direction [deg]")
    fig.savefig(output_dir / 'sim5_deviations.pdf')
    fig.savefig(output_dir / 'sim5_deviations.png', dpi=300)
    accepted = (pos_total <= 3) & (heading_total <= 3)
    print('Directions within total-motion limits:', direction[accepted])
    print(f'Figures saved in: {output_dir}')
    plt.show()


if __name__ == '__main__':
    main()
