import os
"""R3 vision figures and retained R2.1 language results; no training or scoring.

Usage: python support/make_revision3_figures.py
Portable inputs live in support/revision3_data and support/revision2_data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'support/revision3_data'
OLD = ROOT / 'support/revision2_data'
FIG = ROOT / 'figures'
COL = {2: '#176B91', 4: '#C46A20'}
plt.rcParams.update({
    'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amssymb}',
    'axes.unicode_minus': False, 'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 8.5, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white'})

PLOT_ROWS = []


def interval(values):
    a = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(a)) and len(a) in (3, 5)
    mean = float(a.mean())
    half = float(t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a)))
    return mean, mean-half, mean+half


def check_interval(values, row, lower='low', upper='high'):
    result = interval(values)
    np.testing.assert_allclose(result, [row['mean'], row[lower], row[upper]], rtol=1e-11, atol=1e-12)
    return result


def finish(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in (['pdf', 'svg', 'png'] + (['pgf'] if os.environ.get('SNN3_EXPORT_PGF') == '1' else [])):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight', pad_inches=.05, dpi=220)
    plt.close(fig)


def p_values(pairs, init, condition, K, T, contrast='P'):
    q = pairs[(pairs.init_id == init) & (pairs.condition == condition) &
              (pairs.updates == K) & (pairs.infer_T == T) & (pairs.contrast == contrast)]
    q = q.sort_values('seed')
    assert len(q) == {'A': 5, 'B': 3}[init]
    return q.value.to_numpy()


def forest(ax, y, vals, mean, low, high, T):
    # Window identity also uses marker shape, retaining meaning in grayscale.
    marker = 'o' if T == 2 else '^'
    ax.scatter(vals, y + np.linspace(-.12, .12, len(vals)), s=13,
               c=COL[T], marker=marker, alpha=.55, zorder=3)
    ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt=marker,
                ms=4.5, c=COL[T], lw=1.35, capsize=2.5, zorder=4)


def target_effects(pairs, ints):
    language = pd.read_csv(OLD / 'paired_contrasts.csv')
    language_ci = pd.read_csv(OLD / 'contrast_intervals.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), layout='constrained')
    for j, (task, groups) in enumerate([
        ('language', [(512, 2), (512, 4), (4096, 2), (4096, 4)]),
        ('vision', [(512, 2), (512, 4), (1024, 2), (1024, 4), (2048, 2), (2048, 4)])]):
        ax = axes[j]
        ys = list(reversed(range(len(groups))))
        for y, (K, T) in zip(ys, groups):
            if task == 'language':
                vals = language[(language.task == task) & (language.calibration == 'raw') &
                                (language.updates == K)].sort_values('seed')[f'P{T}'].to_numpy()
                r = language_ci[(language_ci.task == task) & (language_ci.calibration == 'raw') &
                                (language_ci.updates.astype(str) == str(K)) &
                                (language_ci.contrast == f'P{T}')].iloc[0]
                mean, lo, hi = check_interval(vals, r, 'ci95_low', 'ci95_high')
            else:
                vals = p_values(pairs, 'A', 'raw', K, T)
                r = ints[(ints.init_id == 'A') & (ints.condition == 'raw') &
                         (ints.updates == K) & (ints.infer_T == T) & (ints.contrast == 'P')].iloc[0]
                mean, lo, hi = check_interval(vals, r)
            forest(ax, y, vals, mean, lo, hi, T)
            PLOT_ROWS.append(dict(figure='target_effects', task=task, init_id='A' if task == 'vision' else 'R2.1',
                                  condition='raw', updates=K, infer_T=T, mean=mean, low=lo, high=hi, n=len(vals)))
        ax.axvline(0, c='#666666', ls='--', lw=.8)
        for split in np.arange(1.5, len(groups)-1, 2):
            ax.axhline(split, c='#DDDDDD', lw=.65)
        ax.set(yticks=ys, yticklabels=[f'{K:,}: $P_{T}$' for K, T in groups],
               xlabel='Reused minus target-adapted CE', ylim=(-.48, len(groups)-.52),
               title=['(a) SmoothSpike / WikiText-2', '(b) SpikingResformer / CIFAR-100'][j])
        ax.grid(axis='x', alpha=.16, lw=.6)
        ax.set_axisbelow(True)
    finish(fig, 'revision3_target_effects')


def bn_effects(pairs):
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.0), layout='constrained',
                             gridspec_kw={'width_ratios': [1.2, 1, 1]})
    ax = axes[0]
    groups = [(K, T) for K in [512, 1024, 2048] for T in [2, 4]]
    for y, (K, T) in zip(reversed(range(6)), groups):
        raw = p_values(pairs, 'A', 'raw', K, T)
        bn = p_values(pairs, 'A', 'bn', K, T)
        vals = bn - raw
        mean, lo, hi = interval(vals)
        forest(ax, y, vals, mean, lo, hi, T)
        PLOT_ROWS.append(dict(figure='bn_effects', task='vision', init_id='A',
                              condition='bn_minus_raw', updates=K, infer_T=T,
                              mean=mean, low=lo, high=hi, n=5))
    ax.axvline(0, c='#777777', ls='--', lw=.7)
    for y in [1.5, 3.5]:
        ax.axhline(y, c='#DDDDDD', lw=.6)
    ax.set(yticks=list(reversed(range(6))), yticklabels=[f'{K:,}: $P_{T}$' for K, T in groups],
           ylim=(-.5, 5.5), xlim=(-.075, .042), xticks=[-.06, -.03, 0, .03],
           xlabel=r'$P_e^{\mathrm{BN}}-P_e^{\mathrm{raw}}$ (CE)', title='(a) Calibration effect')
    ax.grid(axis='x', alpha=.16)
    plane_rows = []
    for ax, K, letter in zip(axes[1:], [512, 2048], ['b', 'c']):
        raw_x = p_values(pairs, 'A', 'raw', K, 4)
        bn_x = p_values(pairs, 'A', 'bn', K, 4)
        raw_y = p_values(pairs, 'A', 'raw', K, 4, 'accuracy_target_minus_reuse_pp')
        bn_y = p_values(pairs, 'A', 'bn', K, 4, 'accuracy_target_minus_reuse_pp')
        ax.axvspan(-.045, 0, color='#F0F0F0', zorder=0)
        for i, seed in enumerate([9621, 9622, 9623, 9624, 9625]):
            ax.annotate('', xy=(bn_x[i], bn_y[i]), xytext=(raw_x[i], raw_y[i]),
                        arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=.7, shrinkA=3, shrinkB=3))
            for condition, xs, ys, fill in [('raw', raw_x, raw_y, 'white'), ('bn', bn_x, bn_y, '#444444')]:
                ax.plot(xs[i], ys[i], 'o', ms=3.6, mec='#444444', mfc=fill, mew=.8)
                plane_rows.append(dict(updates=K, seed=seed, condition=condition,
                                       P4=float(xs[i]), accuracy_target_minus_reuse_pp=float(ys[i])))
        for xs, ys, fill in [(raw_x, raw_y, 'white'), (bn_x, bn_y, '#111111')]:
            ax.plot(xs.mean(), ys.mean(), 'D', ms=5.3, mec='#111111', mfc=fill, mew=1, zorder=5)
        ax.axvline(0, c='#777777', ls='--', lw=.7)
        ax.axhline(0, c='#777777', ls='--', lw=.7)
        ax.set(xlim=(-.045, .045), ylim=(-.13, 2.25), xticks=[-.04, 0, .04], yticks=[0, 1, 2],
               title=f'({letter}) {K:,} updates', xlabel='$P_4$: reused minus target CE')
        ax.grid(alpha=.12)
    axes[1].set_ylabel('Target minus reused accuracy (pp)')
    axes[2].legend(handles=[Line2D([], [], marker='o', mfc='white', mec='#444444', ls='', label='Raw'),
                           Line2D([], [], marker='o', c='#444444', ls='', label='BN'),
                           Line2D([], [], marker='D', c='#111111', ls='', label='Mean')],
                   frameon=False, loc='upper right', fontsize=7.5, handletextpad=.3,
                   labelspacing=.25)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.xaxis.label.set_fontsize(8)
    pd.DataFrame(plane_rows).to_csv(DATA / 'bn_metric_plane.csv', index=False)
    finish(fig, 'revision3_bn_effects')


def robustness(pairs, ints, transfer):
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.95), layout='constrained',
                             gridspec_kw={'height_ratios': [1, 1.35]})
    for j, init in enumerate(['A', 'B']):
        ax = axes[0, j]
        for condition, marker, fill, style in [('raw', 'o', 'white', '--'), ('bn', 's', '#222222', '-')]:
            values = []
            lows = []
            highs = []
            for K in [512, 1024, 2048]:
                a = p_values(pairs, init, condition, K, 4)
                row = ints[(ints.init_id == init) & (ints.condition == condition) &
                           (ints.updates == K) & (ints.infer_T == 4) & (ints.contrast == 'P')].iloc[0]
                mean, lo, hi = check_interval(a, row)
                values.append(mean); lows.append(mean-lo); highs.append(hi-mean)
            ax.errorbar([512, 1024, 2048], values, yerr=[lows, highs], marker=marker,
                        mfc=fill, mec='#222222', color='#222222', ls=style, lw=1,
                        capsize=3, ms=4.5, label='Raw' if condition == 'raw' else 'Matched-window BN')
        ax.axhline(0, c='#888888', lw=.7, ls=':')
        ax.set(xlim=(380, 2190), xticks=[512, 1024, 2048], xticklabels=['512', '1,024', '2,048'],
               ylim=(-.045, .065), yticks=[-.04, 0, .04], xlabel='Updates per adaptation',
               ylabel='$P_4$ (CE)', title=f'({chr(97+j)}) Initialization {init} ($n={5 if init=="A" else 3}$)')
        ax.grid(alpha=.14)
        ax.legend(frameon=False, loc='upper right', fontsize=7.5, handlelength=2, labelspacing=.3)
        ax = axes[1, j]
        categories = [('shared_initial', np.nan, 2, '$S_0$, $e=2$'),
                      ('shared_initial', np.nan, 4, '$S_0$, $e=4$'),
                      ('adapted', 2, 2, '$S_2$, $e=2$'),
                      ('adapted', 2, 4, '$S_2$, $e=4$'),
                      ('adapted', 4, 2, '$S_4$, $e=2$'),
                      ('adapted', 4, 4, '$S_4$, $e=4$')]
        for y, (state, adapt, e, label) in zip(reversed(range(6)), categories):
            for offset, cal, marker, fill in [(.20, 9640, 'o', 'white'),
                                               (0, 9740, 's', '#888888'),
                                               (-.20, 9840, '^', '#111111')]:
                q = transfer[(transfer.init_id == init) & (transfer.state == state) &
                             (transfer.infer_T == e) & (transfer.cal_seed == cal) &
                             (transfer.metric == 'CE_other_cal_minus_matched')]
                if state == 'adapted':
                    q = q[q.adapt_T == adapt]
                assert len(q) == 1
                r = q.iloc[0]
                if r['n'] > 1:
                    ax.errorbar(r['mean'], y+offset,
                                xerr=[[r['mean']-r.low], [r.high-r['mean']]],
                                fmt=marker, mfc=fill, mec='#222222', color='#555555',
                                capsize=2, ms=3.5, lw=.8)
                else:
                    ax.plot(r['mean'], y+offset, marker, mfc=fill, mec='#222222', ms=3.5)
        ax.axvline(0, c='#777777', ls='--', lw=.7)
        for split in [1.5, 3.5]:
            ax.axhline(split, c='#DDDDDD', lw=.6)
        ax.set(yticks=list(reversed(range(6))), yticklabels=[c[3] for c in categories],
               ylim=(-.6, 5.6), xlim=(-.085, .165), xticks=[-.05, 0, .05, .10, .15],
               xlabel='Other-window minus matched-window CE\n'
                      'Negative: other-window statistics lower CE\n'
                      'Positive: matched statistics lower CE',
               title=f'({chr(99+j)}) Initialization {init}: BN transfer at 2,048')
        ax.xaxis.label.set_fontsize(8)
        ax.grid(axis='x', alpha=.13)
        ax.legend(handles=[Line2D([], [], marker=m, mfc=f, mec='#222222', ls='', label=f'C{i}')
                           for i, m, f in [(1, 'o', 'white'), (2, 's', '#888888'), (3, '^', '#111111')]],
                  loc='upper center', bbox_to_anchor=(.53, -.27), ncol=3,
                  frameon=False, fontsize=7.5, columnspacing=1.1, handletextpad=.3)
    for ax in axes.flat:
        ax.set_axisbelow(True)
    finish(fig, 'revision3_robustness')


def main():
    summary = json.loads((DATA / 'summary.json').read_text())
    assert summary['status'] == 'passed' and summary['scoring_paths'] == 506
    assert summary['adaptation_seed_counts'] == {'A': 5, 'B': 3}
    pairs = pd.read_csv(DATA / 'contrasts_by_seed.csv')
    ints = pd.read_csv(DATA / 'contrast_intervals.csv')
    transfer = pd.read_csv(DATA / 'bn_transfer_penalty_intervals.csv')
    transfer_seeds = pd.read_csv(DATA / 'bn_transfer_penalty_by_seed.csv')
    for _, row in transfer[transfer.metric == 'CE_other_cal_minus_matched'].iterrows():
        q = transfer_seeds[(transfer_seeds.init_id == row.init_id) &
                           (transfer_seeds.state == row.state) &
                           (transfer_seeds.cal_seed == row.cal_seed) &
                           (transfer_seeds.infer_T == row.infer_T)]
        if row.state == 'adapted':
            q = q[q.adapt_T == row.adapt_T]
        vals = q.CE_other_cal_minus_matched.to_numpy()
        assert len(vals) == row['n']
        if len(vals) > 1:
            check_interval(vals, row)
        else:
            np.testing.assert_allclose(vals[0], row['mean'], rtol=1e-11, atol=1e-12)
            assert np.isnan(row.low) and np.isnan(row.high)
    target_effects(pairs, ints)
    bn_effects(pairs)
    robustness(pairs, ints, transfer)
    pd.DataFrame(PLOT_ROWS).to_csv(DATA / 'figure_plotted_intervals.csv', index=False)
    inputs = [DATA / f for f in ['summary.json', 'contrasts_by_seed.csv', 'contrast_intervals.csv',
                                'bn_transfer_penalty_intervals.csv', 'bn_transfer_penalty_by_seed.csv']]
    inputs += [OLD / f for f in ['paired_contrasts.csv', 'contrast_intervals.csv']]
    audit = dict(status='generated_and_numeric_checks_passed',
                 source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in inputs},
                 generated_figures=['revision3_target_effects', 'revision3_bn_effects', 'revision3_robustness'],
                 language_source='Retained R2.1 results, three paired adaptation seeds',
                 vision_source='R3, initialization A n=5 and initialization B n=3 analyzed separately',
                 calibration='Matched inference window; C1=9640, C2=9740, C3=9840, 2048 training images each',
                 uncertainty='Pointwise paired Student 95% intervals; fixed initialization and calibration subset',
                 no_extra_n_from_calibration=True, initial_state_uncertainty='Single checkpoint, no interval',
                 numerical_checks='Every plotted P and BN-transfer interval rederived from paired seed values and compared to source intervals',
                 budget_interpretation='1024 is a measured intermediate checkpoint; J still compares 512 and 2048',
                 no_synthetic_result_data=True)
    (DATA / 'figure_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
