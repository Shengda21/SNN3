"""Evidence-preserving redraw of the BN and native-endpoint figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL = {2: '#176B91', 4: '#C46A20'}


def redraw(data, finish):
    scores = pd.read_csv(data / 'quality_scores.csv')
    changes = pd.read_csv(data / 'bn_contrast_changes.csv')
    intervals = pd.read_csv(data / 'bn_contrast_change_intervals.csv')
    times = pd.read_csv(data / 'timing_process_medians.csv')
    ratios = pd.read_csv(data / 'timing_paired_ratios.csv')
    assert len(times) == 96 and len(ratios) == 24 and len(changes) == 10

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.0), layout='constrained',
                             gridspec_kw={'width_ratios': [1.15, 1, 1]})
    ax = axes[0]
    groups = [(512, 2), (512, 4), (2048, 2), (2048, 4)]
    for y, (budget, T) in zip([3, 2, 1, 0], groups):
        vals = changes[changes.updates == budget].sort_values('seed')[f'P{T}_bn_minus_raw']
        r = intervals[(intervals.updates == budget) & (intervals.contrast == f'P{T}')].iloc[0]
        ax.scatter(vals, y + np.linspace(-.13, .13, 5), s=13, color=COL[T], alpha=.55)
        ax.errorbar(r['mean'], y, xerr=[[r['mean']-r.ci95_low], [r.ci95_high-r['mean']]],
                    fmt='s', ms=4, color=COL[T], capsize=2.5, lw=1.2)
    ax.axvline(0, color='#777777', ls='--', lw=.7)
    ax.axhline(1.5, color='#DDDDDD', lw=.6)
    ax.set(yticks=[3, 2, 1, 0], yticklabels=['512: $P_2$', '512: $P_4$', '2,048: $P_2$', '2,048: $P_4$'],
           ylim=(-.5, 3.5), xlim=(-.077, .029), xticks=[-.06, -.03, 0],
           xlabel=r'$\Delta P_e=P_e^{\mathrm{BN}}-P_e$ (CE)', title='(a) Calibration effect')
    ax.grid(axis='x', alpha=.16)
    v = scores[(scores.task == 'vision') & (scores.state == 'adapted') &
               (scores.engine == 'eager') & (scores.infer_T == 4)]
    plane_rows = []
    for ax, budget, letter in zip(axes[1:], [512, 2048], ['b', 'c']):
        q = v[v.updates == budget].pivot(index=['seed', 'calibration'], columns='adapt_T', values=['ce', 'accuracy'])
        xs = q['ce'][2] - q['ce'][4]
        ys = 100 * (q['accuracy'][4] - q['accuracy'][2])
        ax.fill_between([-.05, 0], 0, 2.65, color='#EDF3EE', zorder=0)
        for seed in sorted(v.seed.unique()):
            x = [xs.loc[seed, cal] for cal in ['raw', 'bn_only']]
            y = [ys.loc[seed, cal] for cal in ['raw', 'bn_only']]
            ax.annotate('', xy=(x[1], y[1]), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle='->', color='#9CA5AA', lw=.65, shrinkA=3, shrinkB=3))
            for i, cal in enumerate(['raw', 'bn_only']):
                ax.plot(x[i], y[i], 'o', ms=3.7, mec='#4F7460' if i else '#666666',
                        mfc='#4F7460' if i else 'white', mew=.8)
                plane_rows.append(dict(updates=budget, seed=int(seed), calibration=cal,
                                       P4=float(x[i]), native_minus_reused_accuracy_pp=float(y[i])))
        for cal, fill in [('raw', 'white'), ('bn_only', '#23313C')]:
            ax.plot(xs.xs(cal, level=1).mean(), ys.xs(cal, level=1).mean(),
                    'D', ms=5.3, mec='#23313C', mfc=fill, mew=1, zorder=5)
        ax.axvline(0, color='#777777', ls='--', lw=.7)
        ax.axhline(0, color='#777777', ls='--', lw=.7)
        ax.text(-.047, 2.53, '$S_2$: lower loss\n$S_4$: higher accuracy', va='top', fontsize=8.5, color='#354B3F')
        ax.set(xlim=(-.05, .05), ylim=(-.23, 2.65), xticks=[-.04, 0, .04], yticks=[0, 1, 2],
               title=f'({letter}) {budget:,} updates', xlabel='$P_4$: reused minus native CE')
        ax.grid(alpha=.12)
    axes[1].set_ylabel('Native minus reused accuracy (pp)')
    axes[2].legend(handles=[Line2D([], [], marker='o', mfc='white', mec='#666666', ls='', label='Raw'),
                           Line2D([], [], marker='o', color='#4F7460', ls='', label='BN'),
                           Line2D([], [], marker='D', color='#23313C', ls='', label='Mean')],
                   frameon=False, loc='lower right', bbox_to_anchor=(1, .10), fontsize=8.5, handletextpad=.35)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(8.5)
    finish(fig, 'revision2_bn_effects')
    pd.DataFrame(plane_rows).to_csv(data / 'bn_metric_plane.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), layout='constrained',
                             gridspec_kw={'width_ratios': [1.23, 1]})
    timing_rows = []
    for row, (task, budget) in enumerate([('language', 4096), ('vision', 2048)]):
        ax, rel = axes[row]
        native = scores[(scores.task == task) & (scores.state == 'adapted') &
                        (scores.updates == budget) & (scores.calibration == 'raw') &
                        (scores.adapt_T == scores.infer_T)]
        seeds = sorted(native.seed.unique())
        for seed in seeds:
            for T in [2, 4]:
                g = times[(times.task == task) & (times.seed == seed) & (times['T'] == T)]
                quality = native[(native.seed == seed) & (native.infer_T == T)].set_index('engine')
                for session in [0, 1, 2]:
                    p = g[g.session == session].set_index('engine')
                    ax.plot([p.loc[e, 'median_ms'] for e in ['graph', 'eager']],
                            [quality.loc[e, 'ce'] for e in ['graph', 'eager']],
                            color=COL[T], alpha=.2, lw=.6, zorder=1)
                for engine, filled in [('eager', False), ('graph', True)]:
                    a = g[g.engine == engine]
                    ce = quality.loc[engine, 'ce']
                    ax.scatter(a.median_ms, [ce]*3, marker='o' if T == 2 else '^', s=15,
                               edgecolor=COL[T], facecolor=COL[T] if filled else 'white', lw=.7, zorder=3)
                    ax.scatter([a.median_ms.median()], [ce], marker='o' if T == 2 else '^', s=39,
                               edgecolor=COL[T], facecolor=COL[T] if filled else 'white', lw=1, zorder=4)
                    for r in a.itertuples():
                        timing_rows.append(dict(task=task, seed=int(seed), T=T, engine=engine,
                                                session=int(r.session), median_ms=r.median_ms, ce=ce))
        ax.set(title=f'({chr(97+2*row)}) {task.capitalize()}: quality and latency',
               xlabel='Wall latency (ms)', ylabel='Test cross-entropy')
        ax.grid(alpha=.15)
        legend = [Line2D([], [], marker='o', color=COL[2], ls='', label='$T=2$'),
                  Line2D([], [], marker='^', color=COL[4], ls='', label='$T=4$'),
                  Line2D([], [], marker='o', color='#555555', mfc='white', ls='', label='Eager'),
                  Line2D([], [], marker='o', color='#555555', ls='', label='Graph')]
        ax.legend(handles=legend, frameon=False, ncol=2, fontsize=8, loc='center right',
                  columnspacing=.65, handletextpad=.35)
        sub = ratios[ratios.task == task]
        for j, seed in enumerate(seeds):
            for k, r in enumerate(sub[sub.seed == seed].sort_values('session').itertuples()):
                y = j + [-.18, 0, .18][k]
                rel.plot([r.eager_T2_saving_percent, r.graph_T2_saving_percent], [y, y],
                         color='#A7AFB4', lw=.7, zorder=1)
                rel.plot(r.eager_T2_saving_percent, y, 'o', mfc='white', mec=COL[2], ms=4, mew=.8)
                rel.plot(r.graph_T2_saving_percent, y, 'o', color=COL[2], ms=4)
        rel.set(yticks=list(range(len(seeds))), yticklabels=[str(int(s)) for s in seeds],
                ylim=(-.5, len(seeds)-.5), xlim=(18.5, 39), xticks=[20, 25, 30, 35],
                xlabel='Paired $T=2$ latency saving (%)', ylabel='Training seed',
                title=f'({chr(98+2*row)}) Same-engine savings')
        rel.grid(axis='x', alpha=.15)
        for ax in [ax, rel]:
            ax.set_axisbelow(True)
            ax.title.set_fontsize(9)
    finish(fig, 'revision2_quality_latency')
    assert len(timing_rows) == 96 and len(plane_rows) == 20
    pd.DataFrame(timing_rows).to_csv(data / 'timing_quality_process_points.csv', index=False)
    return {'BN_metric_points': 20, 'timing_process_points': 96, 'paired_process_sessions': 24,
            'accuracy_contrasts': 'descriptive; native S4 minus reused S2 at inference T4',
            'timing_repetitions': 'three processes per seed; no independent-training interpretation'}
