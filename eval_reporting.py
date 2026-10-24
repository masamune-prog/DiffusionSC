"""Shared DiffusionSC-style summaries for DeepSC and diffusion evaluations."""

import csv
import math

import numpy as np


def compute_ci95(samples):
    """Return the sample mean and two-sided 95% confidence interval margin."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    try:
        from scipy.stats import t
        critical_value = float(t.ppf(0.975, df=values.size - 1))
    except ImportError:
        critical_value = 1.96
    return mean, critical_value * standard_error


def report_results(snr_values, runs, predictions, *, channel, checkpoint,
                   results_path, predictions_path, extra_fields=None):
    """Print and save per-SNR means and CIs, plus seed-tagged predictions.

    ``runs`` maps metric names to arrays shaped [number of seeds, number of SNRs].
    """
    order = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'bert_p', 'bert_r', 'bert_f1',
             'latency_ms', 'throughput_sent_per_sec']
    metric_keys = [key for key in order if key in runs and key not in
                   ('latency_ms', 'throughput_sent_per_sec')]
    n_seeds = np.asarray(runs['latency_ms']).shape[0]
    summary = {}
    for key, values in runs.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n_seeds, len(snr_values)):
            raise ValueError(f'{key} has shape {values.shape}; expected {(n_seeds, len(snr_values))}')
        pairs = [compute_ci95(values[:, i]) for i in range(len(snr_values))]
        summary[key] = {'mean': np.array([p[0] for p in pairs]),
                        'ci95': np.array([p[1] for p in pairs])}

    width = 8 + len(metric_keys) * 16 + 28
    print('\n' + '=' * width)
    print(f' Results (Mean ± 95% CI over {n_seeds} seed(s))  |  '
          f'channel={channel}  |  ckpt={checkpoint}')
    print('=' * width)
    header = f'{"SNR":>6}' + ''.join(f'  {key.upper():>14}' for key in metric_keys)
    print(header + f'  {"Lat(ms)":>12}  {"Tput(s/s)":>12}')
    print('-' * width)

    def format_row(label, index=None):
        row = f'{label:>6}'
        for key in metric_keys:
            mean = summary[key]['mean'][index] if index is not None else summary[key]['mean'].mean()
            ci = summary[key]['ci95'][index] if index is not None else summary[key]['ci95'].mean()
            row += f'  {mean:6.4f}±{ci:5.4f}' if n_seeds > 1 else f'  {mean:14.4f}'
        for key, mean_fmt, ci_fmt in [('latency_ms', '7.2f', '4.2f'),
                                      ('throughput_sent_per_sec', '7.1f', '4.1f')]:
            mean = summary[key]['mean'][index] if index is not None else summary[key]['mean'].mean()
            ci = summary[key]['ci95'][index] if index is not None else summary[key]['ci95'].mean()
            row += f'  {format(mean, mean_fmt)}±{format(ci, ci_fmt)}'
        return row

    for index, snr in enumerate(snr_values):
        print(format_row(snr, index))
    print('-' * width)
    print(format_row('Average'))
    print('=' * width)

    extra_fields = extra_fields or {}
    fields = ['channel', 'checkpoint', 'snr_db', 'num_seeds', *extra_fields]
    for key in metric_keys:
        fields += [f'{key}_mean', f'{key}_ci95']
    fields += ['latency_ms_mean', 'latency_ms_ci95',
               'throughput_sent_per_sec_mean', 'throughput_sent_per_sec_ci95']
    with open(results_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, snr in enumerate(snr_values):
            row = {'channel': channel, 'checkpoint': checkpoint, 'snr_db': snr,
                   'num_seeds': n_seeds, **extra_fields}
            for key in [*metric_keys, 'latency_ms', 'throughput_sent_per_sec']:
                row[f'{key}_mean'] = round(float(summary[key]['mean'][index]), 6)
                row[f'{key}_ci95'] = round(float(summary[key]['ci95'][index]), 6)
            writer.writerow(row)
    print(f'\nAggregate results -> {results_path}')

    with open(predictions_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file,
                                fieldnames=['seed', 'snr_db', 'sample_idx',
                                            'predicted', 'reference'])
        writer.writeheader()
        writer.writerows(predictions)
    print(f'Per-sentence predictions -> {predictions_path}')

    return summary
