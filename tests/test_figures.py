import matplotlib
matplotlib.use('Agg')
import pandas as pd
from maize_benchmark.reporting import report


def test_scenario_order_and_contrasting_error_bars(tmp_path):
    assert report.SCENARIO_ORDER == [
        'extreme_dry', 'dry', 'normal', 'limited_water', 'wet'
    ]
    assert report.ERRORBAR_COLOR not in report.BAR_COLORS.values()
    assert report.ERRORBAR_COLOR not in ['blue', '#0072B2']
    rows = []
    for a, algorithm in enumerate(['ppo','crossq','dsac_t','dreamerv3']):
        for j, scenario in enumerate(report.SCENARIO_ORDER):
            mean = 3 + a*0.7 + j*0.3
            rows.append({'algorithm':algorithm,'scenario':scenario,
                         'metric':'dry_yield_tonne_per_ha', 'mean':mean,
                         'ci_lower':mean-0.2, 'ci_upper':mean+0.3,
                         'n_independent_seeds':3})
    report._plot_metric(pd.DataFrame(rows), 'dry_yield_tonne_per_ha',
                        ['ppo','crossq','dsac_t','dreamerv3'], tmp_path)
    assert (tmp_path/'mean_dry_yield_by_scenario.png').stat().st_size > 1_000
    svg = (tmp_path/'mean_dry_yield_by_scenario.svg').read_text()
    assert 'Extreme Dry' in svg and 'Limited Water' in svg
    assert '3.00' in svg
    assert 'Mean dry yield (tonne/ha)' in svg
