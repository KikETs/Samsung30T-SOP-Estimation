"""Shared explicit semantic mapping for the SOP manuscript figures."""
STYLE = {
    'online_scale_adaptation': '#4477AA',
    'fixed_scale': '#EE6677',
    'exceedance': '#AA3377',
    'reference': '#BBBBBB',
    # Author-requested contrast override for nearly coincident measured trials.
    # Reference bands keep the paper-wide pale grey.
    'search_trial_colors': {1: '#555555', 2: '#666666', 3: '#222222', 4: '#222222', 5: '#4477AA'},
    'text': '#222222',
    'axes': '#333333',
    'background': '#FFFFFF',
    'font': 'Arial',
    'width_mm': 190,
    'height_mm': 125,
    'font_pt': 8,
    'label_pt': 10,
    'png_dpi': 600,
    'temperature_strip': {'negative': '#4477AA', 'zero': '#FFFFFF',
                          'positive': '#EE6677', 'limits_C': [-40, 40]},
    'condition_hatches': {'charge': '//////', 'discharge': '', 'short_pulse': 'xxx'},
    'comparators': {
        'Previous pulse power': {'color': '#228833', 'marker': 'o'},
        'Previous pulse resistance': {'color': '#66CCEE', 'marker': 's'},
        'Equivalent circuit, one RC branch': {'color': '#CCBB44', 'marker': '^'},
        'Equivalent circuit, two RC branches': {'color': '#222222', 'marker': 'D'},
    },
}
