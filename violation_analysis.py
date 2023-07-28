import numpy as np
import matplotlib.pyplot as plt

def extract_violation_codes(violation):
    # Function to extract violation codes from a textual list of violations
    violations_list = list(map(lambda v: v.strip(), violation.split('|')))
    violation_dots = [violation.find('.') for violation in violations_list]
    violation_codes = list(set([int(v[:idx]) for v, idx in zip(violations_list, violation_dots) if idx != -1]))
    return violation_codes

def merge_violation_codes(violation_series):
    # Function to merge all violation codes from the column into one flat array
    all_codes = [code for inspection_violation_codes in violation_series.values for code in inspection_violation_codes]
    return all_codes

def violation_counts(violations, max_violation_code):
    # Function to create the histogram for violation codes
    counts, code_bins = np.histogram(violations, bins=np.arange(1, max_violation_code + 2))
    return counts, code_bins

def violations_distribution(df, violation_column='Violation Codes', max_violation_code=70):
    # Function to create the violation codes distribution from the dataframe
    all_codes = merge_violation_codes(df[violation_column])
    counts, code_bins = violation_counts(all_codes, max_violation_code)
    return code_bins[:-1], counts

def plot_violations_stacked_bars(data, title, violation_column='violation Codes', max_violation_code=70, xticks=None):
    # Function to plot stacked bars for violation codes distribution
    plt.figure(figsize=(12, 7))
    plt.title(title)

    results = data['results'].unique()
    bars = []
    total_counts = len(data)
    previous_counts = np.zeros((max_violation_code,))

    for result in results:
        bins, counts = violations_distribution(data[data['results'] == result], violation_column, max_violation_code)
        percentages = counts / total_counts * 100
        bar = plt.bar(bins, percentages, bottom=previous_counts)
        bars.append(bar[0])
        previous_counts += percentages

    if xticks is not None:
        plt.xticks(np.arange(1, max_violation_code + 1), xticks, rotation=40)
    else:
        plt.xlabel('Violation code')

    plt.ylabel('% of inspections with violations')
    plt.ylim((0, 50))

    plt.legend(tuple(bars), tuple(results))
    plt.grid(True, axis='y')

    plt.show()
