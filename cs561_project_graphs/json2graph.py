import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_data(data):
    epsilons = sorted([float(k) for k in data.keys()])
    metrics = ['throughput', 'latency', 'space_amplification']
    
    metric_data = {m: {e: [] for e in epsilons} for m in metrics}
    
    for epsilon, runs in data.items():
        e = float(epsilon)
        for run in runs:
            for metric in metrics:
                metric_data[metric][e].append(run['privacy_metrics']['performance_difference'][metric])
    
    return epsilons, metric_data

def create_box_plot(epsilons, metric_data, metric, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    values = [metric_data[metric][e] for e in epsilons]
    
    bp = ax.boxplot(values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    
    ax.set_xticks(range(1, len(epsilons) + 1))
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Relative Difference')
    ax.set_title(f'{metric.capitalize()} Relative Difference Across Epsilon Values')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(file_path):
    data = load_json_data(file_path)
    epsilons, metric_data = prepare_data(data)
    
    metrics = ['throughput', 'latency', 'space_amplification']
    for metric in metrics:
        output_file = f'{metric}_relative_difference.png'
        create_box_plot(epsilons, metric_data, metric, output_file)

if __name__ == '__main__':
    main('json2.json')