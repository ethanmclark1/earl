import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(problem_instances, all_metrics):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics = list(all_metrics[0].keys())
    approaches = ['A* w/ EARL', 'A* w/ RL']
    num_approaches = len(approaches)    
    
    approach_names = ['Our Approach', 'A* w/ RL']
    metric_names = ['Average Path Cost', 'Losses', 'Reward']

    for metric, name in zip(metrics, metric_names):
        for idx, problem_instance in enumerate(problem_instances):
            problem_dir = f'{results_dir}/{problem_instance}'
            if not os.path.exists(problem_dir):
                os.makedirs(problem_dir)
            
            metrics = all_metrics[idx][metric].copy()
            earl = metrics[approaches[0]]
            rl = metrics[approaches[1]]
            
            x_values = np.arange(num_approaches)
            if metric == 'avg_path_cost':
                plt.bar(x_values[0], earl, width=0.2, label='Our Approach')
                plt.bar(x_values[1], rl, width=0.2, label='A* w/ RL')
                plt.xticks(x_values, labels=approach_names)
                plt.ylim(0, 10)
            else:
                epochs = np.arange(len(earl))
                plt.plot(epochs, earl, label='Our Approach')
                plt.plot(epochs, rl, label='A* w/ RL')
                plt.xlabel('Epochs')
                
            plt.title(f'{problem_instance} {name}')
            plt.ylabel(name)
            plt.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{problem_dir}/{metric}.png')
            plt.clf()