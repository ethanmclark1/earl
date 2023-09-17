import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(problem_instances, metrics, num_episodes):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics = list(all_metrics[0].keys())
    approaches = ['rl', 'voronoi_map', 'grid_world', 'direct_path']
    num_approaches = len(approaches)

    approach_names = ['Our Approach', 'Voronoi Map', 'Grid World', 'Direct Path']
    metric_names = ['Language Safety', 'Ground Agent Success', 'Average Direction Length']
    
    for metric, name in zip(metrics, metric_names):
        for idx, problem_instance in enumerate(problem_instances):
            problem_dir = f'{results_dir}/{problem_instance}'
            if not os.path.exists(problem_dir):
                os.makedirs(problem_dir)
                
            metrics = all_metrics[idx][metric].copy()

            if metric == 'language_safety':
                rl = (metrics[approaches[0]] / num_episodes) * 100
                voronoi_map = (metrics[approaches[1]] / num_episodes) * 100
                grid_world = (metrics[approaches[2]] / num_episodes) * 100
                direct_path = (metrics[approaches[3]] / num_episodes) * 100
            elif metric == 'ground_agent_success':
                language_safety_metrics = all_metrics[idx]['language_safety']
                for approach in approaches:
                    if metrics[approach] != 0:
                        metrics[approach] = metrics[approach] / language_safety_metrics[approach] * 100
                    else:
                        metrics[approach] = 0
                rl = metrics[approaches[0]]
                voronoi_map = metrics[approaches[1]]
                grid_world = metrics[approaches[2]]
                direct_path = metrics[approaches[3]]
            else:
                rl = metrics[approaches[0]]
                voronoi_map = metrics[approaches[1]]
                grid_world = metrics[approaches[2]]
                direct_path = metrics[approaches[3]]
                
            x_values = np.arange(num_approaches)

            plt.bar(x_values[0], rl, width=0.2, label='Our Approach')
            plt.bar(x_values[1], voronoi_map, width=0.2, label='Voronoi Map')
            plt.bar(x_values[2], grid_world, width=0.2, label='Grid World')
            plt.bar(x_values[3], direct_path, width=0.2, label='Direct Path')
            plt.xlabel(problem_instance.capitalize())
            plt.xticks(x_values, labels=approach_names)
            plt.ylim(0, 20 if metric == 'avg_direction_len' else 100)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{results_dir}/{problem_instance}/{metric}.png')
            plt.clf()
        
        for i in range(len(all_metrics)):
            print(f'{problem_instances[i]} {metric}: {all_metrics[i][metric]}')