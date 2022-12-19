from src.measures import MeshMeasurements


def log_scores(scores_dict, output_set):
    for score_key in scores_dict:
        if output_set == 'all':
            for idx in range(scores_dict[score_key].shape[0]):
                measure_name = MeshMeasurements._NAMES[idx]
                measure_label = MeshMeasurements._LABELS[idx]
                score = scores_dict[score_key][idx] * 1000.
                
                print(f'{score_key} ({measure_name}={measure_label}): {score}mm')
        else:
            score = scores_dict[score_key][0] * 1000.
            unit = 'm3' if output_set == 'volume' else 'mm'

            print(f'{score_key} ({output_set}): {score}{unit}')
