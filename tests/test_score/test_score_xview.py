import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from playground_metrics import MeanAveragePrecisionMetric
from tests.resources.xview_scoring.score import score_from_files


class LabelReader:
    def __init__(self, labelfile_path):
        """
        Reads the xView annotation geojson and provides an iterator over the tiles which yields the tile filename and
        a list of every feature in the tile
        :param labelfile_path: The path to the xView geojson label files
        """
        self.labelfilename = labelfile_path
        try:
            with open(self.labelfilename, 'r') as f:
                self.labels = json.load(f)
        except OSError as e:
            raise OSError('The path provided is incorrect : {0}'.format(e))
        self.tiles = set()
        self.tiles_to_record = defaultdict(list)
        for i, feature in enumerate(tqdm(self.features, desc='Parsing geojson')):
            self.tiles.add(feature['properties']['image_id'])
            self.tiles_to_record[feature['properties']['image_id']].append(i)
        self.tiles = list(self.tiles)
        self.current = 0

    def __getitem__(self, item):
        return [self.features[i] for i in self.tiles_to_record[item]]

    def __len__(self):
        return len(self.tiles)

    @property
    def features(self):
        return self.labels['features']


def parse_prediction_directory(threshold, path, annotation_path):
    label_reader = LabelReader(annotation_path)

    tile_list = os.listdir(path)
    map_metric_calculator = MeanAveragePrecisionMetric(threshold, 'xview')

    with tqdm(total=len(tile_list)) as progress:
        for tile in tile_list:
            progress.desc = tile
            progress.refresh()
            predictions = []
            with open(os.path.join(path, tile), 'r') as f:
                for line in f:
                    x_min, y_min, x_max, y_max, label, confidence = line.rstrip('\n').split(' ')[:-1]
                    if float(confidence) <= 0.0:
                        print('{0}: {1} {2} {3} {4} confidence is {5}'.format(tile, x_min, y_min, x_max, y_max,
                                                                              confidence))
                    predictions.append([int(x_min), int(y_min), int(x_max), int(y_max), float(confidence), int(label)])
            predictions = np.array(predictions)
            tile_labels = [label['properties']['bounds_imcoords'].split(',') + [label['properties']['type_id']]
                           for label in label_reader[tile[:-4]]]
            tile_labels = np.int_(np.array(tile_labels))
            map_metric_calculator.update(predictions, tile_labels)
            progress.update()

    map_metric_calculator.compute()
    return map_metric_calculator
