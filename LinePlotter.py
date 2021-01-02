from typing import List

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class LinePlotter:
    def __init__(self, data: List[int or float], title="", x_title="", y_title=""):
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.data = data

        self.fig = None
        self.ax = None
        self.points = None

        self.y_max = 0
        self.y_min = 0

    def plot(self):
        """Update and display plot"""
        if self.fig is None:
            self._create_figure()

        self._update_figure()

    def _create_figure(self):
        self.fig, self.ax = plt.subplots(1, 1)
        plt.show(block=False)
        plt.draw()
        plt.title(self.title)
        plt.xlabel(self.x_title)
        plt.ylabel(self.y_title)
        self.points = self.ax.plot(np.arange(1, len(self.data) + 1, 1), self.data)[0]

    def _update_figure(self):
        self.points.set_data(np.arange(1, len(self.data) + 1, 1), self.data)
        self.ax.set_xlim(0, len(self.data) + 1)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.fig.canvas.draw()

    def update_data(self, new_data: List[int or float] or int or float):
        """Update data with passed data and update y min and max."""
        if hasattr(new_data, '__iter__'):
            min_value, max_value = self._find_min_max(new_data)
            self.data.extend(new_data)
        else:
            min_value = max_value = new_data
            self.data.append(new_data)

        self._update_y_min(min_value)
        self._update_y_max(max_value)

    def _find_min_max(self, new_data):
        min_value = max_value = new_data[0]
        for element in new_data[1:]:
            if element < min_value:
                min_value = element

            if element > max_value:
                max_value = max_value

        return min_value, max_value

    def _update_y_min(self, min):
        if self.y_min > min:
            self.y_min = min

    def _update_y_max(self, max):
        if self.y_max < max:
            self.y_max = max

