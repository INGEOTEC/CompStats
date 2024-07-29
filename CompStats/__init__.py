# Copyright 2024 Sergio Nava Muñoz and Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
__version__ = '0.0.9'
from CompStats.bootstrap import StatisticSamples
from CompStats.measurements import CI, SE, difference_p_value
from CompStats.performance import performance, difference, all_differences, plot_performance, plot_difference
from CompStats.performance import performance_multiple_metrics, difference_multiple, plot_performance_multiple, plot_difference_multiple
from CompStats.performance import all_differences_multiple, plot_performance2, plot_difference2, plot_scatter_matrix
