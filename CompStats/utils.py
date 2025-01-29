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
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False


def progress_bar(arg, use_tqdm: bool=True, **kwargs):
    """Progress bar using tqdm"""
    if not USE_TQDM or not use_tqdm:
        return arg
    return tqdm(arg, **kwargs)
