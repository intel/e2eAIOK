"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import ipywidgets as widgets
from IPython.display import display

class TabWidget:
    
    def __init__(self, children = None, display_flag = True):
        self.view = widgets.Tab()
        self.children = []
        if children:
            [self.append(c[0], c[1]) for c in children]
        if display_flag:
            display(self.view)

    def append(self, tab_name, tab_widget):
        if not self.get_tab_from_children(tab_name):
            idx = len(self.children)
            self.children.append((tab_name, tab_widget))
            self.view.children = [c[1].view for c in self.children]
            self.view.set_title(idx, tab_name)
            
    def get_tab_from_children(self, tab_name):
        for n in self.children:
            if tab_name == n[0]:
                return n
        return None
