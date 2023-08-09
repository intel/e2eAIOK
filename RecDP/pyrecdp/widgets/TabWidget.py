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
