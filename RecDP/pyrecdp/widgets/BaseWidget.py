import ipywidgets as widgets
from IPython.display import display, clear_output

class BaseWidget:

    def __init__(self, display_flag = True):
        self.view = widgets.Output(layout={'border': '1px solid black'})
        if display_flag:
            display(self.view)

    def update(self, func, *args):
        with self.view:
            func(*args)

    def display(self, content):
        def display_func():
            if isinstance(content, str):
                print(content)
            else:
                display(content)
        self.update(display_func)

    def clear(self):
        self.view.clear_output()
    
    # def set_content(self, *embeddables, **kwargs):
    #     content_list = list(embeddables)
    #     self.outlet.set_content(widgets.VBox(content_list))

    # def set_title(self, title):
    #     self.outlet.set_title(title)

    # def hide(self):
    #     self.outlet.hide()

    # def is_displayed(self):
    #     return self.outlet.does_display(self)

    # def is_visible(self):
    #     return self.is_displayed() and self.outlet.is_visible()