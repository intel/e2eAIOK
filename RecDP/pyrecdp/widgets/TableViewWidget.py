from pyrecdp.widgets import BaseWidget

class TableViewWidget(BaseWidget):

    def __init__(self, *args, parent_tabs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_tabs = parent_tabs
        self._df_is_outdated = False

    def df_did_change(self):
        """
        Call this method when the DataFrame did change
        """
        self._df_is_outdated = True

    def tab_got_selected(self):
        """
        Call this method when the tab of this TabViewable got selected
        """
        if self._df_is_outdated:
            self._df_is_outdated = False
            self.render()

    def render_in(self, outlet):
        if isinstance(outlet, Window):
            super().render_in(outlet)
        else:
            # the outlet is a TabSection
            # this is a hotfix to enable the same API for TabViewables e.g.
            # `TabViewable().render_in(tab_section)`
            outlet.add_tab(self)

    def set_content(self, *embeddables, **kwargs):
        # Attention: this method was overriden and copied from Viewable
        # because we need to add a class to the wrapping widgets.VBox
        # This ensures that TabViewables have a min-height
        content_list = list(embeddables)
        box = widgets.VBox(content_list)
        box.add_class("bamboolib-min-height-for-tab-viewables")
        self.outlet.set_content(box)

    def _ipython_display_(self, *args, **kwargs):
        """
        This method is called by Jupyter when trying to display the object.
        We are setting up all the surrounding classes that are needed for the experience
        of displaying a TabViewable within a properly setup TabSection
        """
        from bamboolib.wrangler import (
            Wrangler,
        )  # inline import to prevent circular import

        df_manager = self.df_manager
        tab_section = TabSection(df_manager)
        tab_section.add_tab(
            Wrangler(df_manager=df_manager, parent_tabs=tab_section), closable=False
        )
        tab_section.add_tab(self, closable=False)
        display(tab_section)