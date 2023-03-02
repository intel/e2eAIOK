from .base import BaseFeatureGenerator as super_class
import pandas as pd
from pyrecdp.core import SeriesSchema, DataFrameSchema
from pyrecdp.primitives.operations import Operation
from typing import List
from pyrecdp.core.utils import Timer, update_linklist

class RelationalFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = {}
        self.pipeline_start_idx = 0
        self.pipeline_main_idx = 0
        
    def fit_prepare(self, pipeline, children, max_idx):
        self.pipeline_start_idx = max_idx
        self.pipeline_main_idx = pipeline[children[0]].idx
        self.others = dict((i, pipeline[i].output) for i in children[1:])
        self.candidates = self.others.copy()
        
        pa_schema = pipeline[children[0]].output
        stop = False
        while not stop:
            related_tables = self.find_tables_share_same_feature(pa_schema)
            if len(related_tables) > 0:
                pa_schema = self.dry_run_merge_tables(pa_schema, related_tables)
            else:
                stop = True
            if len(self.candidates) == 0:
                stop = True
        pipeline.update(self.pipeline)
        return pipeline, self.pipeline_main_idx, self.pipeline_start_idx
    
    def get_function_pd(self):
        def merge_tables(df):                
            return df
        return merge_tables

    def get_function_spark(self, rdp):        
        raise NotImplementedError("RelationalFeatureGenerator spark implementation is WIP")
    
    #======== Private utility funcs ==========#
    def dry_run_merge_tables(self, target_table_schema, related_tables):            
        def _dry_merge(child_idx, on, tgt_schema, src_schema):
            tgt_col_names = [col.name for col in tgt_schema]
            src_col_names = [col.name for col in src_schema]
            error = False
            # check on is containing in both tgt and src
            for n in on:
                if n not in tgt_col_names or n not in src_col_names:
                    error = True
                    break
            if error:
                return tgt_schema
            # remove primary keys in src
            src_schema = [n for n in src_schema if n.name not in on]
            # fix conflicted col_names
            rename_cols = dict()
            for idx, n in enumerate(src_schema):
                if n.name in tgt_col_names:
                    # found conflicted name
                    fix = 1
                    still_conflict = True
                    while still_conflict:
                        cur_name = f"{n.name}_{fix}"
                        if cur_name not in tgt_col_names:
                            still_conflict = False
                        else:
                            fix += 1
                    rename_cols[n.name] = cur_name
                    src_schema[idx].name = cur_name
            # update pipeline
            left_idx = self.pipeline_main_idx
            if len(rename_cols) > 0:
                self.pipeline_start_idx += 1
                cur_idx = self.pipeline_start_idx
                self.pipeline[cur_idx] = Operation(cur_idx, [child_idx], src_schema, 'rename', rename_cols)
                right_idx = self.pipeline_start_idx
            else:
                right_idx = child_idx
            # merge to tgt
            [tgt_schema.append(n) for n in src_schema]
            
            # add merge to pipeline
            self.pipeline_start_idx += 1
            cur_idx = self.pipeline_start_idx
            self.pipeline[cur_idx] = Operation(cur_idx, [left_idx, right_idx], tgt_schema, 'merge', {'on': on, 'how': 'left'})
            self.pipeline_main_idx = cur_idx
            return tgt_schema
                
        for t_name, t_primary_keys in related_tables.items():
            target_table_schema = _dry_merge(t_name, t_primary_keys, target_table_schema, self.others[t_name])
        return target_table_schema

    def find_tables_share_same_feature(self, target_table_schema):
        related_tables = {}
        to_remove_candidates = []
        target_table_col_names = [s.name for s in target_table_schema] 
        for t, df_schema in self.candidates.items():
            df_col_names = [s.name for s in df_schema]            
            for key in df_col_names:
                if key in target_table_col_names:
                    update_linklist(related_tables, t, key)
                    to_remove_candidates.append(t)
                    
        # update candidates
        for t in to_remove_candidates:
            if t in self.candidates:
                del self.candidates[t] 
        return related_tables