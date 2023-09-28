from pyrecdp.core import DiGraph
from pyrecdp.primitives.operations import Operation
import logging
from pyrecdp.core.utils import Timer, sample_read, deepcopy
from IPython.display import display
import json, yaml
import graphviz

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class BasePipeline:
    def __init__(self):
        self.rdp = None
        self.pipeline = DiGraph()
        self.transformed_cache = None

    def __repr__(self):
        return repr(self.pipeline)

    def export(self, file_path = None):
        json_object = self.pipeline.json_dump()
        if file_path:
            # Writing to sample.json
            with open(file_path, "w") as outfile:
                outfile.write(json_object)
        else:
            return json_object
    
    def import_from_json(self, file_path):
        try:
            json_object = json.loads(file_path)
            flag = 'loaded'
        except ValueError as e:
            flag = 'try_load_as_file'
        if flag == 'try_load_as_file':
            with open(file_path, "r") as infile:
                json_object = json.load(infile)
                flag = 'loaded'
        if flag != 'loaded':
            raise ValueError("Unable to load as json string or json file")
        for idx, op_config in json_object.items():
            idx = int(idx)
            if idx in self.pipeline:
                continue
            self.pipeline[idx] = Operation.load(idx, op_config)
        
    def import_from_yaml(self, file_path):
        with open(file_path, "r") as infile:
            yaml_object = yaml.safe_load(file)
        
        for idx, op_config in json_object.items():
            idx = int(idx)
            if idx in self.pipeline:
                continue
            self.pipeline[idx] = Operation.load(idx, op_config)
        
    def create_executable_pipeline(self):
        node_chain = self.pipeline.convert_to_node_chain()
        executable_pipeline = DiGraph()
        executable_sequence = []
        for idx in node_chain:
            actual_op = self.pipeline[idx].instantiate()
            if actual_op:
                executable_pipeline[idx] = actual_op
                executable_sequence.append(executable_pipeline[idx])
        return executable_pipeline, executable_sequence
    
    def add_operation(self, config):
        pass

    def delete_operation(self, id):
        cur_idx = id
        pipeline_chain = self.to_chain()
        children = self.pipeline[cur_idx].children    
        # we need to find nexts
        for to_replace_child in children:
            next = []
            for idx in pipeline_chain:
                if self.pipeline[idx].children and to_replace_child in self.pipeline[idx].children:
                    next.append(idx)
            if len(next) == 1:
                self.pipeline[next[0]].children = children
            else:            
                for idx in next:
                    # replace next's children with new added operator
                    children_in_next = self.pipeline[idx].children
                    found = {}
                    for id, child in enumerate(children_in_next):
                        if child == cur_idx:
                            found[id] = to_replace_child
                    for k, v in found.items():
                        self.pipeline[idx].children[k] = v
        if hasattr(self, 'transformed_end_idx'):
            self.transformed_end_idx = children[0]
        del self.pipeline[cur_idx]
          
    def plot(self):
        f = graphviz.Digraph(format='svg')
        edges = []
        nodes = []
        f.attr(fontsize='10')
        def add_escape(input):
            input = input.replace('<', '\<').replace('>', '\>')
            #input = input.replace("'", "\\\'").replace("\"", "\\\"")
            return input

        def add_break(input):
            if isinstance(input, list) and len(input) < 3:
                for line in input:
                    if isinstance(line, str):
                        ret = str(input)
                        return ret
            if isinstance(input, dict):
                input = [f"{k}: {add_break(v)}" for k, v in input.items()]
            if isinstance(input, list):
                try_str = str(input)
                if len(try_str) < 200:
                    return try_str
                ret = "" + "\l"
                for line in input:
                    ret += str(add_break(line)) + "\l"
                return ret
            return input

        for node_id, config in self.pipeline.items():
            nodes.append([str(node_id), f"{node_id}:{config.op} |{add_escape(str(add_break(config.config)))}"])
            if config.children:
                for src_id in config.children:
                    edges.append([str(src_id), str(node_id)])
        for node in nodes:
            f.node(node[0], node[1], shape='record', fontsize='12')
        for edge in edges:
            f.edge(*edge)
        try:
            f.render(filename='pipeline', view = False)
        except:
            pass
        return f

    def to_chain(self):
        return self.pipeline.convert_to_node_chain()

    def get_transformed_cache(self):
        if hasattr(self, 'transformed_cache') and self.transformed_cache is not None:
            return self.transformed_cache
        else:
            print("No transformed data detected.")