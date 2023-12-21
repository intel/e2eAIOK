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

from collections import defaultdict
                
class Graph:
    # Constructor
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = []
        self.src_list = []
        self.dst_list = []
 
    def addEdge(self,u,v):
        self.graph[v].append(u)
        if u not in self.nodes:
            self.nodes.append(u)
        if v not in self.nodes:
            self.nodes.append(v)
        if v not in self.src_list:
            self.src_list.append(v)
        if u not in self.dst_list:
            self.dst_list.append(u)

    def find_last_node(self):
        last_node_candidates = []
        for n in self.nodes:
            if n in self.src_list and n not in self.dst_list:
                last_node_candidates.append(n)
        if len(last_node_candidates) == 1:
            return last_node_candidates[-1]
        elif len(last_node_candidates) == 0:
            raise ValueError(f"Found no leaf node in pipeline DAG")
        else:
            raise NotImplementedError(f"Found more than one node as leaf {last_node_candidates}, not supported yet")
        
    def bfs(self, visited, node, queue, ret): #function for BFS
        visited.add(node)
        queue.append(node)

        while queue:
            m = queue.pop(0)
            ret.append(m)

            for neighbour in self.graph[m]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return ret

    def dfs(self, visited, node, ret):  #function for dfs 
        if node not in visited:
            ret.append(node)
            visited.add(node)
            for neighbour in self.graph[node]:
                ret = self.dfs(visited, neighbour, ret)
        return ret

    def chain(self, method = 'dfs'):
        last_node = self.find_last_node()
        ret = []
        if method == 'bfs':
            # Go BFS
            queue = []
            visited = set()
            ret = self.bfs(visited, last_node, queue, ret)
        if method == 'dfs':
            # Go DFS
            visited = set()
            ret = self.dfs(visited, last_node, ret)
        ret.reverse()
        return ret
        

class DiGraph(dict):
    def get_max_idx(self):
        id_list = list(self.keys())
        return max(id_list) if len(id_list) > 0 else -1
    
    def convert_to_node_chain(self):
        graph = Graph()
        edges = []

        for node_id, config in self.items():
            if config.children:
                for src_id in config.children:
                    edges.append([src_id, node_id])
        
        if len(edges):
            for edge in edges:
                graph.addEdge(*edge)
                
            ret = graph.chain()
        else:
            ret = list(self.keys())
        return ret

    def json_dump(self, base_dir = ""):
        import json
        to_dump = dict((node_id, op.dump(base_dir)) for node_id, op in self.items())
        return json.dumps(to_dump, indent=4)

    def copy(self):
        import copy
        return copy.copy(self)
    