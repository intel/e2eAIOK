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
        return max(list(self.keys()))
    
    def convert_to_node_chain(self):
        graph = Graph()
        edges = []

        for node_id, config in self.items():
            if config.children:
                for src_id in config.children:
                    edges.append([src_id, node_id])
        for edge in edges:
            graph.addEdge(*edge)
            
        ret = graph.chain()
        return ret

    def json_dump(self):
        import json
        to_dump = dict((node_id, op.dump()) for node_id, op in self.items())
        return json.dumps(to_dump, indent=4)
        
    