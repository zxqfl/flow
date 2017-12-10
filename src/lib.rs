#[link(name="flow")]
extern {
    fn network_simplex_mcmf_i64(num_vertices: i64, num_edges: i64,
        node_supply: *const i64, edge_a: *const i64, edge_b: *const i64,
        edge_capacity: *const i64, edge_cost: *const i64, edge_flow_result: *mut i64) -> i64;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Node(pub usize);

pub struct Edge {
    pub a: Node,
    pub b: Node,
    pub data: EdgeData,
}

pub struct Graph {
    nodes: Vec<NodeData>,
    edges: Vec<Edge>,
}

impl Graph {
    pub fn new(nodes: Vec<NodeData>) -> Self {
        Graph {nodes, edges: Vec::new()}
    }
    pub fn add_edge(&mut self, a: Node, b: Node, data: EdgeData) -> &mut Self {
        assert!(a.0 < self.nodes.len());
        assert!(b.0 < self.nodes.len());
        self.edges.push(Edge {a, b, data});
        self
    }
    pub fn extract(self) -> (Vec<NodeData>, Vec<Edge>) {
        (self.nodes, self.edges)
    }
}

impl Graph {
    pub fn new_default(num_vertices: usize) -> Self {
        let nodes = vec![Default::default(); num_vertices];
        Graph {nodes, edges: Vec::new()}
    }
}

#[derive(Clone, Copy, Default)]
pub struct NodeData {
    supply: i64,
}

#[derive(Clone, Copy)]
pub struct EdgeData {
    pub cost: i64,
    pub capacity: i64,
    pub flow: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cost(pub i64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capacity(pub i64);

impl EdgeData {
    pub fn new(cost: Cost, capacity: Capacity) -> Self {
        let cost = cost.0;
        let capacity = capacity.0;
        EdgeData {cost, capacity, flow: Default::default()}
    }
}

impl Graph {
    pub fn increase_supply(&mut self, node: Node, amount: i64) {
        self.delta_supply(node, amount);
    }
    pub fn decrease_supply(&mut self, node: Node, amount: i64) {
        self.delta_supply(node, -amount);
    }
    pub fn delta_supply(&mut self, node: Node, amount: i64) {
        self.nodes[node.0].supply += amount;
    }

    pub fn mcmf(&mut self) -> i64 {
        let num_vertices = self.nodes.len() as i64;
        let num_edges = self.edges.len() as i64;
        let node_supply: Vec<_> = self.nodes.iter().map(|x| x.supply).collect();
        let edge_a: Vec<_> = self.edges.iter().map(|x| x.a.0 as i64).collect();
        let edge_b: Vec<_> = self.edges.iter().map(|x| x.b.0 as i64).collect();
        let edge_capacity: Vec<_> = self.edges.iter().map(|x| x.data.capacity).collect();
        let edge_cost: Vec<_> = self.edges.iter().map(|x| x.data.cost).collect();
        let mut edge_flow_result = vec![0; self.edges.len()];
        let result;
        unsafe {
            result = network_simplex_mcmf_i64(num_vertices, num_edges,
                node_supply.as_ptr(), edge_a.as_ptr(), edge_b.as_ptr(),
                edge_capacity.as_ptr(), edge_cost.as_ptr(), edge_flow_result.as_mut_ptr());
        }
        for (edge, &flow) in self.edges.iter_mut().zip(edge_flow_result.iter()) {
            edge.data.flow = flow;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn mcmf() {
        let mut G = Graph::new_default(4);
        G.increase_supply(Node(0), 20);
        G.decrease_supply(Node(3), 20);
        G.add_edge(Node(0), Node(1), EdgeData::new(Cost(100), Capacity(10)));
        G.add_edge(Node(0), Node(2), EdgeData::new(Cost(300), Capacity(20)));
        G.add_edge(Node(1), Node(2), EdgeData::new(Cost(50), Capacity(5)));
        G.add_edge(Node(1), Node(3), EdgeData::new(Cost(200), Capacity(10)));
        G.add_edge(Node(2), Node(3), EdgeData::new(Cost(100), Capacity(20)));
        let cost = G.mcmf();
        let (_, edges) = G.extract();
        let flow: Vec<_> = edges.iter().map(|x| x.data.flow).collect();
        assert_eq!(cost, 6750);
        assert_eq!(flow, vec![10, 10, 5, 5, 15]);
    }
}
