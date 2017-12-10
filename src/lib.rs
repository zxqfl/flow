//! This crate is for solving instances of the [minimum cost maximum flow problem](https://en.wikipedia.org/wiki/Minimum-cost_flow_problem).
//! It uses the network simplex algorithm from the [LEMON](http://lemon.cs.elte.hu/trac/lemon) graph optimization library.
//! 
//! # Example
//! ```
//! use flow::{GraphBuilder, Vertex, Cost, Capacity};
//! let (cost, flows) = GraphBuilder::new()
//!     .add_edge(Vertex::Source, "Vancouver", Capacity(2), Cost(0))
//!     .add_edge("Vancouver", "Toronto", Capacity(2), Cost(100))
//!     .add_edge("Toronto", "Halifax", Capacity(1), Cost(150))
//!     .add_edge("Vancouver", "Halifax", Capacity(5), Cost(400))
//!     .add_edge("Halifax", Vertex::Sink, Capacity(2), Cost(0))
//!     .mcmf();
//! assert_eq!(cost, 650);
//! ```

use std::collections::BTreeMap;
use std::iter;

#[link(name="flow")]
extern {
    fn network_simplex_mcmf_i64(num_vertices: i64, num_edges: i64,
        node_supply: *const i64, edge_a: *const i64, edge_b: *const i64,
        edge_capacity: *const i64, edge_cost: *const i64, edge_flow_result: *mut i64) -> i64;
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Node(usize);

struct Edge {
    pub a: Node,
    pub b: Node,
    pub data: EdgeData,
}

struct Graph {
    nodes: Vec<NodeData>,
    edges: Vec<Edge>,
}

impl Graph {
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
struct NodeData {
    supply: i64,
}

#[derive(Clone, Copy)]
struct EdgeData {
    cost: i64,
    capacity: i64,
    flow: i64,
}

/// Wrapper type representing the cost of an edge in the graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cost(pub i64);
/// Wrapper type representing the capacity of an edge in the graph.
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


#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Vertex<T: Ord + Clone> {
    Source, Sink, Node(T)
}

impl<T> From<T> for Vertex<T> where T: Clone + Ord {
    fn from(x: T) -> Vertex<T> {
        Vertex::Node(x)
    }
}

/// Represents flow in a solution to the minimum cost maximum flow problem.
#[derive(Clone, Copy)]
pub struct Flow<V, Cost> {
    pub a: V,
    pub b: V,
    pub amount: i64,
    pub cost: Cost
}

/// Use this struct to build a graph, then call the `mcmf()` function to find its minimum cost maximum flow.
#[derive(Clone)]
pub struct GraphBuilder<T: Ord + Clone> {
    pub edge_list: Vec<(Vertex<T>, Vertex<T>, Capacity, Cost)>
}

impl<T> GraphBuilder<T> where T: Ord + Clone {
    pub fn new() -> Self {
        GraphBuilder {edge_list: Vec::new()}
    }

    /// Add an edge to the graph.
    ///
    /// `capacity` and `cost` have wrapper types so that you can't mix them up.
    pub fn add_edge<A: Into<Vertex<T>>, B: Into<Vertex<T>>>(&mut self, a: A, b: B, capacity: Capacity, cost: Cost) -> &mut Self {
        let a = a.into();
        let b = b.into();
        assert!(a != b);
        assert!(a != Vertex::Sink);
        assert!(b != Vertex::Source);
        self.edge_list.push((a, b, capacity, cost));
        self
    }

    /// Computes the minimum cost maximum flow.
    /// Returns a tuple (total cost, vector of flows).
    pub fn mcmf(&self) -> (i64, Vec<Flow<Vertex<T>, i64>>) {
        let mut next_id = 0;
        let source = Vertex::Source.clone();
        let sink = Vertex::Sink.clone();
        let mut index_mapper = BTreeMap::new();
        for vertex in self.edge_list.iter()
                .flat_map(move |&(ref a, ref b, _, _)| iter::once(a).chain(iter::once(b)))
                .chain(iter::once(&source))
                .chain(iter::once(&sink)) {
            if !index_mapper.contains_key(&vertex) {
                index_mapper.insert(vertex, next_id);
                next_id += 1;
            }
        }
        let num_vertices = next_id;
        let mut g = Graph::new_default(num_vertices);
        for &(ref a, ref b, cap, cost) in &self.edge_list {
            let node_a = Node(*index_mapper.get(&a).unwrap());
            let node_b = Node(*index_mapper.get(&b).unwrap());
            if *a == Vertex::Source || *b == Vertex::Sink {
                // The + supply and - supply must be equal because of how LEMON interprets
                // its input.
                // http://lemon.cs.elte.hu/pub/doc/latest/a00005.html
                g.increase_supply(Node(*index_mapper.get(&Vertex::Source).unwrap()), cap.0);
                g.decrease_supply(Node(*index_mapper.get(&Vertex::Sink).unwrap()), cap.0);
            }
            g.add_edge(node_a, node_b, EdgeData::new(cost, cap));
        }
        let total_amount = g.mcmf();
        let (_, edges) = g.extract();
        let inverse_mapping: BTreeMap<_, _> =
            index_mapper.into_iter().map(|(a, b)| (b, a)).collect();
        (total_amount, edges.into_iter().map(|x| {
            let a = (**inverse_mapping.get(&x.a.0).unwrap()).clone();
            let b = (**inverse_mapping.get(&x.b.0).unwrap()).clone();
            let amount = x.data.flow;
            let cost = x.data.cost;
            Flow {a, b, amount, cost}
        })
            .filter(|x| x.amount != 0)
            .collect())
    }
}

/// A wrapper around `GraphBuilder` with floats as the costs.
#[derive(Clone)]
pub struct GraphBuilderFloat<T: Ord + Clone> {
    pub builder: GraphBuilder<T>,
    unit: f64,
}

impl<T> GraphBuilderFloat<T> where T: Ord + Clone {
    /// Creates a new `GraphBuilderFloat`.
    /// Edge costs will be rounded to the nearest multiple of `unit`.
    pub fn new(unit: f64) -> Self {
        GraphBuilderFloat {builder: GraphBuilder::new(), unit}
    }

    fn quantize(&self, x: f64) -> i64 {
        (x / self.unit).round() as i64
    }
    fn unquantize(&self, x: i64) -> f64 {
        x as f64 * self.unit
    }

    /// Add an edge to the graph.
    pub fn add_edge<A: Into<Vertex<T>>, B: Into<Vertex<T>>>(&mut self, a: A, b: B,
            capacity: i64, cost: f64) {
        let cost = self.quantize(cost);
        self.builder.add_edge(a, b, Capacity(capacity), Cost(cost));
    }

    /// Computes the minimum cost maximum flow.
    /// Returns a tuple (total cost, vector of flows).
    pub fn mcmf(&self) -> (f64, Vec<Flow<Vertex<T>, f64>>) {
        let (total, flows) = self.builder.mcmf();
        let total = self.unquantize(total);
        let flows = flows.into_iter()
            .map(|flow| Flow {
                a: flow.a,
                b: flow.b,
                amount: flow.amount,
                cost: self.unquantize(flow.cost)})
            .collect();
        (total, flows)
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
