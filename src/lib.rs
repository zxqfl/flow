//! This crate is for solving instances of the [minimum cost maximum flow problem](https://en.wikipedia.org/wiki/Minimum-cost_flow_problem).
//! It uses the network simplex algorithm from the [LEMON](http://lemon.cs.elte.hu/trac/lemon) graph optimization library.
//!
//! # Example
//! ```
//! use mcmf::{GraphBuilder, Vertex, Cost, Capacity};
//! let (cost, paths) = GraphBuilder::new()
//!     .add_edge(Vertex::Source, "Vancouver", Capacity(2), Cost(0))
//!     .add_edge("Vancouver", "Toronto", Capacity(2), Cost(100))
//!     .add_edge("Toronto", "Halifax", Capacity(1), Cost(150))
//!     .add_edge("Vancouver", "Halifax", Capacity(5), Cost(400))
//!     .add_edge("Halifax", Vertex::Sink, Capacity(2), Cost(0))
//!     .mcmf();
//! assert_eq!(cost, 650);
//! assert_eq!(cost, paths.iter().map(|path| path.cost()).sum());
//! assert_eq!(paths.len(), 2);
//! assert!(
//!     paths[0].vertices() == vec![
//!         &Vertex::Source,
//!         &Vertex::Node("Vancouver"),
//!         &Vertex::Node("Halifax"),
//!         &Vertex::Sink]);
//! assert!(
//!     paths[1].vertices() == vec![
//!         &Vertex::Source,
//!         &Vertex::Node("Vancouver"),
//!         &Vertex::Node("Toronto"),
//!         &Vertex::Node("Halifax"),
//!         &Vertex::Sink]);
//! ```

use std::cmp::min;
use std::collections::BTreeMap;
use std::iter;

#[link(name = "flow")]
extern "C" {
    fn network_simplex_mcmf_i64(
        num_vertices: i64,
        num_edges: i64,
        node_supply: *const i64,
        edge_a: *const i64,
        edge_b: *const i64,
        edge_capacity: *const i64,
        edge_cost: *const i64,
        edge_flow_result: *mut i64,
    ) -> i64;
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
        self.edges.push(Edge { a, b, data });
        self
    }
    pub fn extract(self) -> (Vec<NodeData>, Vec<Edge>) {
        (self.nodes, self.edges)
    }
}

impl Graph {
    pub fn new_default(num_vertices: usize) -> Self {
        let nodes = vec![Default::default(); num_vertices];
        Graph {
            nodes,
            edges: Vec::new(),
        }
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
pub struct Cost(pub i32);
/// Wrapper type representing the capacity of an edge in the graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capacity(pub i32);

impl EdgeData {
    pub fn new(cost: Cost, capacity: Capacity) -> Self {
        let cost = cost.0 as i64;
        let capacity = capacity.0 as i64;
        assert!(capacity >= 0);
        EdgeData {
            cost,
            capacity,
            flow: Default::default(),
        }
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
        let node_supply: Vec<_> = self.nodes.iter().map(|x| clamp_to_i32(x.supply)).collect();
        let edge_a: Vec<_> = self.edges.iter().map(|x| x.a.0 as i64).collect();
        let edge_b: Vec<_> = self.edges.iter().map(|x| x.b.0 as i64).collect();
        let edge_capacity: Vec<_> = self.edges.iter().map(|x| x.data.capacity).collect();
        let edge_cost: Vec<_> = self.edges.iter().map(|x| x.data.cost).collect();
        let mut edge_flow_result = vec![0; self.edges.len()];
        let result;
        unsafe {
            result = network_simplex_mcmf_i64(
                num_vertices,
                num_edges,
                node_supply.as_ptr(),
                edge_a.as_ptr(),
                edge_b.as_ptr(),
                edge_capacity.as_ptr(),
                edge_cost.as_ptr(),
                edge_flow_result.as_mut_ptr(),
            );
        }
        for (edge, &flow) in self.edges.iter_mut().zip(edge_flow_result.iter()) {
            edge.data.flow = flow;
        }
        result
    }
}

fn clamp_to_i32(x: i64) -> i64 {
    let limit = std::i32::MAX as i64;
    let x = std::cmp::min(x, limit);

    std::cmp::max(x, -limit)
}

/// This class represents a vertex in a graph.
/// It is parametrized by `T` so that users of the library can use the most convenient type for representing nodes in the graph.
#[derive(Copy, Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Vertex<T: Clone + Ord> {
    Source,
    Sink,
    Node(T),
}

impl<T> Vertex<T>
where
    T: Clone + Ord,
{
    /// Maps `Source`, `Sink`, and `Node(x)` to `None`, `None`, and `Some(x)` respectively.
    pub fn as_option(self) -> Option<T> {
        match self {
            Vertex::Source => None,
            Vertex::Sink => None,
            Vertex::Node(x) => Some(x),
        }
    }
}

impl<T> From<T> for Vertex<T>
where
    T: Clone + Ord,
{
    fn from(x: T) -> Vertex<T> {
        Vertex::Node(x)
    }
}

/// Represents flow in a solution to the minimum cost maximum flow problem.
#[derive(Clone)]
pub struct Flow<T: Clone + Ord> {
    pub a: Vertex<T>,
    pub b: Vertex<T>,
    pub amount: u64,
    pub cost: i32,
}

/// Represents a path from the source to the sink in a solution to the minimum cost maximum flow problem.
pub struct Path<T: Clone + Ord> {
    pub flows: Vec<Flow<T>>,
}

impl<T> Path<T>
where
    T: Clone + Ord,
{
    /// A list of all the vertices in the path.
    /// Always begins with `Vertex::Source` and ends with `Vertex::Sink`.
    pub fn vertices(&self) -> Vec<&Vertex<T>> {
        iter::once(&self.flows[0].a)
            .chain(self.flows.iter().map(|x| &x.b))
            .collect()
    }

    /// A list of all the edges in the path.
    pub fn edges(&self) -> Vec<&Flow<T>> {
        self.flows.iter().collect()
    }

    /// Returns the total cost of the path.
    /// `path.cost()` is always a multiple of `path.amount()`.
    pub fn cost(&self) -> i32 {
        self.flows
            .iter()
            .map(|flow| flow.amount as i32 * flow.cost)
            .sum()
    }

    /// Returns the amount of flow in the path.
    pub fn amount(&self) -> u64 {
        self.flows[0].amount
    }

    /// Returns the number of edges in the path.
    pub fn len(&self) -> usize {
        self.flows.len()
    }

    /// Returns the number of edges in the path.
    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }
}

/// Use this struct to build a graph, then call the `mcmf()` function to find its minimum cost maximum flow.
/// # Example
/// ```
/// use mcmf::{GraphBuilder, Vertex, Cost, Capacity};
/// let (cost, paths) = GraphBuilder::new()
///     .add_edge(Vertex::Source, "Vancouver", Capacity(2), Cost(0))
///     .add_edge("Vancouver", "Toronto", Capacity(2), Cost(100))
///     .add_edge("Toronto", "Halifax", Capacity(1), Cost(150))
///     .add_edge("Vancouver", "Halifax", Capacity(5), Cost(400))
///     .add_edge("Halifax", Vertex::Sink, Capacity(2), Cost(0))
///     .mcmf();
/// assert_eq!(cost, 650);
/// assert_eq!(cost, paths.iter().map(|path| path.cost()).sum());
/// assert_eq!(paths.len(), 2);
/// assert!(
///     paths[0].vertices() == vec![
///         &Vertex::Source,
///         &Vertex::Node("Vancouver"),
///         &Vertex::Node("Halifax"),
///         &Vertex::Sink]);
/// assert!(
///     paths[1].vertices() == vec![
///         &Vertex::Source,
///         &Vertex::Node("Vancouver"),
///         &Vertex::Node("Toronto"),
///         &Vertex::Node("Halifax"),
///         &Vertex::Sink]);
/// ```
#[derive(Clone, Default)]
pub struct GraphBuilder<T: Clone + Ord> {
    pub edge_list: Vec<(Vertex<T>, Vertex<T>, Capacity, Cost)>,
}

impl<T> GraphBuilder<T>
where
    T: Clone + Ord,
{
    /// Creates a new empty graph.
    pub fn new() -> Self {
        GraphBuilder {
            edge_list: Vec::new(),
        }
    }

    /// Add an edge to the graph.
    ///
    /// `capacity` and `cost` have wrapper types so that you can't mix them up.
    ///
    /// Panics if `capacity` is negative.
    pub fn add_edge<A: Into<Vertex<T>>, B: Into<Vertex<T>>>(
        &mut self,
        a: A,
        b: B,
        capacity: Capacity,
        cost: Cost,
    ) -> &mut Self {
        if capacity.0 < 0 {
            panic!("capacity cannot be negative (capacity was {})", capacity.0)
        }
        let a = a.into();
        let b = b.into();
        assert!(a != b);
        assert!(a != Vertex::Sink);
        assert!(b != Vertex::Source);
        self.edge_list.push((a, b, capacity, cost));
        self
    }

    /// Computes the minimum cost maximum flow.
    ///
    /// Returns a tuple (total cost, list of paths). The paths are sorted in ascending order by length.
    ///
    /// This gives incorrect results when the total cost or the total flow exceeds 2^(31)-1.
    /// It is the responsibility of the caller to ensure that the total cost doesn't exceed 2^(31)-1.
    pub fn mcmf(&self) -> (i32, Vec<Path<T>>) {
        let mut next_id = 0;
        let source = Vertex::Source.clone();
        let sink = Vertex::Sink.clone();
        let mut index_mapper = BTreeMap::new();
        for vertex in self
            .edge_list
            .iter()
            .flat_map(move |(a, b, _, _)| iter::once(a).chain(iter::once(b)))
            .chain(iter::once(&source))
            .chain(iter::once(&sink))
        {
            if let std::collections::btree_map::Entry::Vacant(e) = index_mapper.entry(vertex) {
                e.insert(next_id);
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
                g.increase_supply(
                    Node(*index_mapper.get(&Vertex::Source).unwrap()),
                    cap.0 as i64,
                );
                g.decrease_supply(
                    Node(*index_mapper.get(&Vertex::Sink).unwrap()),
                    cap.0 as i64,
                );
            }
            g.add_edge(node_a, node_b, EdgeData::new(cost, cap));
        }
        let total_amount = g.mcmf() as i32;
        let (_, edges) = g.extract();
        let inverse_mapping: BTreeMap<_, _> =
            index_mapper.into_iter().map(|(a, b)| (b, a)).collect();
        let flows = edges
            .into_iter()
            .map(|x| {
                let a = (**inverse_mapping.get(&x.a.0).unwrap()).clone();
                let b = (**inverse_mapping.get(&x.b.0).unwrap()).clone();
                let amount = x.data.flow as u64;
                let cost = x.data.cost as i32;
                Flow { a, b, amount, cost }
            })
            .filter(|x| x.amount != 0)
            .collect();
        let mut paths = GraphBuilder::path_decomposition(flows);
        paths.sort_by_key(|path| path.len());
        (total_amount, paths)
    }

    fn path_decomposition(flows: Vec<Flow<T>>) -> Vec<Path<T>> {
        let mut adj: BTreeMap<Vertex<T>, Vec<Flow<T>>> =
            flows.iter().map(|x| (x.a.clone(), Vec::new())).collect();
        for x in flows {
            adj.get_mut(&x.a).unwrap().push(x);
        }
        fn decompose<T: Clone + Ord>(
            adj: &mut BTreeMap<Vertex<T>, Vec<Flow<T>>>,
            v: &Vertex<T>,
            parent_amount: u64,
        ) -> (u64, Vec<Flow<T>>) {
            if *v == Vertex::Sink {
                (std::u64::MAX, Vec::new())
            } else if adj.get(v).into_iter().all(|x| x.is_empty()) {
                (0, Vec::new())
            } else {
                let flow = adj.get_mut(v).unwrap().pop().unwrap();
                let amount = min(parent_amount, flow.amount);
                let (child_amount, child_path) = decompose(adj, &flow.b, amount);
                let amount = min(amount, child_amount);
                let mut path = child_path;
                if amount < flow.amount {
                    adj.get_mut(v).unwrap().push(Flow {
                        amount: flow.amount - amount,
                        ..flow.clone()
                    });
                }
                path.push(Flow { amount, ..flow });
                (amount, path)
            }
        }
        let mut result = Vec::new();
        loop {
            let (flow, path) = decompose(&mut adj, &Vertex::Source, std::u64::MAX);
            if flow == 0 {
                break;
            } else {
                result.push(path.into_iter().rev().collect());
            }
        }
        result.into_iter().map(|x| Path { flows: x }).collect()
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

    #[test]
    fn large_number() {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
        enum OnlyNode {
            Only,
        }
        for i in 0..48 {
            let x = i * 1000;
            println!("x={}", x);
            let (total, _) = GraphBuilder::new()
                .add_edge(Vertex::Source, OnlyNode::Only, Capacity(x), Cost(x))
                .add_edge(Vertex::Source, OnlyNode::Only, Capacity(x), Cost(x))
                .add_edge(OnlyNode::Only, Vertex::Sink, Capacity(x), Cost(0))
                .mcmf();
            assert_eq!(total, (x as i64 * x as i64) as i32);
        }
    }

    #[test]
    fn empty_graph() {
        let (cost, paths) = GraphBuilder::<i32>::new().mcmf();
        assert_eq!(cost, 0);
        assert!(paths.is_empty())
    }

    #[test]
    fn large_capacities() {
        let max = 1 << 30;
        assert_eq!(max, 1073741824);
        let (cost, paths) = GraphBuilder::new()
            .add_edge(Vertex::Source, "A", Capacity(max), Cost(0))
            .add_edge("A", Vertex::Sink, Capacity(max), Cost(0))
            .mcmf();
        assert_eq!(cost, 0);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    #[should_panic]
    fn negative_capacity_panics() {
        GraphBuilder::new().add_edge("a", "b", Capacity(-1), Cost(0));
    }
}
