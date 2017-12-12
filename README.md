# flow
This crate is for solving instances of the [minimum cost maximum flow problem](https://en.wikipedia.org/wiki/Minimum-cost_flow_problem).
It uses the network simplex algorithm from the [LEMON](http://lemon.cs.elte.hu/trac/lemon) graph optimization library.

# Example
```rust
use flow::{GraphBuilder, Vertex, Cost, Capacity};
let (cost, paths) = GraphBuilder::new()
    .add_edge(Vertex::Source, "Vancouver", Capacity(2), Cost(0))
    .add_edge("Vancouver", "Toronto", Capacity(2), Cost(100))
    .add_edge("Toronto", "Halifax", Capacity(1), Cost(150))
    .add_edge("Vancouver", "Halifax", Capacity(5), Cost(400))
    .add_edge("Halifax", Vertex::Sink, Capacity(2), Cost(0))
    .mcmf();
assert_eq!(cost, 650);
assert_eq!(cost, paths.iter().map(|path| path.cost()).sum());
assert_eq!(paths.len(), 2);
assert!(
    paths[0].vertices() == vec![
        &Vertex::Source,
        &Vertex::Node("Vancouver"),
        &Vertex::Node("Halifax"),
        &Vertex::Sink]);
assert!(
    paths[1].vertices() == vec![
        &Vertex::Source,
        &Vertex::Node("Vancouver"),
        &Vertex::Node("Toronto"),
        &Vertex::Node("Halifax"),
        &Vertex::Sink]);
```
