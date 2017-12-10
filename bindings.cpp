#include "lemon/network_simplex.h"
#include "lemon/smart_graph.h"

#include <stdint.h>

#include <vector>
#include <cassert>

using namespace lemon;

template<typename Int>
Int network_simplex_mcmf(Int num_vertices, Int num_edges, const Int* node_supply, const Int* edge_a,
        const Int* edge_b, const Int* edge_capacity, const Int* edge_cost, Int* edge_flow_result) {
    for (Int i = 0; i < num_edges; i++) {
        assert(0 <= edge_a[i] && edge_a[i] < num_vertices);
        assert(0 <= edge_b[i] && edge_b[i] < num_vertices);
    }
    DIGRAPH_TYPEDEFS(SmartDigraph);
    SmartDigraph G;
    std::vector<Node> nodes;
    for (Int i = 0; i < num_vertices; i++) {
        nodes.push_back(G.addNode());
    }
    std::vector<Arc> arcs;
    for (Int i = 0; i < num_edges; i++) {
        arcs.push_back(G.addArc(nodes[edge_a[i]], nodes[edge_b[i]]));
    }
    SmartDigraph::NodeMap<Int> supplies(G);
    for (Int i = 0; i < num_vertices; i++) {
        supplies[nodes[i]] = node_supply[i];
    }
    SmartDigraph::ArcMap<Int> capacities(G);
    for (Int i = 0; i < num_edges; i++) {
        capacities[arcs[i]] = edge_capacity[i];
    }
    SmartDigraph::ArcMap<Int> costs(G);
    for (Int i = 0; i < num_edges; i++) {
        costs[arcs[i]] = edge_cost[i];
    }
    NetworkSimplex<SmartDigraph> ns(G);
    ns.supplyMap(supplies).upperMap(capacities).costMap(costs);
    ns.run();
    for (Int i = 0; i < num_edges; i++) {
        edge_flow_result[i] = ns.flow(arcs[i]);
    }
    return ns.totalCost();
}

extern "C" int64_t network_simplex_mcmf_i64(int64_t num_vertices, int64_t num_edges,
        const int64_t* node_supply, const int64_t* edge_a, const int64_t* edge_b,
        const int64_t* edge_capacity, const int64_t* edge_cost, int64_t* edge_flow_result) {
    return network_simplex_mcmf(num_vertices, num_edges, node_supply, edge_a, edge_b, edge_capacity,
        edge_cost, edge_flow_result);
}
