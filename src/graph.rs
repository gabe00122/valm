extern crate rand;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::{distr::Uniform, prelude::*};

#[derive(Clone, FromPyObject)]
pub struct GraphSettings {
    num_nodes: usize,
    num_edges: usize,
}

pub struct GraphShared {
    settings: GraphSettings,
}

impl EnvShared for GraphShared {
    type Settings = GraphSettings;

    fn new(settings: Self::Settings) -> Self {
        Self { settings }
    }
}

struct GraphEnvInstance {
    shared: Arc<GraphShared>,
    rng: SmallRng,
    adj: Vec<Vec<usize>>,
    num_nodes: usize,
    start: usize,
    end: usize,
    shortest_path: Vec<usize>,
}

fn node_label(i: usize) -> char {
    (b'A' + i as u8) as char
}

fn label_to_index(c: char) -> Option<usize> {
    if c.is_ascii_uppercase() {
        Some((c as u8 - b'A') as usize)
    } else {
        None
    }
}

/// Generate a connected undirected graph with `num_nodes` nodes and `num_edges` edges.
/// First builds a random spanning tree to guarantee connectivity, then adds random edges.
fn generate_connected_graph(rng: &mut impl Rng, num_nodes: usize, num_edges: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![vec![]; num_nodes];
    let mut edge_set = std::collections::HashSet::new();

    // Build a random spanning tree by shuffling nodes and connecting them in sequence
    let mut perm: Vec<usize> = (0..num_nodes).collect();
    perm.shuffle(rng);

    for i in 1..num_nodes {
        let a = perm[i - 1];
        let b = perm[i];
        let key = (a.min(b), a.max(b));
        edge_set.insert(key);
        adj[a].push(b);
        adj[b].push(a);
    }

    // Add remaining random edges until we reach num_edges
    let max_possible = num_nodes * (num_nodes - 1) / 2;
    let target = num_edges.min(max_possible);

    let node_dist = Uniform::new(0, num_nodes).unwrap();
    while edge_set.len() < target {
        let a: usize = rng.sample(node_dist);
        let b: usize = rng.sample(node_dist);
        if a == b {
            continue;
        }
        let key = (a.min(b), a.max(b));
        if edge_set.insert(key) {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    adj
}

/// BFS from `start` to `end`. Returns the shortest path as a list of node indices.
fn bfs_shortest_path(adj: &[Vec<usize>], start: usize, end: usize) -> Vec<usize> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if node == end {
            break;
        }
        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = node;
                queue.push_back(neighbor);
            }
        }
    }

    // Reconstruct path
    let mut path = vec![];
    let mut cur = end;
    while cur != usize::MAX {
        path.push(cur);
        if cur == start {
            break;
        }
        cur = parent[cur];
    }
    path.reverse();
    path
}

fn edges_as_string(adj: &[Vec<usize>]) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut parts = vec![];
    for (i, neighbors) in adj.iter().enumerate() {
        for &j in neighbors {
            let key = (i.min(j), i.max(j));
            if seen.insert(key) {
                parts.push(format!("{}-{}", node_label(i), node_label(j)));
            }
        }
    }
    parts.join(", ")
}

/// Parse the LLM response to extract a sequence of node labels.
/// Accepts formats like "A -> B -> D -> E", "A B D E", "A, B, D, E", or mixed.
fn parse_path(response: &str, num_nodes: usize) -> Option<Vec<usize>> {
    // Extract all uppercase single letters that are valid node labels
    let max_label = (b'A' + num_nodes as u8) as char;
    let mut nodes = vec![];

    for c in response.chars() {
        if c.is_ascii_uppercase() && c < max_label {
            nodes.push(label_to_index(c).unwrap());
        }
    }

    if nodes.is_empty() {
        None
    } else {
        Some(nodes)
    }
}

/// Check if a path is valid: all consecutive pairs must share an edge.
fn is_valid_path(adj: &[Vec<usize>], path: &[usize]) -> bool {
    for pair in path.windows(2) {
        if !adj[pair[0]].contains(&pair[1]) {
            return false;
        }
    }
    true
}

impl EnvInstance for GraphEnvInstance {
    type Shared = GraphShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        GraphEnvInstance {
            shared,
            rng: SmallRng::seed_from_u64(seed),
            adj: vec![],
            num_nodes: 0,
            start: 0,
            end: 0,
            shortest_path: vec![],
        }
    }

    fn reset(&mut self) -> String {
        let num_nodes = self.shared.settings.num_nodes;
        let num_edges = self.shared.settings.num_edges;

        let adj = generate_connected_graph(&mut self.rng, num_nodes, num_edges);

        // Pick two distinct random nodes
        let node_dist = Uniform::new(0, num_nodes).unwrap();
        let start: usize = self.rng.sample(node_dist);
        let mut end: usize = self.rng.sample(node_dist);
        while end == start {
            end = self.rng.sample(node_dist);
        }

        let shortest_path = bfs_shortest_path(&adj, start, end);
        let edge_str = edges_as_string(&adj);

        let prompt = format!(
            "Graph edges: {}\nFind the shortest path from {} to {}.",
            edge_str,
            node_label(start),
            node_label(end)
        );

        self.adj = adj;
        self.num_nodes = num_nodes;
        self.start = start;
        self.end = end;
        self.shortest_path = shortest_path;

        prompt
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let reward = match parse_path(action, self.num_nodes) {
            Some(path) => {
                if path.is_empty()
                    || *path.first().unwrap() != self.start
                    || *path.last().unwrap() != self.end
                {
                    // Doesn't start/end correctly
                    0.0
                } else if !is_valid_path(&self.adj, &path) {
                    // Invalid edges
                    0.0
                } else {
                    let optimal_len = self.shortest_path.len();
                    let actual_len = path.len();
                    if actual_len <= optimal_len {
                        1.0
                    } else {
                        optimal_len as f32 / actual_len as f32
                    }
                }
            }
            None => 0.0,
        };

        let done = true;
        (self.reset(), reward, done)
    }
}

create_env_wrapper!(
    GraphEnv,
    GraphEnvInstance,
    GraphSettings,
    "You are given an undirected graph as a list of edges and two nodes. Find the shortest path between them. Output the path as a sequence of node labels separated by ' -> ', for example: A -> B -> D -> E. Only output the path, nothing else."
);
