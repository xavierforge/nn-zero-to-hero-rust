use crate::engine::Value;
use std::collections::HashMap;

/// Trace the computation graph and return node ID map and edges
fn trace(
    root: &Value,
) -> (
    HashMap<usize, Value>,
    Vec<(usize, usize, Option<&'static str>)>,
) {
    let mut seen: HashMap<*const (), usize> = HashMap::new();
    let mut nodes: HashMap<usize, Value> = HashMap::new();
    let mut edges: Vec<(usize, usize, Option<&'static str>)> = Vec::new();
    let mut next_id: usize = 0;

    fn build(
        v: &Value,
        seen: &mut HashMap<*const (), usize>,
        nodes: &mut HashMap<usize, Value>,
        edges: &mut Vec<(usize, usize, Option<&'static str>)>,
        next_id: &mut usize,
    ) -> usize {
        let ptr = v.ptr();
        if let Some(&id) = seen.get(&ptr) {
            return id;
        }

        let id = *next_id;
        *next_id += 1;

        seen.insert(ptr, id);
        nodes.insert(id, v.clone());

        for child in v.prev() {
            let child_id = build(&child, seen, nodes, edges, next_id);
            edges.push((child_id, id, v.op()));
        }

        id
    }

    build(root, &mut seen, &mut nodes, &mut edges, &mut next_id);
    (nodes, edges)
}

pub fn draw_dot(root: &Value, output_path: &str) {
    let (nodes, edges) = trace(root);

    let mut dot_string = String::new();
    dot_string.push_str("digraph trace_graph {\n");
    dot_string.push_str("rankdir=LR;\n");

    // Add all value nodes
    for (id, val) in &nodes {
        let name = match val.label() {
            Some(name) => name,
            None => String::new(),
        };
        let label = format!(
            "{{ {} | data {:.4} | grad {:.4} }}",
            name,
            val.data(),
            val.grad()
        );
        dot_string.push_str(&format!("n{} [label=\"{}\", shape=record];\n", id, label));

        if val.op().is_some() {
            dot_string.push_str(&format!(
                "op{} [label=\"{}\", shape=circle];\n",
                id,
                val.op().unwrap()
            ));
            dot_string.push_str(&format!("op{} -> n{};\n", id, id)); // op to this value
        }
    }

    // Draw edges from input nodes to op nodes
    for (from, to, op) in &edges {
        if op.is_some() {
            dot_string.push_str(&format!("n{} -> op{};\n", from, to)); // input to op
        } else {
            dot_string.push_str(&format!("n{} -> n{};\n", from, to)); // raw edge (no op)
        }
    }

    dot_string.push_str("}\n");

    let graph = graphviz_rust::parse(&dot_string).expect("Failed to parse dot");
    graphviz_rust::exec(
        graph,
        &mut graphviz_rust::printer::PrinterContext::default(),
        vec![
            graphviz_rust::cmd::Format::Svg.into(),
            graphviz_rust::cmd::CommandArg::Output(output_path.to_string()),
        ],
    )
    .expect("Failed to generate graph");
}
