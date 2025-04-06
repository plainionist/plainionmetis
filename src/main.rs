use std::env;

mod utils;
mod use_cases;
use use_cases::{chat, cluster, explore, query};

// prepare:
// - "ollama pull phi3:mini"
// - "ollama pull nomic-embed-text"
// then run "ollama run phi3:mini"
// then run this program

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage:\n\
            plainionmetis <config-file> <idea>\n\
            plainionmetis explore <config-file> <topic>\n\
            plainionmetis cluster <config-file> <num-clusters>\n\
            plainionmetis chat <config-file>"
        );
        std::process::exit(1);
    }

    let mode = &args[1];
    match mode.as_str() {
        "explore" => explore::run(&args[2], &args[3..].join(" ")),
        "cluster" => {
            let num_clusters = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5);
            cluster::run(&args[2], num_clusters);
        }
        "chat" => {
            let config_path = &args[2];
            chat::run(config_path);
        }
        _ => query::run(&args[1], &args[2..].join(" ")),
    }
}
