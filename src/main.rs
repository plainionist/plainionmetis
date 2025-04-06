use std::env;

mod use_cases;
mod utils;
use use_cases::{chat, cluster, explore, query};
use utils::config;

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
            plainionmetis <config-file> query <idea>\n\
            plainionmetis <config-file> explore <topic>\n\
            plainionmetis <config-file> cluster <num-clusters>\n\
            plainionmetis <config-file> chat"
        );
        std::process::exit(1);
    }

    let config_file_path = &args[1];
    let config = config::load(config_file_path);

    let cmd = &args[2];
    match cmd.as_str() {
        "explore" => explore::run(&config, &args[3..].join(" ")),
        "cluster" => {
            let num_clusters = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5);
            cluster::run(&config, num_clusters);
        }
        "chat" => chat::run(&config),
        "query" => query::run(&config, &args[2..].join(" ")),
        _ => {
            eprintln!("Unknown command: {}", cmd);
            std::process::exit(1);
        }
    }
}
