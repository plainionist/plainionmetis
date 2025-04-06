use rand::{seq::SliceRandom, thread_rng};
use std::env;
use std::io::{self, Write};

mod chunking;
mod config;
mod ollama;
use chunking::Chunk;

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
        "explore" => explore_mode(&args[2], &args[3..].join(" ")),
        "cluster" => {
            let num_clusters = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5);
            cluster_mode(&args[2], num_clusters);
        }
        "chat" => {
            let config_path = &args[2];
            chat_mode(config_path);
        }
        _ => query_mode(&args[1], &args[2..].join(" ")),
    }
}

fn query_mode(config_file_path: &str, idea: &str) {
    let config = config::load(config_file_path); // use config file path now

    let embedded_chunks = chunking::load(&config);

    let query_embedding = ollama::embed_text(idea).expect("Failed to embed idea text");

    let top_chunks = find_similar_chunks(&embedded_chunks, &query_embedding, 5);

    println!("Top {} matching chunks:\n", top_chunks.len());
    for (i, chunk) in top_chunks.iter().enumerate() {
        println!("{}. {}", i + 1, chunk.file_path);
    }

    let joined = top_chunks
        .iter()
        .map(|c| format!("• {}", c.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "These ideas appear related:\n\n{}\n\nDescribe their connection or common theme.",
        joined
    );

    let result = ollama::ask(&prompt);
    println!("\n🧠 >\n{}", result);
}

fn find_similar_chunks(chunks: &[Chunk], query: &[f32], top_n: usize) -> Vec<Chunk> {
    let mut scored: Vec<(f32, &Chunk)> = chunks
        .iter()
        .filter_map(|chunk| cosine_similarity(&chunk.embedding, query).map(|score| (score, chunk)))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_n)
        .map(|(_, chunk)| chunk.clone())
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    Some(dot / (norm_a * norm_b + 1e-8))
}

fn explore_mode(config_file_path: &str, topic: &str) {
    println!("Exploring topic: '{}'", topic);

    let config = config::load(config_file_path);

    let embedded_chunks = chunking::load(&config);

    let topic_embedding = ollama::embed_text(topic).expect("Failed to embed topic");

    let top_chunks = find_similar_chunks(&embedded_chunks, &topic_embedding, 10);
    println!("Top matching ideas related to '{}':\n", topic);

    for (i, chunk) in top_chunks.iter().enumerate() {
        println!("{}. {}", i + 1, chunk.file_path);
    }

    let joined = top_chunks
        .iter()
        .map(|c| format!("• {}", c.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "Based on the following notes, summarize what I think or understand about '{}':\n\n{}",
        topic, joined
    );

    let summary = ollama::ask(&prompt);
    println!("\n🧠 >\n{}", summary);
}

fn cluster_mode(config_file_path: &str, k: usize) {
    println!("Clustering ideas");

    let config = config::load(config_file_path);
    let embedded_chunks = chunking::load(&config);

    // K-means lite: randomly pick k initial centers
    let mut rng = thread_rng();
    let mut centroids: Vec<Vec<f32>> = embedded_chunks
        .choose_multiple(&mut rng, k)
        .map(|c| c.embedding.clone())
        .collect();

    let mut assignments: Vec<usize> = vec![0; embedded_chunks.len()];

    // Iterative refinement (just a few steps)
    for _ in 0..5 {
        // Assign
        for (i, chunk) in embedded_chunks.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, cosine_similarity(&chunk.embedding, c).unwrap_or(-1.0)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap_or(0);
            assignments[i] = best;
        }

        // Recompute centroids
        for i in 0..k {
            let members: Vec<&Vec<f32>> = embedded_chunks
                .iter()
                .zip(&assignments)
                .filter(|(_, &a)| a == i)
                .map(|(c, _)| &c.embedding)
                .collect();

            if members.is_empty() {
                continue;
            }

            let mut new_centroid = vec![0.0; members[0].len()];
            for vec in &members {
                for (i, val) in vec.iter().enumerate() {
                    new_centroid[i] += val;
                }
            }
            for val in &mut new_centroid {
                *val /= members.len() as f32;
            }

            centroids[i] = new_centroid;
        }
    }

    // Group by cluster
    let mut clusters: Vec<Vec<&Chunk>> = vec![vec![]; k];
    for (chunk, &cluster_idx) in embedded_chunks.iter().zip(&assignments) {
        clusters[cluster_idx].push(chunk);
    }

    println!("\nClustered into {} groups:\n", k);

    for (i, group) in clusters.iter().enumerate() {
        let sample = group
            .iter()
            .take(5)
            .map(|c| format!("- {}", c.text))
            .collect::<Vec<_>>()
            .join("\n");

        let label_prompt = format!(
            "Here are some notes:\n\n{}\n\nWhat common theme or topic do they share? Respond with just a short label.",
            sample
        );

        let label = ollama::ask(&label_prompt);

        println!(
            "Cluster {} - {}\n{} items",
            i + 1,
            label.trim(),
            group.len()
        );
        for chunk in group.iter().take(5) {
            println!("•  {}", chunk.file_path);
        }
        println!();
    }
}

fn chat_mode(config_path: &str) {
    println!("Chat mode started. Ask your brain anything. Ctrl+C to quit.\n");

    let config = config::load(config_path);
    let embedded_chunks = chunking::load(&config);

    loop {
        print!("\nYou: ");
        io::stdout().flush().unwrap();
        let mut question = String::new();
        if io::stdin().read_line(&mut question).is_err() {
            break;
        }
        let question = question.trim();
        if question.is_empty() {
            continue;
        }

        let q_embedding = ollama::embed_text(question);
        if q_embedding.is_none() {
            println!("Could not embed question.");
            continue;
        }
        let q_embedding = q_embedding.unwrap();

        let top_chunks = find_similar_chunks(&embedded_chunks, &q_embedding, 8);
        let context = top_chunks
            .iter()
            .map(|c| format!("• {}", c.text))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Answer the following question based on my notes below.\n\
        If not enough info is present, say so.\n\n\
        Notes:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, question
        );

        let response = ollama::ask(&prompt);
        println!("\n🧠 {}", response.trim());
    }
}
