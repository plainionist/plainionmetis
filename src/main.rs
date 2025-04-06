use rand::{seq::SliceRandom, thread_rng};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use std::{env, fs};
use walkdir::WalkDir;

mod config;
use config::{load_config, Config};

mod ollama;

// prepare:
// - "ollama pull phi3:mini"
// - "ollama pull nomic-embed-text"
// then run "ollama run phi3:mini"
// then run this program

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Chunk {
    embedding: Vec<f32>,
    text: String,
    file_path: String,
}

type Cache = HashMap<String, Chunk>;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage:\n\
            plainionmetis <notes-folder> <idea>\n\
            plainionmetis explore <notes-folder> <topic>"
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
    let config = load_config(config_file_path); // use config file path now

    let embedded_chunks = load_embedded_chunks(&config);

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

fn hash_chunk(text: &str, file_path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update(file_path.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn collect_markdown_chunks(paths: &[String], max_words: usize) -> Vec<(String, String)> {
    let mut chunks = vec![];

    for base_path in paths {
        for entry in WalkDir::new(base_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map(|ext| ext == "md").unwrap_or(false))
        {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                let chunked = chunk_text(&content, max_words);
                let path_str = entry.path().display().to_string();
                for chunk in chunked {
                    chunks.push((chunk, path_str.clone()));
                }
            }
        }
    }

    chunks
}

fn load_cache(path: &Path) -> Cache {
    fs::read_to_string(path)
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
        .unwrap_or_default()
}

fn save_cache(path: &Path, cache: &Cache) {
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = fs::write(path, json);
    }
}

fn chunk_text(text: &str, max_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = usize::min(start + max_words, words.len());
        chunks.push(words[start..end].join(" "));
        start = end;
    }

    chunks
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

    let config = load_config(config_file_path);

    let embedded_chunks = load_embedded_chunks(&config);

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

    let config = load_config(config_file_path);
    let embedded_chunks = load_embedded_chunks(&config);

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

    let config = load_config(config_path);
    let embedded_chunks = load_embedded_chunks(&config);

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

fn load_embedded_chunks(config: &Config) -> Vec<Chunk> {
    let cache_path = Path::new(&config.config.cache_file);
    let content_paths = &config.config.content_paths;

    let raw_chunks = collect_markdown_chunks(content_paths, 400);

    println!("Loaded chunks: {}", raw_chunks.len());

    let mut cache: HashMap<String, Chunk> = if cache_path.exists() {
        load_cache(&cache_path)
    } else {
        HashMap::new()
    };

    let mut embedded_chunks = vec![];

    for (text, file_path) in raw_chunks {
        let hash = hash_chunk(&text, &file_path);

        if let Some(cached) = cache.get(&hash) {
            embedded_chunks.push(cached.clone());
        } else if let Some(embedding) = ollama::embed_text(&text) {
            let chunk = Chunk {
                text: text.clone(),
                embedding: embedding.clone(),
                file_path: file_path.clone(),
            };

            cache.insert(hash, chunk.clone());
            embedded_chunks.push(chunk);
        }
    }

    save_cache(&cache_path, &cache);

    println!("Embedded chunks: {}", embedded_chunks.len());

    embedded_chunks
}
