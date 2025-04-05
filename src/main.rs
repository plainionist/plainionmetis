use reqwest::blocking::Client;
use serde_json::json;
use std::{env, fs};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
struct Chunk {
    file_path: String,
    text: String,
    embedding: Vec<f32>,
}

// prepare:
// - "ollama pull phi3:mini"
// - "ollama pull nomic-embed-text"
// then run "ollama run phi3:mini"
// then run this program

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: plainionmetis <markdown-folder> <idea>");
        std::process::exit(1);
    }

    let notes_dir = &args[1];
    let idea = &args[2..].join(" "); // remaining args build the idea

    let chunks = collect_markdown_chunks(notes_dir, 400);
    println!("Loaded chunks: {}", chunks.len());

    let embedded_chunks: Vec<Chunk> = chunks
        .into_iter()
        .filter_map(|(text, file_path)| {
            embed_text(&text).map(|embedding| Chunk {
                text,
                embedding,
                file_path,
            })
        })
        .collect();

    println!("🧠 Embedded chunks: {}", embedded_chunks.len());

    let query_embedding = embed_text(idea).expect("Failed to embed idea text");

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

    let result = ask_ollama(&prompt);
    println!("\n🧠 >\n{}", result);
}

fn collect_markdown_chunks(base_path: &str, max_words: usize) -> Vec<(String, String)> {
    let mut chunks = vec![];

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

    chunks
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

fn embed_text(text: &str) -> Option<Vec<f32>> {
    let client = Client::new();
    let response = client
        .post("http://localhost:11434/api/embeddings")
        .json(&json!({
            "model": "nomic-embed-text",
            "prompt": text
        }))
        .send()
        .ok()?;

    let json: serde_json::Value = response.json().ok()?;
    json["embedding"]
        .as_array()?
        .iter()
        .map(|v| v.as_f64().map(|f| f as f32))
        .collect()
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

fn ask_ollama(prompt: &str) -> String {
    let client = Client::new();

    let response = client
        .post("http://localhost:11434/api/generate")
        .json(&json!({
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": false
        }))
        .send()
        .expect("Failed to send request to Ollama");

    let json: serde_json::Value = response.json().expect("Failed to parse response");
    json["response"]
        .as_str()
        .unwrap_or("[No response]")
        .to_string()
}
