use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use walkdir::WalkDir;
use crate::utils::{config::Config, ollama};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub embedding: Vec<f32>,
    pub text: String,
    pub file_path: String,
}

type Cache = HashMap<String, Chunk>;

pub fn load(config: &Config) -> Vec<Chunk> {
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
