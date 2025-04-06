use crate::utils::{chunking, config::Config, ollama, similarity};

pub fn run(config: &Config, idea: &str) {
    let embedded_chunks = chunking::load(&config);

    let query_embedding = ollama::embed_text(idea).expect("Failed to embed idea text");

    let top_chunks = similarity::find_similar_chunks(&embedded_chunks, &query_embedding, 5);

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

    let response = ollama::ask(&prompt);
    println!("\n🧠 >\n{}", response);
}
