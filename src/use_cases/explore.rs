
use crate::utils::{chunking, config, ollama, similarity};

pub fn run(config_file_path: &str, topic: &str) {
    println!("Exploring topic: '{}'", topic);

    let config = config::load(config_file_path);

    let embedded_chunks = chunking::load(&config);

    let topic_embedding = ollama::embed_text(topic).expect("Failed to embed topic");

    let top_chunks = similarity::find_similar_chunks(&embedded_chunks, &topic_embedding, 10);
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

    let response = ollama::ask(&prompt);
    println!("\n🧠 >\n{}", response);
}
