use std::io::{self, Write};

use crate::utils::{chunking, config::Config, ollama, similarity};

pub fn run(config: &Config) {
    println!("Chat started - Ctrl+C to quit\n");

    let chunks = chunking::load(&config);

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

        let top_chunks = similarity::find_similar_chunks(&chunks, &q_embedding, 8);
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
        println!("\n🧠 >\n{}", response);
    }
}
