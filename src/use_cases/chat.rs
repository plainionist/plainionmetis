use std::io::{self, Write};

use crate::utils::{chunking, config::Config, ollama, similarity};

pub fn run(config: &Config) {
    println!("Chat started - Ctrl+C to quit\n");

    let chunks = chunking::load(&config);

    loop {
        let question = match read_user_input() {
            Some(q) => q,
            None => continue,
        };

        let response = ask(&chunks, &question);
        println!("\n🧠 >\n{}", response);
    }
}

fn read_user_input() -> Option<String> {
    print!("\nYou: ");
    io::stdout().flush().ok()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input).ok()?;
    let trimmed = input.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn ask(chunks: &Vec<chunking::Chunk>, question: &str) -> String {
    let question_embedding = ollama::embed_text(question).expect("Failed to embed question");

    let top_chunks = similarity::find_similar_chunks(chunks, &question_embedding, 10);

    let context = top_chunks
        .iter()
        .map(|c| format!("- {}", c.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "Answer the following question based on my notes below.\n\
            If not enough info is present, say so.\n\n\
            Notes:\n{}\n\nQuestion: {}\n\nAnswer:",
        context, question
    );

    ollama::ask(&prompt)
}
