use reqwest::blocking::Client;
use serde_json::json;
use std::{env, fs};
use walkdir::WalkDir;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: plainionmetis <markdown-folder>");
        std::process::exit(1);
    }

    let notes_dir = &args[1];

    let notes = collect_markdown_files(notes_dir);

    println!("Loaded markdown files: {}", notes.len());

    let question = "Give me a summary";
    let context = notes.join("\n\n---\n\n");

    println!("Total word count: {}", context.split_whitespace().count());

    let prompt = format!(
        "Using the following notes:\n{}\n\nAnswer this:\n{}",
        context, question
    );

    let response = ask_ollama(&prompt);

    println!("\n🧠 >\n{}", response);
}

fn collect_markdown_files(base_path: &str) -> Vec<String> {
    WalkDir::new(base_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map(|ext| ext == "md").unwrap_or(false))
        .filter_map(|entry| fs::read_to_string(entry.path()).ok())
        .collect()
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
