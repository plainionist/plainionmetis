use reqwest::blocking::Client;
use serde_json::json;

pub fn embed_text(text: &str) -> Option<Vec<f32>> {
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

pub fn ask(prompt: &str) -> String {
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
