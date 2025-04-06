use std::fs;

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    pub config: InnerConfig,
}

#[derive(Debug, serde::Deserialize)]
pub struct InnerConfig {
    pub cache_file: String,
    pub content_paths: Vec<String>,
}

pub fn load_config(path: &str) -> Config {
    let contents = fs::read_to_string(path).expect("Failed to read config file");
    toml::from_str(&contents).expect("Failed to parse config file")
}
