use crate::utils::chunking::Chunk;

pub fn find_similar_chunks(chunks: &[Chunk], query: &[f32], top_n: usize) -> Vec<Chunk> {
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

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    Some(dot / (norm_a * norm_b + 1e-8))
}
