use rand::{seq::SliceRandom, thread_rng};

use crate::utils::{chunking, chunking::Chunk, config::Config, ollama, similarity};

pub fn run(config: &Config, k: usize) {
    println!("Clustering ideas");

    let embedded_chunks = chunking::load(&config);

    // K-means lite: randomly pick k initial centers
    let mut rng = thread_rng();
    let mut centroids: Vec<Vec<f32>> = embedded_chunks
        .choose_multiple(&mut rng, k)
        .map(|c| c.embedding.clone())
        .collect();

    let mut assignments: Vec<usize> = vec![0; embedded_chunks.len()];

    // Iterative refinement (just a few steps)
    for _ in 0..5 {
        // Assign
        for (i, chunk) in embedded_chunks.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    (
                        j,
                        similarity::cosine_similarity(&chunk.embedding, c).unwrap_or(-1.0),
                    )
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap_or(0);
            assignments[i] = best;
        }

        // Recompute centroids
        for i in 0..k {
            let members: Vec<&Vec<f32>> = embedded_chunks
                .iter()
                .zip(&assignments)
                .filter(|(_, &a)| a == i)
                .map(|(c, _)| &c.embedding)
                .collect();

            if members.is_empty() {
                continue;
            }

            let mut new_centroid = vec![0.0; members[0].len()];
            for vec in &members {
                for (i, val) in vec.iter().enumerate() {
                    new_centroid[i] += val;
                }
            }
            for val in &mut new_centroid {
                *val /= members.len() as f32;
            }

            centroids[i] = new_centroid;
        }
    }

    // Group by cluster
    let mut clusters: Vec<Vec<&Chunk>> = vec![vec![]; k];
    for (chunk, &cluster_idx) in embedded_chunks.iter().zip(&assignments) {
        clusters[cluster_idx].push(chunk);
    }

    println!("\nClustered into {} groups:\n", k);

    for (i, group) in clusters.iter().enumerate() {
        let sample = group
            .iter()
            .take(5)
            .map(|c| format!("- {}", c.text))
            .collect::<Vec<_>>()
            .join("\n");

        let label_prompt = format!(
            "Here are some notes:\n\n{}\n\nWhat common theme or topic do they share? Respond with just a short label.",
            sample
        );

        let label = ollama::ask(&label_prompt);

        println!(
            "Cluster {} - {}\n{} items",
            i + 1,
            label.trim(),
            group.len()
        );
        for chunk in group.iter().take(5) {
            println!("•  {}", chunk.file_path);
        }
        println!();
    }
}
