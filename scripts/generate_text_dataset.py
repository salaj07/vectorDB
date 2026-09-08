"""
VectorForge Script — Generate 5,000 Text Statement Dataset

Generates 5,000 diverse short text statements across Web Dev, AI/ML, Vector Search,
Emotions, Databases, and DevOps, embeds them into 128-dimensional vectors, and
computes exact ground truth for 500 query statements.
"""

import os
import json
import hashlib
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, TOP_K
from app.core.distance import normalize
from app.evaluation.ground_truth import generate_ground_truth
from app.indexes.manager import IndexManager


# Base templates per topic to generate 5,000 diverse, realistic text statements
TOPIC_TEMPLATES = {
    "Web Frameworks & Backend": [
        "Express.js framework simplifies HTTP routing, middleware handling, and REST API development in Node.js.",
        "Node.js uses an event-driven non-blocking I/O model for fast network applications.",
        "React.js component-based architecture enables dynamic front-end state management.",
        "JavaScript ES6 features like async/await, promises, and arrow functions streamline web code.",
        "TypeScript adds static type definitions to JavaScript for enterprise application safety.",
        "FastAPI leverages Python type hints for fast execution and automated OpenAPI documentation.",
        "Django framework provides an Object-Relational Mapper (ORM) and built-in administration interface.",
        "REST APIs use standard HTTP verbs GET, POST, PUT, and DELETE for stateless JSON exchange.",
        "GraphQL enables client applications to request exact data fields in a single HTTP payload.",
        "Next.js framework provides server-side rendering and static page generation for React applications.",
        "WebSockets provide full-duplex bi-directional communication channels over a TCP connection.",
        "JWT authentication tokens securely transmit user identity claims between client and server.",
    ],

    "Artificial Intelligence & Machine Learning": [
        "Deep neural networks learn hierarchical feature representations through multi-layer architectures.",
        "Convolutional neural networks extract spatial feature maps for image classification tasks.",
        "Recurrent neural networks and LSTMs process sequential time-series and natural language tokens.",
        "Transformer models utilize self-attention mechanisms for language understanding and translation.",
        "Large language models generate human-like text responses based on prompt context vectors.",
        "Supervised learning trains classification models on labeled input-output dataset pairs.",
        "Unsupervised clustering algorithms discover latent groupings in unannotated multi-dimensional data.",
        "Reinforcement learning agents optimize cumulative rewards through environment state exploration.",
        "Gradient descent optimization adjusts neural network weights to minimize loss functions.",
        "Hyperparameter tuning improves model validation accuracy and prevents overfitting on training data.",
        "PyTorch and TensorFlow deep learning libraries offer GPU acceleration for tensor computations.",
        "Natural language processing pipelines tokenize text, extract lemmas, and compute embeddings.",
    ],

    "Vector Search & Databases": [
        "Vector databases index high-dimensional embeddings for fast nearest neighbor similarity search.",
        "Hierarchical Navigable Small World graphs provide sub-logarithmic ANN query performance.",
        "Inverted File Flat indexes cluster vector spaces into Voronoi cells to prune search candidates.",
        "Cosine similarity measures the angular distance between normalized vectors in embedding space.",
        "Euclidean distance computes straight-line spatial distance between points in multi-dimensional space.",
        "Dot product calculation on normalized vectors equals cosine similarity score.",
        "Approximate nearest neighbor search trades marginal recall accuracy for dramatic query speedup.",
        "Locality Sensitive Hashing maps similar high-dimensional vectors into identical hash buckets.",
        "Product quantization compresses high-dimensional floating-point vectors into short codebook keys.",
        "Index rebuilding updates vector graph edges and cluster centroids after batch ingestion.",
    ],

    "Emotions & Psychology": [
        "Emotional intelligence enables individuals to manage interpersonal stress and demonstrate empathy.",
        "Human emotions like joy, sadness, fear, and surprise guide adaptive social behavior.",
        "Cognitive behavioral therapy helps reframe negative thought patterns and regulate mood states.",
        "Mindfulness meditation practices reduce psychological stress and enhance emotional resilience.",
        "Empathy allows individuals to understand and share the feelings and perspectives of others.",
        "Motivation and intrinsic drive push individuals toward personal growth and goal attainment.",
        "Positive psychology focuses on human strengths, happiness, and psychological well-being.",
        "Emotional regulation strategies help maintain calmness under high-pressure environments.",
    ],

    "Databases & Cloud DevOps": [
        "Relational databases enforce ACID transactions and relational integrity using SQL queries.",
        "PostgreSQL database provides advanced JSON querying, index types, and relational features.",
        "NoSQL document databases like MongoDB store unstructured JSON documents with flexible schemas.",
        "Redis in-memory data structure store serves as a high-speed caching and messaging broker.",
        "Docker containers package application microservices with isolated dependencies and environments.",
        "Kubernetes cluster orchestration automates container deployment, scaling, and self-healing.",
        "Infrastructure as Code tools like Terraform automate cloud server provisioning and networking.",
        "Continuous Integration and Deployment pipelines automate software testing and artifact deployment.",
        "Git version control manages code branch merging, pull requests, and commit history.",
        "Linux operating system provides bash scripting, process management, and permissions control.",
        "Microservices architecture decouples complex systems into small, independently scalable units.",
        "Cybersecurity firewalls and encryption protocols safeguard data in transit and at rest.",
    ]
}

# Modifiers to expand sentences into 5,000 unique, distinct text statements
MODIFIERS = [
    "In modern software development,", "Engineering teams report that",
    "Technical documentation highlights how", "Best practices indicate that",
    "Computer science theory proves that", "Production systems demonstrate that",
    "Industry research confirms that", "System architects recommend that",
    "Core specifications state that", "Practical applications show that"
]


import re


def text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    """Embed text statement into normalized D-dimensional float32 vector."""
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean_text.split()
    vec = np.zeros(dim, dtype=np.float32)

    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += sign

    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += 1.5 * sign

    return normalize(vec)


def generate_5k_text_dataset(total_count: int = 5000, num_queries: int = 500, output_dir: str = DATA_DIR):
    print(f"Generating {total_count} text statement dataset across all topics...")

    all_sentences = []
    metadata_map = {}
    categories = list(TOPIC_TEMPLATES.keys())

    idx = 0
    mod_len = len(MODIFIERS)

    # Loop through base templates and modifiers to generate 5,000 unique statements
    while len(all_sentences) < total_count:
        cat = categories[idx % len(categories)]
        templates = TOPIC_TEMPLATES[cat]
        base_template = templates[(idx // len(categories)) % len(templates)]
        modifier = MODIFIERS[(idx // (len(categories) * len(templates))) % mod_len]

        if idx < len(categories) * len(templates):
            statement = base_template
        else:
            statement = f"{modifier} {base_template.lower()}"

        vid = f"text_{len(all_sentences)}"
        all_sentences.append({
            "id": vid,
            "text": statement,
            "category": cat
        })
        metadata_map[vid] = {"text": statement, "category": cat}
        idx += 1

    print(f"Generated {len(all_sentences)} text statements.")

    # Vectorize all 5,000 text statements
    print("Vectorizing 5,000 text statements into 128-dimensional float32 vectors...")
    vectors = np.array([text_to_vector(s["text"], dim=128) for s in all_sentences], dtype=np.float32)
    ids_arr = np.array([s["id"] for s in all_sentences], dtype=object)

    # Generate 500 query statements across topics
    print(f"Generating {num_queries} query statements for ground-truth calculation...")
    query_texts = []
    for q_idx in range(num_queries):
        cat = categories[q_idx % len(categories)]
        base = TOPIC_TEMPLATES[cat][q_idx % len(TOPIC_TEMPLATES[cat])]
        words = base.split()
        # Create query from subset of statement words
        q_words = words[:min(5, len(words))]
        query_texts.append(" ".join(q_words))

    query_vectors = np.array([text_to_vector(q, dim=128) for q in query_texts], dtype=np.float32)

    # Save files to output_dir
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "vectors.npy"), vectors)
    np.save(os.path.join(output_dir, "ids.npy"), ids_arr)
    np.save(os.path.join(output_dir, "queries.npy"), query_vectors)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_map, f, indent=2)

    with open(os.path.join(output_dir, "text_corpus.json"), "w") as f:
        json.dump(all_sentences, f, indent=2)

    print(f"Saved 5,000 text statement dataset to '{output_dir}/':")
    print(f"  vectors.npy    : {vectors.shape}")
    print(f"  ids.npy        : {ids_arr.shape}")
    print(f"  queries.npy    : {query_vectors.shape}")
    print(f"  text_corpus.json: {len(all_sentences)} text statements")

    # Compute ground truth for queries
    print("Computing exact ground truth (k=10) using VectorStore...")
    manager = IndexManager(dimension=128)
    manager.store.load(output_dir)
    gt_list = generate_ground_truth(manager.store, query_vectors, k=TOP_K)

    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(gt_list, f, indent=2)

    print(f"Saved ground truth for {len(gt_list)} query statements to '{gt_path}'.")


if __name__ == "__main__":
    generate_5k_text_dataset()
