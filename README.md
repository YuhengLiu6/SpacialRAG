# Spatial RAG

Spatial RAG is a **Retrieval-Augmented Generation** system designed for embodied agents. It allows an agent to explore a 3D environment, build a semantic memory of what it sees, and retrieve specific objects or locations based on natural language queries (e.g., "Where did I see a potted plant?").

## Design Strategy

The core of this project is a **Hybrid Retrieval System** that combines global semantic understanding with precise object localization.

### 1. Hybrid Perception (CLIP + YOLO)
Standard RAG systems often rely solely on CLIP embeddings of entire images. However, in spatial environments, a user might ask about a small object (e.g., "keys on the table") that is lost in the global image embedding.

Our approach uses a dual-branch perception module:
*   **Global Branch (CLIP)**: Encodes the entire view into a semantic vector. This captures the "gist" of the scene (e.g., "a living room," "a kitchen").
*   **Local Branch (YOLO)**: Detects specific objects (e.g., "chair," "plant," "laptop") and records their bounding boxes and confidence scores.

### 2. Spatial Memory (FAISS + Metadata)
We use a vector database to store the agent's experiences. Each memory entry consists of:
*   **Vector**: The CLIP embedding of the view.
*   **Metadata**:
    *   `image_path`: Path to the saved ego-centric view.
    *   `position`: The (x, y, z) coordinates of the agent.
    *   `detections`: A list of objects detected in that view by YOLO.

### 3. Retrieval Logic
When a user asks a question (e.g., "Find the red sofa"), the system performs a two-stage scoring process:

1.  **Semantic Search**: We embed the text query using CLIP and find the top-k most similar images from memory.
2.  **Object Re-ranking**: We analyze the YOLO detections associated with those top images. If the query mentions an object (e.g., "sofa") and YOLO also detected a "couch" or "sofa" in that image, we boost the score.

The final score is a weighted combination:
$$ Score = w_{clip} \cdot S_{semantic} + w_{yolo} \cdot S_{detection} $$

This ensures that we return images that not only *look* like the answer but arguably *contain* the specific object requested.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YuhengLiu6/SpacialRAG.git
    cd SpacialRAG
    ```

2.  **Create a Conda environment**:
    ```bash
    conda create -n spatial_rag python=3.10
    conda activate spatial_rag
    ```

3.  **Install Dependencies**:
    ```bash
    pip install torch torchvision torchaudio
    pip install git+https://github.com/openai/CLIP.git
    pip install ultralytics faiss-cpu numpy opencv-python matplotlib
    ```

    *Note: For Mac users with Apple Silicon, `faiss-cpu` is recommended over `faiss-gpu`.*

4.  **Download Habitat Scene**:
    Ensure you have the `habitat-test-scenes` data. This project is configured to use `apartment_1.glb`.

## Usage

### 1. Run the Main Simulation
The main script starts the agent, explores the environment to build memory, and then accepts user queries.

```bash
python -m spatial_rag.main
```

*   **Exploration Phase**: The agent will move randomly for `NUM_STEPS` (defined in `config.py`), capturing images and detecting objects.
*   **Query Phase**: After exploration, you can type queries like "Where is the bed?" or "Show me the kitchen."

### 2. Result Visualization
Retrieval results are saved in the `retrieval_results/` directory. Each result includes:
*   The original image seen by the agent.
*   Bounding boxes around detected objects.
*   Confidence scores and ranking.

### 3. Memory Management
*   **Inspect Memory**: View stats about the stored vectors and detections.
    ```bash
    python -m spatial_rag.inspect_memory
    ```
*   **Reset Memory**: Clear all stored data to start fresh.
    ```bash
    python -m spatial_rag.reset_memory
    ```

## Configuration
All hyperparameters are centralled in `spatial_rag/config.py`:
*   `SCENE_PATH`: Path to the .glb scene file.
*   `YOLO_MODEL_PATH`: `yolov8n.pt`, `yolov8m.pt`, etc.
*   `CLIP_MODEL_NAME`: `ViT-B/32`, `ViT-L/14`, etc.
*   `TOP_K`: Number of retrieval results to return.