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

## Strategy Iteration & Experiments

Our approach has evolved through several phases to address limitations in standard retrieval methods:

### Phase 1: Baseline (CLIP Only)
*   **Approach**: We started by simply embedding every ego-centric image with CLIP (`ViT-B-16`) and performing cosine similarity search against the text query.
*   **Limitation**: CLIP is excellent at global scene understanding (e.g., "a kitchen") but struggles with small, specific objects (e.g., "keys on the table") or spatial prepositions. It often retrieves images that have the right *vibe* but miss the target object.

### Phase 2: Object-Aware Retrieval (CLIP + YOLO)
*   **Improvement**: We integrated `YOLOv8` (medium model) to detect visible objects in every frame.
*   **Logic**: If the user asks for a "chair", we explicitly boost the score of images where YOLO detected a `chair`.
*   **Result**: Recall for specific common objects improved significantly. However, vague queries like "Where can I sleep?" failed because YOLO doesn't detect "sleeping places," only "beds" or "couches".

### Phase 3: Semantic Expansion & Hybrid Scoring (Current)
*   **Query Expansion**: We now use an LLM (`gpt-4o-mini`) to translate vague user intentions into specific visual targets.
    *   *User*: "Where can I sleep?" -> *LLM*: "bed, couch, sofa"
*   **Hybrid Scoring**: We tuned a weighted scoring formula to balance semantic similarity, object detection confidence, and keyword matching.
    *   `Score = 0.55 * CLIP + 0.15 * YOLO + 0.3 * BM25`
*   **Benefit**: This handles specific object queries with high precision (YOLO/BM25) while maintaining the ability to find "similar looking" scenes (CLIP) when detections fail.

### Phase 4: Future Experiments
*   **Open-Vocabulary Detection**: We are experimenting with **GroundingDINO** to detect arbitrary objects not in the COCO dataset (e.g., "red pill", "specific painting").
*   **Diverse Environments**: Testing generalization across different Habitat scenes (e.g., `apartment_1`, `van-gogh-room`).

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
    pip install ultralytics faiss-cpu numpy opencv-python matplotlib rank_bm25 openai
    ```


    *Note: For Mac users with Apple Silicon, `faiss-cpu` is recommended over `faiss-gpu`.*

    **For V100 (Linux/CUDA) Users:**
    If running on a remote server with a V100 GPU:
    1.  Ensure you have CUDA installed (e.g., CUDA 11.8).
    2.  Install `habitat-sim` with headless support:
        ```bash
        conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
        ```
    3.  Install dependencies using the updated requirements:
        ```bash
        pip install -r requirements.txt
        ```
    4.  The code automatically detects CUDA (`torch.cuda.is_available()`) and will run on the GPU.

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