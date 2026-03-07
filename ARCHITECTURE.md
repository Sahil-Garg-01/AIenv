# GaiaGuard System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard (Port 8501)  │   FastAPI REST API (Port 8000) │
│  - Image Upload                    │   - /train endpoint            │
│  - Result Visualization            │   - /predict endpoint          │
│  - Report Display                  │   - /docs (Swagger UI)        │
└──────────────┬──────────────────────┬────────────────────────────┘
               │                      │
               │                      │
        ┌──────▼──────────────────────▼─────────────┐
        │      Application Logic Layer              │
        ├──────────────────────────────────────────┤
        │  Main App (app/main.py)                  │
        │  - Routes HTTP requests                 │
        │  - Orchestrates training & prediction  │
        │  - Manages logging                      │
        └──────────┬────────────┬──────────────────┘
                   │            │
        ┌──────────▼──┐  ┌──────▼────────────┐
        │ML Pipeline  │  │AI Agent Pipeline  │
        │(Training &  │  │(Report Generation)│
        │Prediction)  │  │                    │
        └──────┬──────┘  └──────┬─────────────┘
               │                │
        ┌──────▼─────────────────▼──────────┐
        │   Processing & Configuration      │
        ├───────────────────────────────────┤
        │  - Model (ResNet50)               │
        │  - Preprocessing (Image)          │
        │  - LangGraph Agent Workflow       │
        │  - Google Gemini Integration      │
        │  - Config Management             │
        └──────┬──────────────────┬────────┘
               │                  │
        ┌──────▼────────┐  ┌──────▼──────────┐
        │   Data Layer  │  │ External Services│
        ├───────────────┤  ├──────────────────┤
        │ - Training    │  │ - Google Gemini  │
        │   Dataset     │  │   API            │
        │ - Model       │  │                  │
        │   Weights     │  │                  │
        └───────────────┘  └──────────────────┘
```

---

## Component Overview

### 1. **User Interface Layer**

#### Streamlit Dashboard (`dashboard/streamlit_app.py`)
- Web-based interface for non-technical users
- Features:
  - Image file upload (PNG/JPG/JPEG)
  - Image preview
  - Analysis trigger button
  - Hazard classification display
  - Incident report rendering

#### FastAPI REST API (`app/main.py`)
- RESTful API for programmatic access
- Endpoints:
  - `POST /train` - Initiate model training
  - `POST /predict` - Classify image and generate report
- Features:
  - Async request handling
  - Request validation
  - Automatic documentation (Swagger UI at `/docs`)
  - Comprehensive logging

---

### 2. **Application Logic Layer** (`app/main.py`)

```
Request Flow:
  Dashboard/API Client
         │
         ▼
   FastAPI Router
         │
         ├─→ /train endpoint
         │      │
         │      ▼
         │   train_model()
         │      │
         │      ▼
         │   Model Training Pipeline
         │
         └─→ /predict endpoint
                │
                ▼
            predict()
                │
                ▼
            Classification
                │
                ▼
            build_graph()
                │
                ▼
            generate_report()
                │
                ▼
            Response with Hazard + Report
```

**Key Functions:**
- `train()` - Trigger training, log progress
- `predict_image(file)` - Load image, predict, generate report
- Logging at all stages for monitoring

---

## Low-Level Architecture

### 3. **ML Pipeline Layer**

#### Model Architecture (`models/model.py`)

```
ResNet50 Base Model (Pretrained on ImageNet)
    │
    ├─ Layer 1-3: Frozen (Weights Not Updated)
    │   └─ ImageNet features: edges, textures, basic shapes
    │
    ├─ Layer 4: Fine-tuned (Trainable)
    │   └─ Adapt to environmental hazards
    │
    └─ Classification Head: New FC Layer
       └─ Input: 2048 features from ResNet
       └─ Output: 4 classes (softmax logits)

Classes:
  - deforestation
  - oil_spill
  - wildfire
  - normal
```

**Transfer Learning Strategy:**
```python
for name, param in model.named_parameters():
    if "layer4" not in name:
        param.requires_grad = False  # Freeze layers 1-3
    # Only layer4 and fc are trainable
```

#### Training Pipeline (`models/train.py`)

```
Step 1: Data Preparation
  merged_dataset/
      ├─ deforestation/  ──┐
      ├─ oil_spill/      ──┼─→ ImageFolder Loader
      ├─ wildfire/       ──┤
      └─ normal/        ──┘
            │
            ▼
  Random Split: 80% Train | 20% Validation
            │
            ├─ Train Dataset
            │    └─ Augmentation Applied
            │        ├─ RandomResizedCrop(224)
            │        ├─ RandomHorizontalFlip()
            │        ├─ RandomVerticalFlip()
            │        ├─ RandomRotation(20°)
            │        └─ ColorJitter(brightness, contrast, saturation)
            │
            └─ Validation Dataset
                 └─ No Augmentation
                    └─ Resize(224x224)

Step 2: Model Initialization
  model = get_model(num_classes=4)
      └─ Load ResNet50 pretrained weights
      └─ Freeze layers 1-3
      └─ Replace FC layer: 2048 → 4
      └─ Move to DEVICE (CUDA or CPU)

Step 3: Training Loop (20 Epochs)
  For each epoch:
    ├─ Training Phase
    │   For each batch:
    │     ├─ images, labels = batch
    │     ├─ outputs = model(images)
    │     ├─ loss = CrossEntropyLoss(outputs, labels)
    │     ├─ optimizer.zero_grad()
    │     ├─ loss.backward()
    │     └─ optimizer.step()
    │
    └─ Validation Phase
        For each batch:
          ├─ images, labels = batch
          ├─ outputs = model(images)
          ├─ loss = CrossEntropyLoss(outputs, labels)
          └─ Running loss accumulation

Step 4: Learning Rate Scheduling
  ReduceLROnPlateau Scheduler
    ├─ Monitor: Validation Loss
    ├─ Patience: 3 epochs
    ├─ Factor: 0.3 (multiply LR by 0.3 if val_loss doesn't improve)

Step 5: Model Checkpointing
  If val_loss < best_val_loss:
    └─ Save model weights to models/gaia_guard_best_model.pt

Step 6: Visualization
  Plot and save training_curve.png
    ├─ X-axis: Epoch
    ├─ Y-axis: Loss
    └─ Lines: train_loss, val_loss
```

#### Inference Pipeline (`models/predict.py`)

```
Input: Image File (from API or User)
  │
  ▼
Load Model
  ├─ Check if model weights exist
  ├─ Load from models/gaia_guard_best_model.pt
  ├─ Set to eval mode
  └─ Move to DEVICE

Preprocess Image (utils/preprocess.py)
  ├─ Read file from buffer
  ├─ Convert to RGB (PIL Image)
  ├─ Apply val_transform:
  │  ├─ Resize to 224x224
  │  ├─ Normalize (ImageNet stats)
  │  └─ Convert to Tensor
  └─ Unsqueeze(0) for batch dimension
      └─ Shape: (1, 3, 224, 224)

Forward Pass
  Input Tensor (1, 3, 224, 224)
      │
      ▼
  ResNet50 Feature Extractor
      │
      ├─ Layer 1-4: Feature maps
      │
      └─ GlobalAvgPool → (1, 2048)
            │
            ▼
        FC Layer: (1, 2048) → (1, 4)
            │
            ▼
        Softmax (implicit in CrossEntropyLoss)
            │
            ▼
        Logits: [p0, p1, p2, p3]
            │
  Classes:  [0=deforestation, 1=oil_spill, 2=wildfire, 3=normal]

Prediction
  torch.max(outputs, 1) → argmax
      │
      ▼
  Predicted Class Index
      │
      ▼
  predicted_class = CLASSES[pred.item()]
      │
      ▼
  return "deforestation" | "oil_spill" | "wildfire" | "normal"
```

---

### 4. **AI Agent Layer** (Report Generation)

#### LangGraph Workflow (`agent/graph.py`)

```
StateGraph Definition:
  ├─ AgentState (TypedDict)
  │  ├─ hazard: str (from model prediction)
  │  └─ report: str (generated by Gemini)
  │
  ├─ Add Node: "report_node" → generate_report function
  │
  ├─ Set Entry Point: "report_node"
  │
  └─ Set Finish Point: "report_node"
         │
         ▼
    Compiled Graph (Single-node graph)
         │
         ▼
    graph.invoke(state) → Executes report_node
```

#### Report Generation (`agent/nodes.py`)

```
Input State:
  {
    "hazard": "wildfire",
    "report": ""
  }

Process:
  1. Extract hazard from state
     → hazard = "wildfire"

  2. Create Prompt
     → Template with hazard embedded
     → Instructions for Gemini
        ├─ Act as environmental analyst
        ├─ Generate structured report
        │   ├─ Incident Summary (overview, scale, implications)
        │   ├─ Ecological Impact Assessment (biodiversity, ecosystems, health)
        │   ├─ Recommended Actions (immediate, medium-term, long-term)
        │   └─ Additional Insights (further investigation, contacts)
        └─ Keep 300-500 words

  3. Call LLM
     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        └─ model="gemini-2.5-flash-lite" (fast, precise)
        └─ temperature=0.7 (balanced creativity & factuality)

     response = llm.invoke(prompt)
        │
        ▼
     (Calls Google Gemini API with prompt)
        │
        ▼
     Returns: LLM Message object

  4. Extract Content
     report_text = response.content

  5. Return Updated State
     return {
       "hazard": "wildfire",
       "report": "Incident Summary: ...\n\nEcological Impact: ...\n\nRecommended Actions: ..."
     }
```

**Gemini API Integration:**
```
Authentication:
  - API Key from environment variable: GOOGLE_API_KEY
  - Loaded via: load_dotenv() from .env file

Model Details:
  - Model: gemini-2.5-flash-lite
  - Type: Flash model (optimized for speed)
  - Temperature: 0.7 (creative but grounded)
  - Streaming: Not enabled (full response after completion)
```

---

### 5. **Data Flow Sequences**

#### Training Flow

```
START
  ├─ POST /train endpoint
  ├─ train() function called
  │   │
  │   ├─ train_model(data_dir='merged_dataset')
  │   │   │
  │   │   ├─ ImageFolder('merged_dataset')
  │   │   │  └─ Loads all images by class
  │   │   │
  │   │   ├─ Random Split (80-20)
  │   │   │  ├─ train_dataset: 4000 images (example)
  │   │   │  └─ val_dataset: 1000 images
  │   │   │
  │   │   ├─ Create DataLoaders
  │   │   │  ├─ train_loader: BATCH_SIZE=32, shuffle=True
  │   │   │  └─ val_loader: BATCH_SIZE=32, shuffle=False
  │   │   │
  │   │   ├─ Model setup
  │   │   │  ├─ get_model(num_classes=4)
  │   │   │  ├─ optimizer: Adam(lr=1e-4)
  │   │   │  └─ scheduler: ReduceLROnPlateau
  │   │   │
  │   │   └─ Training Loop (EPOCHS=20)
  │   │       For epoch in range(20):
  │   │         ├─ Loop through train_loader
  │   │         │  └─ Forward → Loss → Backward → Update
  │   │         │
  │   │         └─ Loop through val_loader
  │   │            └─ Calculate validation loss
  │   │            └─ Update scheduler
  │   │            └─ Save best model if improved
  │   │
  │   └─ Return "Training completed"
  │
  └─ Response: {"status": "Training completed"}
END
```

#### Prediction + Report Flow

```
START: User uploads image via API or Dashboard
  │
  ├─ POST /predict with file
  │
  ├─ predict_image(file) async function
  │   │
  │   ├─ predict(file.file)
  │   │   │
  │   │   ├─ File Buffer → preprocess_image()
  │   │   │   │
  │   │   │   ├─ file.seek(0) - Reset pointer
  │   │   │   │
  │   │   │   ├─ PIL.Image.open(file).convert('RGB')
  │   │   │   │
  │   │   │   ├─ val_transform (Resize + Normalize)
  │   │   │   │
  │   │   │   └─ unsqueeze(0) → Tensor Shape: (1, 3, 224, 224)
  │   │   │
  │   │   ├─ model(image_tensor)
  │   │   │   │
  │   │   │   ├─ Forward pass through ResNet50
  │   │   │   │
  │   │   │   └─ Output: logits shape (1, 4)
  │   │   │
  │   │   ├─ torch.max(output, 1) → argmax
  │   │   │
  │   │   └─ Return: hazard_class_name
  │   │
  │   └─ hazard = "wildfire" (example)
  │
  ├─ state = {"hazard": "wildfire", "report": ""}
  │
  ├─ graph = build_graph()
  │
  ├─ result = graph.invoke(state)
  │   │
  │   ├─ Executes report_node
  │   │   │
  │   │   ├─ generate_report(state)
  │   │   │
  │   │   ├─ Create prompt with hazard
  │   │   │
  │   │   ├─ Call Google Gemini API
  │   │   │   │
  │   │   │   └─ LLM generates comprehensive report
  │   │   │
  │   │   └─ Return updated state with report
  │   │
  │   └─ result = {"hazard": "wildfire", "report": "..."}
  │
  └─ Response: 
     {
       "hazard": "wildfire",
       "report": "Incident Summary: Active wildfire...\n..."
     }
END: Return to user
```

---

### 6. **Configuration & Device Management** (`configs/config.py`)

```
Configuration Parameters:
┌─────────────────────────────────────────┐
│ Data Configuration                      │
├─────────────────────────────────────────┤
│ DATA_DIR = "merged_dataset"             │
│ IMG_SIZE = 224 (ResNet50 standard)      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Training Hyperparameters                │
├─────────────────────────────────────────┤
│ BATCH_SIZE = 32 (GPU memory balanced)   │
│ EPOCHS = 20 (sufficient for convergence)│
│ LR = 1e-4 (fine-tuning learning rate)   │
│ SEED = 42 (reproducibility)             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Model Configuration                     │
├─────────────────────────────────────────┤
│ MODEL_PATH = "models/...model.pt"       │
│ CLASSES = ["deforestation",             │
│            "oil_spill",                 │
│            "wildfire",                  │
│            "normal"]                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Device Management                       │
├─────────────────────────────────────────┤
│ DEVICE = torch.cuda.is_available()      │
│    ├─ True → CUDA GPU (Fast)            │
│    └─ False → CPU (Fallback)            │
└─────────────────────────────────────────┘
```

---

### 7. **Preprocessing Pipeline** (`utils/preprocess.py`)

```
Training Augmentation (train_transform):
  Input: PIL Image
    │
    ├─ RandomResizedCrop(224)
    │  └─ Crop to random aspect ratio, then resize
    │     (Handles different image compositions)
    │
    ├─ RandomHorizontalFlip()
    │  └─ Mirror image 50% of time
    │     (Left/right invariance)
    │
    ├─ RandomVerticalFlip()
    │  └─ Flip vertically 50% of time
    │     (Up/down invariance)
    │
    ├─ RandomRotation(20°)
    │  └─ Rotate ±20 degrees
    │     (Handle various satellite angles)
    │
    ├─ ColorJitter(brightness, contrast, saturation)
    │  └─ Vary image colors 50% of time
    │     (Handle different lighting/weather)
    │
    ├─ ToTensor()
    │  └─ Convert to normalized PyTorch tensor
    │     (Range: [0, 1] with ImageNet stats)
    │
    └─ Output: Tensor shape (3, 224, 224)

Validation/Inference (val_transform):
  Input: PIL Image
    │
    ├─ Resize((224, 224))
    │  └─ No cropping, just resize equally
    │
    ├─ ToTensor()
    │  └─ Normalize with ImageNet stats
    │
    └─ Output: Tensor shape (3, 224, 224)
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Web dashboard for image upload |
| **API** | FastAPI | RESTful API with async support |
| **ML Framework** | PyTorch | Model training & inference |
| **Vision** | torchvision | ResNet50, transforms |
| **Image Processing** | Pillow | Image I/O and conversion |
| **AI Agents** | LangGraph | Workflow orchestration |
| **LLM Integration** | LangChain | Google Gemini integration |
| **LLM API** | Google Gemini | Report generation |
| **Environment** | python-dotenv | API key management |
| **Async** | asyncio | Concurrent request handling |
| **Logging** | Python logging | Event tracking |

---

## Data Types & Structures

### Request/Response Objects

**POST /train**
```python
Request: Empty body {}

Response:
{
  "status": "Training completed"
}
```

**POST /predict**
```python
Request: 
  - Multipart form data
  - File key: "file"
  - File types: PNG, JPG, JPEG
  - Max size: Configurable (FastAPI default: 25MB)

Response:
{
  "hazard": "string",        # One of 4 classes
  "report": "string"         # Markdown formatted report
}
```

### Agent State TypedDict
```python
class AgentState(TypedDict):
    hazard: str              # Classification result
    report: str              # AI-generated report
```

---

## Deployment Considerations

### Single Machine Deployment
```
GPU Machine (Recommended):
  ├─ NVIDIA GPU with CUDA 11.0+
  ├─ 8GB+ VRAM for model inference
  ├─ Batch training: 16GB+ RAM, 16GB VRAM ideal
  └─ Training time: ~10-30 min with GPU

CPU-only Machine:
  ├─ Intel/AMD CPU with SSE2 support
  ├─ 16GB+ RAM
  └─ Training time: ~2-5 hours
```

### Scaling Considerations
```
Current Limits:
  - Single request handling (sequential)
  - Model loading delays (~3-5sec)
  - Gemini API rate limits (depends on plan)

Optimization Opportunities:
  - Model caching (load once, reuse)
  - Async batch processing
  - Request queuing with task workers
  - Model quantization for inference speed
  - Distributed training for larger datasets
```

---

## Monitoring & Logging

### Logging Points
- `app/main.py`: API requests, predictions, errors
- `models/train.py`: Epoch progress, loss values, model saves
- `models/predict.py`: Model loads, predictions, errors
- `agent/nodes.py`: Report generation, API calls

### Log Levels
- **INFO**: Normal operations (requests, completions)
- **WARNING**: Potential issues (slow performance)
- **ERROR**: Failed operations (missing files, API errors)

### Metrics to Monitor
- Training loss progression
- Validation accuracy
- Inference latency per request
- API response times
- Gemini API quota usage
- GPU memory utilization
