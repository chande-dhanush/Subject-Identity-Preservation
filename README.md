
# 🧬 Subject Identity Preservation with OmniGen

**Subject-Identity-Preservation** is a custom pipeline built upon the **open-source OmniGen model**.  
It allows dynamic **background variation** while preserving the **core identity of the subject** in an input image — making it perfect for consistent image generation with creative diversity.

---

## 🎯 Key Features

- 🖼️ Preserve the identity of a subject across multiple generations.
- 🌆 Dynamically vary the background using textual prompts.
- 🔧 Easily extendable and modifiable pipeline.
- 🧩 LoRA Checkpoint integration for fine-tuned control.

---

## 📦 Repo Structure

```
subject-identity-preservation/
├── OmniGen/                    # Cloned OmniGen repository
│   ├── app.py                  # Original inference pipeline
│   ├── new_pipe.py             # Custom pipeline for identity preservation
│   └── checkpoints/
│       └── lora_checkpoint.safetensors
├── requirements.txt
├── README.md
```

---

## 🛠️ Setup Instructions

### 🔧 Requirements

- Python 3.10+
- CUDA-compatible GPU
- torch, diffusers, accelerate, etc. (see requirements.txt)

---

### 🚀 Running the Pipeline

1. **Set up your environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Clone the OmniGen repository**
   ```bash
   git clone https://github.com/VectorSpaceLab/OmniGen.git
   cd OmniGen
   ```

3. **Integrate the modified files**
   - Copy `new_pipe.py` and the LoRA checkpoint file into the `OmniGen/` directory.

4. **Run the model**
   - To use the **original OmniGen** pipeline:
     ```bash
     python app.py
     ```
   - To use the **modified identity-preserving pipeline**:
     ```bash
     python new_pipe.py
     ```

---

## 🧠 How It Works

1. Takes an input image of a subject (person/object).
2. Extracts the subject while preserving detailed features.
3. Prompts are used to generate various realistic or stylized backgrounds.
4. The subject is blended back into the generated environment.

---

## 🌍 Applications

- AI avatars with consistent facial identity
- Subject-preserving fashion try-on
- Creative background alterations for profile images
- Storyboarding with visual continuity

---


## 📬 Contact

For collaborations, improvements, or suggestions – feel free to connect!

---

