# Qwen2.5-VL-7B "Think with Image" Training System

This system implements multi-turn reinforcement learning via Group-Relative Policy Optimization (GRPO) for training Qwen2.5-VL-7B to "think with image" - a paradigm where the model strategically uses image processing tools to enhance its reasoning capabilities.

## 🎯 Overview

The core idea is to train vision-language models to:
1. **Think systematically** about visual problems within `<think></think>` tags
2. **Use image tools strategically** to process images when needed
3. **Learn from image rewards** - treating processed images as continuous reward signals
4. **Multi-turn reasoning** with tool usage and environment feedback

## 🏗️ Architecture

### Key Components

1. **VisionLanguageEnv** - Multi-turn environment for image reasoning
2. **Image Processing Tools** - 11 basic image manipulation tools
3. **Advanced Reward System** - 4 types of image-based rewards
4. **GRPO Training** - Optimized for vision-language tasks

### Reward Mechanisms

The system implements multiple reward types:

#### Standard Rewards
- **Format Reward**: Proper XML formatting (`<think>`, `<tool>`, `<answer>`)
- **Tool Reward**: Appropriate and successful tool usage
- **Answer Reward**: Providing final answers

#### Image-as-Reward (Novel)
- **Self-Reward**: Same model judges image processing quality
- **Teacher-Reward**: Larger model (72B) judges quality  
- **Attention-Reward**: Based on attention patterns between image and text
- **Embedding-Reward**: CLIP-based similarity between processed image and question

## 🚀 Quick Start

### 1. Installation

```bash
# Install core dependencies
pip install -r requirements_vision.txt

# Install flash attention for better performance (optional)
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 2. Start vLLM Inference Server

```bash
# Use 2 GPUs for inference
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-VL-7B-Instruct' --tensor-parallel-size 2
```

### 3. Run Training

```bash
# Use remaining GPUs for training
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml train_qwen_vision.py \
    --use_lora \
    --num_samples 1000 \
    --max_turns 5 \
    --reward_types format tool answer image \
    --reward_weights 0.2 0.3 0.3 0.2 \
    --image_reward_types self embedding \
    --image_reward_weights 0.7 0.3
```

## 🛠️ Image Processing Tools

The system includes 11 image processing tools:

1. **crop_image** - Crop image to specified region
2. **zoom_image** - Zoom in/out with optional center point
3. **rotate_image** - Rotate by specified angle
4. **flip_image** - Horizontal or vertical flip
5. **adjust_brightness** - Modify brightness levels
6. **adjust_contrast** - Modify contrast levels
7. **apply_blur** - Gaussian blur effect
8. **apply_sharpen** - Sharpen image details
9. **convert_grayscale** - Convert to grayscale
10. **resize_image** - Resize with aspect ratio options
11. **get_image_info** - Extract image metadata

### Tool Usage Example

```xml
<think>
I need to analyze this image more carefully. The image seems quite dark, so I should brighten it first to see the details better.
</think>

<tool>{"name": "adjust_brightness", "args": {"factor": 1.5}}</tool>
```

## 📊 Training Configuration

### Recommended Settings

```python
# Environment Configuration
reward_types = ["format", "tool", "answer", "image"]
reward_weights = [0.2, 0.3, 0.3, 0.2]  # Balanced approach
image_reward_types = ["self", "embedding"]  # Fast and effective
image_reward_weights = [0.7, 0.3]

# Training Configuration
training_args = {
    "learning_rate": 1e-6,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "max_turns": 5,
    "num_generations": 4,  # GRPO group size
    "beta": 0.01,  # KL penalty
    "use_lora": True,
    "lora_rank": 64
}
```

### Advanced Configurations

#### High-Quality Training (Slower)
```bash
python train_qwen_vision.py \
    --image_reward_types self teacher attention embedding \
    --image_reward_weights 0.3 0.3 0.2 0.2 \
    --teacher_model_name "Qwen/Qwen2.5-VL-72B-Instruct"
```

#### Fast Training (Development)
```bash
python train_qwen_vision.py \
    --image_reward_types embedding \
    --image_reward_weights 1.0 \
    --num_samples 100 \
    --max_turns 3
```

## 📈 Monitoring and Evaluation

### Weights & Biases Integration

The system automatically logs:
- Reward breakdowns (format, tool, answer, image)
- Tool usage statistics
- Image processing patterns
- Training metrics

```bash
# Enable wandb logging (default)
python train_qwen_vision.py --run_name "my-vision-experiment"

# Disable wandb
python train_qwen_vision.py --no_wandb
```

### Key Metrics to Monitor

1. **Reward Components**:
   - `format_reward`: XML formatting quality
   - `tool_reward`: Tool usage appropriateness
   - `answer_reward`: Final answer quality
   - `image_reward`: Image processing effectiveness

2. **Tool Usage**:
   - Tool call success rate
   - Tool diversity
   - Average tools per conversation

3. **Training Stability**:
   - KL divergence
   - Gradient norms
   - Loss trends

## 🧪 Testing and Development

### Run Example Tests

```bash
# Test all components
python examples/vision_training_example.py

# Test specific components
python -c "
from verifiers.envs.vision_env import VisionLanguageEnv
from verifiers.tools.image_tools import IMAGE_TOOLS
env = VisionLanguageEnv(tools=IMAGE_TOOLS)
print(f'Environment created with {len(env.tools)} tools')
"
```

### Custom Dataset Integration

```python
from datasets import Dataset
from verifiers.envs.vision_env import VisionLanguageEnv

# Create your dataset
def create_custom_dataset():
    return Dataset.from_list([
        {
            'image_data': base64_encoded_image,
            'question': 'Your question here',
            'answer': 'Expected answer'
        }
        # ... more samples
    ])

# Use with environment
dataset = create_custom_dataset()
env = VisionLanguageEnv(tools=IMAGE_TOOLS)
# ... training setup
```

## 🔧 Advanced Features

### Custom Reward Functions

```python
from verifiers.rubrics.vision_rewards import ImageRewardCalculator

def custom_image_reward(original_image, processed_image, question, reasoning, **kwargs):
    # Your custom reward logic here
    return reward_score

# Integrate with environment
reward_calc = ImageRewardCalculator()
reward_calc.custom_reward = custom_image_reward
```

### Custom Image Tools

```python
def my_custom_tool(image_data: str, param1: int, param2: str) -> str:
    """
    Custom image processing tool.
    
    Args:
        image_data: Base64 encoded image
        param1: Integer parameter
        param2: String parameter
    
    Returns:
        Processed base64 image or result string
    """
    # Your tool logic here
    return processed_image_data

# Add to environment
from verifiers.tools.image_tools import IMAGE_TOOLS
custom_tools = IMAGE_TOOLS + [my_custom_tool]
env = VisionLanguageEnv(tools=custom_tools)
```

## 📚 Research and Theory

### Core Concepts

1. **Image as Reward**: Using processed images as continuous reward signals rather than just discrete task completion rewards.

2. **Multi-turn Tool Use**: Models learn when and how to use tools strategically across multiple conversation turns.

3. **Think with Image**: Explicit reasoning about visual content with tool-assisted analysis.

### Reward Signal Design

The image reward system addresses the key challenge of providing meaningful feedback for vision-language tool use:

- **Immediate Feedback**: Tools provide instant visual feedback
- **Continuous Signals**: Image similarity/quality metrics as continuous rewards
- **Multi-modal Alignment**: CLIP and attention-based rewards ensure image-text coherence
- **Self-improvement**: Models learn to judge their own image processing quality

### Training Dynamics

GRPO with image rewards creates several beneficial training dynamics:

1. **Exploration**: Models explore different tool combinations
2. **Specialization**: Tools are used for specific visual reasoning needs
3. **Efficiency**: Models learn when NOT to use tools
4. **Generalization**: Tool usage patterns transfer across visual domains

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Issues**:
   ```bash
   # Reduce batch size
   python train_qwen_vision.py --batch_size 1 --gradient_accumulation_steps 8
   
   # Use LoRA
   python train_qwen_vision.py --use_lora --lora_rank 32
   ```

2. **vLLM Connection Issues**:
   ```bash
   # Check server status
   curl http://localhost:8000/health
   
   # Restart with different port
   vf-vllm --model 'Qwen/Qwen2.5-VL-7B-Instruct' --port 8001
   ```

3. **Image Processing Errors**:
   ```bash
   # Install additional dependencies
   pip install opencv-python-headless
   pip install Pillow --upgrade
   ```

4. **Import Errors**:
   ```bash
   # Install vision dependencies
   pip install clip-by-openai
   pip install torchvision
   ```

### Performance Optimization

1. **Memory Optimization**:
   - Use LoRA training (`--use_lora`)
   - Enable gradient checkpointing
   - Reduce sequence lengths
   - Use bf16 precision

2. **Speed Optimization**:
   - Use only embedding rewards for development
   - Reduce max_turns for faster episodes
   - Use smaller datasets for testing
   - Enable flash attention

## 📖 API Reference

### VisionLanguageEnv

```python
class VisionLanguageEnv(MultiTurnEnv):
    def __init__(
        self,
        tools: List[Callable] = None,
        system_prompt: str = VISION_SYSTEM_PROMPT,
        max_turns: int = 10,
        reward_types: List[str] = ["format", "tool", "answer", "image"],
        reward_weights: List[float] = None,
        judge_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    )
```

### ImageRewardCalculator

```python
class ImageRewardCalculator:
    def __init__(
        self,
        self_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        teacher_model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        clip_model_name: str = "openai/clip-vit-base-patch32"
    )
    
    def combined_image_reward(
        self,
        original_image: str,
        processed_image: str,
        question: str,
        reasoning: str,
        reward_types: List[str] = ["self", "embedding"],
        weights: List[float] = None
    ) -> Dict[str, float]
```

## 🤝 Contributing

1. **Adding New Tools**: Implement functions following the image tool signature
2. **Custom Rewards**: Extend ImageRewardCalculator with new reward methods
3. **Environment Extensions**: Inherit from VisionLanguageEnv for specialized use cases
4. **Dataset Adapters**: Create loaders for new vision-language datasets

## 📄 License

This project extends the verifiers framework and follows the same licensing terms.

## 🙏 Acknowledgments

- Built on the [verifiers](https://github.com/willccbb/verifiers) framework
- Uses Qwen2.5-VL models from Alibaba Cloud
- Incorporates CLIP for vision-language understanding
- Inspired by recent work in vision-language reasoning and tool use

---

For more examples and advanced usage, see the `examples/` directory and the main verifiers documentation.