#!/usr/bin/env python3
"""
Example script demonstrating the vision-language environment for Qwen2.5-VL-7B training.

This example shows how to:
1. Set up the vision-language environment
2. Load and process image datasets
3. Test the environment with different reward mechanisms
4. Run inference with the trained model

Usage:
    python examples/vision_training_example.py
"""

import os
import sys
import base64
import io
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from datasets import Dataset
from openai import OpenAI

# Add verifiers to path
sys.path.append(str(Path(__file__).parent.parent))

import verifiers as vf
from verifiers.envs.vision_env import VisionLanguageEnv
from verifiers.tools.image_tools import IMAGE_TOOLS
from verifiers.rubrics.vision_rewards import ImageRewardCalculator, create_image_reward_func


def create_sample_image() -> str:
    """Create a sample image for testing."""
    # Create a simple test image
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
    draw.circle([300, 100], 40, fill='yellow', outline='black', width=2)
    draw.text((50, 200), "Sample Image", fill='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def create_sample_dataset() -> Dataset:
    """Create a sample dataset for testing."""
    sample_image = create_sample_image()
    
    samples = [
        {
            'image_data': sample_image,
            'question': 'What shapes can you see in this image?',
            'answer': 'I can see a red rectangle and a yellow circle in this image.'
        },
        {
            'image_data': sample_image,
            'question': 'What colors are present in this image?',
            'answer': 'The image contains red, yellow, blue, and black colors.'
        },
        {
            'image_data': sample_image,
            'question': 'Is there any text in this image?',
            'answer': 'Yes, there is text that says "Sample Image" in the image.'
        }
    ]
    
    return Dataset.from_list(samples)


def test_environment():
    """Test the vision-language environment."""
    print("🔧 Creating vision-language environment...")
    
    # Create environment with different reward configurations
    env = VisionLanguageEnv(
        tools=IMAGE_TOOLS,
        max_turns=3,
        reward_types=["format", "tool", "answer", "image"],
        reward_weights=[0.2, 0.3, 0.3, 0.2]
    )
    
    print(f"✅ Environment created with {len(env.tools)} image processing tools")
    print(f"📊 Reward functions: {env.get_reward_func_names()}")
    
    return env


def test_image_tools():
    """Test image processing tools."""
    print("\n🖼️  Testing image processing tools...")
    
    # Create sample image
    sample_image = create_sample_image()
    print("✅ Created sample image")
    
    # Test crop tool
    from verifiers.tools.image_tools import crop_image, rotate_image, adjust_brightness
    
    try:
        cropped = crop_image(sample_image, 50, 50, 200, 150)
        print("✅ Crop tool works")
    except Exception as e:
        print(f"❌ Crop tool failed: {e}")
    
    try:
        rotated = rotate_image(sample_image, 45)
        print("✅ Rotate tool works")
    except Exception as e:
        print(f"❌ Rotate tool failed: {e}")
    
    try:
        brightened = adjust_brightness(sample_image, 1.5)
        print("✅ Brightness adjustment works")
    except Exception as e:
        print(f"❌ Brightness adjustment failed: {e}")


def test_reward_mechanisms():
    """Test different reward mechanisms."""
    print("\n🎯 Testing reward mechanisms...")
    
    # Create image reward calculator
    reward_calc = ImageRewardCalculator(
        self_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        clip_model_name="openai/clip-vit-base-patch32"
    )
    
    # Test with sample data
    original_image = create_sample_image()
    
    # Create a "processed" version (slightly rotated)
    from verifiers.tools.image_tools import rotate_image
    processed_image = rotate_image(original_image, 15)
    
    question = "What shapes can you see in this image?"
    reasoning = "I can see geometric shapes including a rectangle and a circle."
    
    print("Testing embedding-based reward...")
    try:
        embedding_reward = reward_calc.embedding_reward(
            original_image=original_image,
            processed_image=processed_image,
            question=question,
            reasoning=reasoning
        )
        print(f"✅ Embedding reward: {embedding_reward:.3f}")
    except Exception as e:
        print(f"❌ Embedding reward failed: {e}")
    
    # Test combined rewards
    print("Testing combined image rewards...")
    try:
        combined_rewards = reward_calc.combined_image_reward(
            original_image=original_image,
            processed_image=processed_image,
            question=question,
            reasoning=reasoning,
            reward_types=["embedding"],  # Use only embedding for testing
            weights=[1.0]
        )
        print(f"✅ Combined rewards: {combined_rewards}")
    except Exception as e:
        print(f"❌ Combined rewards failed: {e}")


def test_rollout_simulation():
    """Simulate a rollout with the environment."""
    print("\n🎮 Testing rollout simulation...")
    
    # Create environment
    env = test_environment()
    
    # Create sample dataset
    dataset = create_sample_dataset()
    sample = dataset[0]
    
    # Create mock client for testing
    class MockClient:
        def chat(self, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': '''<think>
I can see this image contains geometric shapes. Let me analyze it more carefully to identify all the shapes and colors present.
</think>

<tool>{"name": "get_image_info", "args": {}}</tool>'''
                        })()
                    })()]
            return MockResponse()
    
    mock_client = MockClient()
    
    # Create prompt
    system_msg = {"role": "system", "content": env.system_prompt}
    user_msg = {
        "role": "user", 
        "content": [
            {"type": "text", "text": sample['question']},
            {"type": "image", "image": sample['image_data']}
        ]
    }
    prompt = [system_msg, user_msg]
    
    print("📝 Created sample prompt")
    print(f"❓ Question: {sample['question']}")
    
    try:
        # Test environment response generation
        env.set_image(sample['image_data'])
        
        # Simulate tool call
        tool_result = env.call_tool('{"name": "get_image_info", "args": {}}')
        print(f"🔧 Tool result: {tool_result[:100]}...")
        
        print("✅ Rollout simulation successful")
    except Exception as e:
        print(f"❌ Rollout simulation failed: {e}")


def demonstrate_training_setup():
    """Demonstrate how to set up training."""
    print("\n🚀 Demonstrating training setup...")
    
    # Create environment
    env = VisionLanguageEnv(
        tools=IMAGE_TOOLS,
        max_turns=5,
        reward_types=["format", "tool", "answer", "image"],
        reward_weights=[0.2, 0.3, 0.3, 0.2]
    )
    
    # Create sample dataset
    dataset = create_sample_dataset()
    
    # Format for training
    def format_sample(example):
        system_msg = {"role": "system", "content": env.system_prompt}
        user_msg = {
            "role": "user", 
            "content": [
                {"type": "text", "text": example['question']},
                {"type": "image", "image": example['image_data']}
            ]
        }
        return {
            "prompt": [system_msg, user_msg],
            "answer": example['answer'],
            "image_data": example['image_data'],
            "question": example['question']
        }
    
    formatted_dataset = dataset.map(format_sample)
    
    print(f"✅ Formatted dataset with {len(formatted_dataset)} samples")
    print("📋 Sample training data structure:")
    sample = formatted_dataset[0]
    print(f"  - Prompt length: {len(sample['prompt'])}")
    print(f"  - Answer: {sample['answer'][:50]}...")
    print(f"  - Has image data: {'image_data' in sample}")
    
    # Show GRPO configuration
    print("\n⚙️  GRPO Training Configuration:")
    training_config = {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "batch_size": 2,
        "learning_rate": 1e-6,
        "num_epochs": 3,
        "max_turns": 5,
        "reward_types": ["format", "tool", "answer", "image"],
        "reward_weights": [0.2, 0.3, 0.3, 0.2],
        "image_reward_types": ["self", "embedding"],
        "use_lora": True,
        "lora_rank": 64
    }
    
    for key, value in training_config.items():
        print(f"  {key}: {value}")


def main():
    """Main function to run all tests."""
    print("🎯 Vision-Language Environment Testing")
    print("=" * 50)
    
    # Test image tools
    test_image_tools()
    
    # Test environment
    test_environment()
    
    # Test reward mechanisms
    test_reward_mechanisms()
    
    # Test rollout simulation
    test_rollout_simulation()
    
    # Demonstrate training setup
    demonstrate_training_setup()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n📚 Next steps:")
    print("1. Install required dependencies: pip install -r requirements.txt")
    print("2. Start vLLM server: CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-VL-7B-Instruct'")
    print("3. Run training: python train_qwen_vision.py --use_lora --num_samples 1000")
    print("4. Monitor training with wandb or logs")


if __name__ == "__main__":
    main()