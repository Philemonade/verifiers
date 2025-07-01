#!/usr/bin/env python3
"""
Training script for Qwen2.5-VL-7B with "think with image" capabilities using GRPO.

This script implements multi-turn reinforcement learning with Group-Relative Policy Optimization
for training vision-language models to use image processing tools strategically.

Features:
- Vision-language environment with image processing tools
- Multiple reward mechanisms (format, tool, answer, image rewards)
- Image-as-reward signals (self-reward, teacher-reward, attention-reward, embedding-reward)
- Multi-turn conversations with tool usage
- GRPO training with vision-specific optimizations

Usage:
    # Start vLLM inference server first:
    CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-VL-7B-Instruct' --tensor-parallel-size 2
    
    # Run training:
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml train_qwen_vision.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
import io
from PIL import Image
import torch
from datasets import Dataset, load_dataset
import wandb

# Add verifiers to path
sys.path.append(str(Path(__file__).parent))

import verifiers as vf
from verifiers.envs.vision_env import VisionLanguageEnv
from verifiers.tools.image_tools import IMAGE_TOOLS
from verifiers.rubrics.vision_rewards import ImageRewardCalculator, create_image_reward_func


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train_qwen_vision.log')
        ]
    )


def load_vision_dataset(dataset_name: str = "nlphuji/flickr30k", 
                       num_samples: int = 1000,
                       split: str = "test") -> Dataset:
    """
    Load and prepare a vision-language dataset for training.
    
    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to use
        split: Dataset split to use
        
    Returns:
        Processed dataset with image and question pairs
    """
    logging.info(f"Loading dataset: {dataset_name}")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    def process_sample(example):
        """Process each sample to create question-answer pairs."""
        # Convert PIL image to base64
        if 'image' in example and example['image'] is not None:
            img = example['image']
            if isinstance(img, Image.Image):
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                image_b64 = base64.b64encode(buffer.getvalue()).decode()
            else:
                image_b64 = None
        else:
            image_b64 = None
        
        # Create question from caption or use generic questions
        if 'caption' in example:
            caption = example['caption']
            questions = [
                f"What is happening in this image? Describe it in detail.",
                f"Can you identify the main objects or people in this image?",
                f"What is the setting or location of this image?",
                f"Describe the colors, lighting, and composition of this image.",
                f"What emotions or mood does this image convey?"
            ]
            # Use caption as ground truth answer
            answer = caption if isinstance(caption, str) else caption[0] if isinstance(caption, list) else "No caption available"
        else:
            questions = [
                "Describe what you see in this image.",
                "What are the main elements in this image?",
                "Analyze the composition and visual elements of this image."
            ]
            answer = "Please analyze the image carefully."
        
        return {
            'image_data': image_b64,
            'question': questions[0],  # Use first question
            'answer': answer,
            'all_questions': questions
        }
    
    # Process the dataset
    processed_dataset = dataset.map(process_sample, remove_columns=dataset.column_names)
    
    # Filter out samples without images
    processed_dataset = processed_dataset.filter(lambda x: x['image_data'] is not None)
    
    logging.info(f"Processed dataset: {len(processed_dataset)} samples")
    return processed_dataset


def create_vision_environment(
    reward_types: List[str] = ["format", "tool", "answer", "image"],
    reward_weights: List[float] = [0.2, 0.3, 0.3, 0.2],
    image_reward_types: List[str] = ["self", "embedding"],
    image_reward_weights: List[float] = [0.7, 0.3],
    max_turns: int = 5
) -> VisionLanguageEnv:
    """
    Create the vision-language environment with specified reward configuration.
    
    Args:
        reward_types: Types of rewards to use
        reward_weights: Weights for each reward type
        image_reward_types: Types of image rewards to use
        image_reward_weights: Weights for image rewards
        max_turns: Maximum number of conversation turns
        
    Returns:
        Configured VisionLanguageEnv instance
    """
    logging.info("Creating vision-language environment")
    
    # Create image reward calculator
    image_reward_calc = ImageRewardCalculator(
        self_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        teacher_model_name="Qwen/Qwen2.5-VL-72B-Instruct",  # Use larger model if available
        clip_model_name="openai/clip-vit-base-patch32"
    )
    
    # Create the environment
    env = VisionLanguageEnv(
        tools=IMAGE_TOOLS,
        max_turns=max_turns,
        reward_types=reward_types,
        reward_weights=reward_weights,
        judge_model_name="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    
    # Add advanced image reward function
    if "image" in reward_types:
        image_reward_func = create_image_reward_func(
            reward_calculator=image_reward_calc,
            reward_types=image_reward_types,
            weights=image_reward_weights
        )
        # Replace the simple image reward with advanced one
        env.rubric.reward_funcs = [
            func if func.__name__ != '_image_reward' else image_reward_func
            for func in env.rubric.reward_funcs
        ]
    
    return env


def prepare_dataset_for_training(dataset: Dataset, env: VisionLanguageEnv) -> Dataset:
    """
    Prepare the dataset for GRPO training by converting to the required format.
    
    Args:
        dataset: Vision dataset
        env: Vision-language environment
        
    Returns:
        Dataset formatted for GRPO training
    """
    def format_sample(example):
        """Format each sample for training."""
        # Create system message
        system_msg = {
            "role": "system",
            "content": env.system_prompt
        }
        
        # Create user message with image and question
        user_content = [
            {"type": "text", "text": example['question']},
            {"type": "image", "image": example['image_data']}
        ]
        
        user_msg = {
            "role": "user", 
            "content": user_content
        }
        
        return {
            "prompt": [system_msg, user_msg],
            "answer": example['answer'],
            "image_data": example['image_data'],
            "question": example['question']
        }
    
    formatted_dataset = dataset.map(format_sample)
    logging.info(f"Formatted dataset for training: {len(formatted_dataset)} samples")
    return formatted_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL-7B with vision tools")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", default="nlphuji/flickr30k", help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--max_turns", type=int, default=5, help="Maximum conversation turns")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluation every N steps")
    parser.add_argument("--output_dir", default="./outputs/qwen-vision-grpo", help="Output directory")
    parser.add_argument("--run_name", default="qwen-vision-think", help="Run name for logging")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA training")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--reward_types", nargs="+", default=["format", "tool", "answer", "image"], 
                       help="Types of rewards to use")
    parser.add_argument("--reward_weights", nargs="+", type=float, default=[0.2, 0.3, 0.3, 0.2],
                       help="Weights for each reward type")
    parser.add_argument("--image_reward_types", nargs="+", default=["self", "embedding"],
                       help="Types of image rewards to use")
    parser.add_argument("--image_reward_weights", nargs="+", type=float, default=[0.7, 0.3],
                       help="Weights for image rewards")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting Qwen2.5-VL-7B vision training")
    logging.info(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_vision_dataset(
        dataset_name=args.dataset_name,
        num_samples=args.num_samples
    )
    
    # Create environment
    env = create_vision_environment(
        reward_types=args.reward_types,
        reward_weights=args.reward_weights,
        image_reward_types=args.image_reward_types,
        image_reward_weights=args.image_reward_weights,
        max_turns=args.max_turns
    )
    
    # Prepare dataset for training
    train_dataset = prepare_dataset_for_training(dataset, env)
    
    # Load model and tokenizer
    logging.info(f"Loading model: {args.model_name}")
    model, tokenizer = vf.get_model_and_tokenizer(args.model_name)
    
    # Setup LoRA if requested
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        logging.info(f"Applied LoRA with rank {args.lora_rank}")
    
    # Create GRPO training arguments
    training_args = vf.grpo_defaults(
        run_name=args.run_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=50,
        report_to="wandb" if not args.no_wandb else None,
        remove_unused_columns=False,  # Important for vision models
        dataloader_num_workers=0,  # Avoid multiprocessing issues with images
        bf16=True,  # Use bfloat16 for better stability
        gradient_checkpointing=True,  # Save memory
        max_grad_norm=1.0,  # Gradient clipping
        num_generations=4,  # Number of generations per prompt for GRPO
        beta=0.01,  # KL penalty coefficient
        num_iterations=1,  # Number of update steps per batch
    )
    
    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(
            project="qwen-vision-grpo",
            name=args.run_name,
            config=vars(args)
        )
    
    # Create GRPO trainer
    logging.info("Creating GRPO trainer")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # Start training
    logging.info("Starting GRPO training")
    try:
        trainer.train()
        logging.info("Training completed successfully")
        
        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_output_dir)
        logging.info(f"Final model saved to: {final_output_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    # Evaluation
    logging.info("Running final evaluation")
    try:
        # Create evaluation dataset (smaller subset)
        eval_dataset = dataset.select(range(min(100, len(dataset))))
        eval_dataset = prepare_dataset_for_training(eval_dataset, env)
        
        # Run evaluation
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        logging.info(f"Evaluation results: {eval_results}")
        
        # Log to wandb if enabled
        if not args.no_wandb:
            wandb.log({"final_eval": eval_results})
            wandb.finish()
            
    except Exception as e:
        logging.warning(f"Evaluation failed: {e}")
    
    logging.info("Training script completed")


if __name__ == "__main__":
    main()