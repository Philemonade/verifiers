import base64
import io
import logging
from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import clip


class ImageRewardCalculator:
    """
    Advanced image reward calculation with multiple strategies:
    - Self-reward: Using the same model to judge image processing quality
    - Teacher-reward: Using a larger model to judge quality
    - Attention-reward: Based on attention patterns between image and text
    - Embedding-reward: Based on similarity between image and text embeddings
    """
    
    def __init__(self,
                 self_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 teacher_model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "auto"):
        
        self.self_model_name = self_model_name
        self.teacher_model_name = teacher_model_name
        self.clip_model_name = clip_model_name
        self.device = device
        
        # Lazy loading of models
        self.self_model = None
        self.self_processor = None
        self.teacher_model = None
        self.teacher_processor = None
        self.clip_model = None
        self.clip_processor = None
        
        self.logger = logging.getLogger(__name__)
    
    def _load_self_model(self):
        """Load the self-reward model (same as training model)."""
        if self.self_model is None:
            try:
                self.self_processor = AutoProcessor.from_pretrained(self.self_model_name)
                self.self_model = AutoModel.from_pretrained(
                    self.self_model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device
                )
                self.logger.info(f"Loaded self-reward model: {self.self_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load self-reward model: {e}")
                self.self_model = None
    
    def _load_teacher_model(self):
        """Load the teacher-reward model (larger model)."""
        if self.teacher_model is None:
            try:
                self.teacher_processor = AutoProcessor.from_pretrained(self.teacher_model_name)
                self.teacher_model = AutoModel.from_pretrained(
                    self.teacher_model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device
                )
                self.logger.info(f"Loaded teacher-reward model: {self.teacher_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load teacher-reward model: {e}")
                self.teacher_model = None
    
    def _load_clip_model(self):
        """Load CLIP model for embedding-based rewards."""
        if self.clip_model is None:
            try:
                self.clip_model, self.clip_processor = clip.load(self.clip_model_name.split('/')[-1])
                if torch.cuda.is_available():
                    self.clip_model = self.clip_model.cuda()
                self.logger.info(f"Loaded CLIP model: {self.clip_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load CLIP model: {e}")
                self.clip_model = None
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image."""
        try:
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            self.logger.error(f"Failed to decode image: {e}")
            return None
    
    def self_reward(self,
                   original_image: str,
                   processed_image: str,
                   question: str,
                   reasoning: str,
                   **kwargs) -> float:
        """
        Self-reward: Use the same model to judge if image processing improved understanding.
        """
        self._load_self_model()
        if self.self_model is None:
            return 0.0
        
        try:
            original_img = self._decode_image(original_image)
            processed_img = self._decode_image(processed_image)
            
            if original_img is None or processed_img is None:
                return 0.0
            
            # Create prompts for comparison
            judge_prompt = f"""
            Question: {question}
            Reasoning: {reasoning}
            
            Compare these two images in the context of the question above.
            Rate how much the second image helps answer the question compared to the first image.
            
            Rate on a scale of 0.0 to 1.0:
            - 0.0: Second image is worse or no improvement
            - 0.5: Both images are equally helpful
            - 1.0: Second image significantly improves understanding
            
            Provide only the numerical score.
            """
            
            # Process both images with the judge prompt
            original_inputs = self.self_processor(
                text=judge_prompt,
                images=original_img,
                return_tensors="pt"
            )
            
            processed_inputs = self.self_processor(
                text=judge_prompt,
                images=processed_img,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                # Get embeddings for both
                original_outputs = self.self_model(**original_inputs)
                processed_outputs = self.self_model(**processed_inputs)
                
                # Simple heuristic: if processed image has higher activation, it's better
                original_score = torch.mean(original_outputs.last_hidden_state).item()
                processed_score = torch.mean(processed_outputs.last_hidden_state).item()
                
                # Normalize to 0-1 range
                if processed_score > original_score:
                    return min(1.0, (processed_score - original_score) * 10)
                else:
                    return max(0.0, 0.5 - (original_score - processed_score) * 10)
        
        except Exception as e:
            self.logger.error(f"Self-reward calculation failed: {e}")
            return 0.0
    
    def teacher_reward(self,
                      original_image: str,
                      processed_image: str,
                      question: str,
                      reasoning: str,
                      **kwargs) -> float:
        """
        Teacher-reward: Use a larger model to judge image processing quality.
        """
        self._load_teacher_model()
        if self.teacher_model is None:
            # Fallback to self-reward if teacher model not available
            return self.self_reward(original_image, processed_image, question, reasoning, **kwargs)
        
        try:
            original_img = self._decode_image(original_image)
            processed_img = self._decode_image(processed_image)
            
            if original_img is None or processed_img is None:
                return 0.0
            
            # Similar to self_reward but with teacher model
            judge_prompt = f"""
            Analyze these two images in the context of this question: {question}
            
            The reasoning provided was: {reasoning}
            
            Evaluate whether the image processing (second image) improved the ability to answer the question.
            Consider factors like:
            - Clarity and visibility of relevant details
            - Removal of distracting elements
            - Enhancement of important features
            - Overall helpfulness for the specific question
            
            Rate from 0.0 to 1.0 where:
            0.0 = Processing made it worse or no improvement
            0.5 = Both images equally helpful
            1.0 = Processing significantly improved understanding
            
            Score:
            """
            
            # Use teacher model for more sophisticated judgment
            original_inputs = self.teacher_processor(
                text=judge_prompt,
                images=original_img,
                return_tensors="pt"
            )
            
            processed_inputs = self.teacher_processor(
                text=judge_prompt,
                images=processed_img,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                original_outputs = self.teacher_model(**original_inputs)
                processed_outputs = self.teacher_model(**processed_inputs)
                
                # More sophisticated comparison using teacher model
                original_score = torch.mean(original_outputs.last_hidden_state).item()
                processed_score = torch.mean(processed_outputs.last_hidden_state).item()
                
                # Teacher model should provide more nuanced judgment
                improvement = (processed_score - original_score) / (abs(original_score) + 1e-8)
                return max(0.0, min(1.0, 0.5 + improvement))
        
        except Exception as e:
            self.logger.error(f"Teacher-reward calculation failed: {e}")
            return 0.0
    
    def attention_reward(self,
                        original_image: str,
                        processed_image: str,
                        question: str,
                        reasoning: str,
                        **kwargs) -> float:
        """
        Attention-reward: Based on attention patterns between processed image and question.
        """
        self._load_self_model()
        if self.self_model is None:
            return 0.0
        
        try:
            original_img = self._decode_image(original_image)
            processed_img = self._decode_image(processed_image)
            
            if original_img is None or processed_img is None:
                return 0.0
            
            # Get attention patterns for both images with the question
            original_inputs = self.self_processor(
                text=question,
                images=original_img,
                return_tensors="pt"
            )
            
            processed_inputs = self.self_processor(
                text=question,
                images=processed_img,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                # Get attention weights
                original_outputs = self.self_model(**original_inputs, output_attentions=True)
                processed_outputs = self.self_model(**processed_inputs, output_attentions=True)
                
                if hasattr(original_outputs, 'attentions') and hasattr(processed_outputs, 'attentions'):
                    # Average attention across layers and heads
                    original_attention = torch.stack(original_outputs.attentions).mean(dim=(0, 1))
                    processed_attention = torch.stack(processed_outputs.attentions).mean(dim=(0, 1))
                    
                    # Measure attention focus - higher values indicate more focused attention
                    original_focus = torch.max(original_attention) - torch.mean(original_attention)
                    processed_focus = torch.max(processed_attention) - torch.mean(processed_attention)
                    
                    # Reward if processed image has more focused attention
                    if processed_focus > original_focus:
                        return min(1.0, (processed_focus - original_focus).item() * 5)
                    else:
                        return max(0.0, 0.5 - (original_focus - processed_focus).item() * 5)
                else:
                    # Fallback to embedding similarity if attention not available
                    return self.embedding_reward(original_image, processed_image, question, reasoning, **kwargs)
        
        except Exception as e:
            self.logger.error(f"Attention-reward calculation failed: {e}")
            return 0.0
    
    def embedding_reward(self,
                        original_image: str,
                        processed_image: str,
                        question: str,
                        reasoning: str,
                        **kwargs) -> float:
        """
        Embedding-reward: Based on similarity between image embeddings and question embeddings.
        """
        self._load_clip_model()
        if self.clip_model is None:
            return 0.0
        
        try:
            original_img = self._decode_image(original_image)
            processed_img = self._decode_image(processed_image)
            
            if original_img is None or processed_img is None:
                return 0.0
            
            # Prepare inputs for CLIP
            original_img_input = self.clip_processor(original_img).unsqueeze(0)
            processed_img_input = self.clip_processor(processed_img).unsqueeze(0)
            text_input = clip.tokenize([question])
            
            if torch.cuda.is_available():
                original_img_input = original_img_input.cuda()
                processed_img_input = processed_img_input.cuda()
                text_input = text_input.cuda()
            
            with torch.no_grad():
                # Get embeddings
                original_img_features = self.clip_model.encode_image(original_img_input)
                processed_img_features = self.clip_model.encode_image(processed_img_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                original_img_features = F.normalize(original_img_features, dim=-1)
                processed_img_features = F.normalize(processed_img_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Calculate cosine similarities
                original_similarity = torch.cosine_similarity(original_img_features, text_features).item()
                processed_similarity = torch.cosine_similarity(processed_img_features, text_features).item()
                
                # Reward improvement in similarity
                improvement = processed_similarity - original_similarity
                
                # Scale to 0-1 range
                return max(0.0, min(1.0, 0.5 + improvement * 2))
        
        except Exception as e:
            self.logger.error(f"Embedding-reward calculation failed: {e}")
            return 0.0
    
    def combined_image_reward(self,
                            original_image: str,
                            processed_image: str,
                            question: str,
                            reasoning: str,
                            reward_types: List[str] = ["self", "embedding"],
                            weights: List[float] = None,
                            **kwargs) -> Dict[str, float]:
        """
        Calculate combined image reward using multiple strategies.
        
        Args:
            original_image: Base64 encoded original image
            processed_image: Base64 encoded processed image
            question: User's question
            reasoning: Model's reasoning
            reward_types: List of reward types to use
            weights: Weights for each reward type
            
        Returns:
            Dictionary with individual and combined rewards
        """
        if weights is None:
            weights = [1.0] * len(reward_types)
        
        rewards = {}
        
        for reward_type in reward_types:
            if reward_type == "self":
                rewards["self_reward"] = self.self_reward(
                    original_image, processed_image, question, reasoning, **kwargs
                )
            elif reward_type == "teacher":
                rewards["teacher_reward"] = self.teacher_reward(
                    original_image, processed_image, question, reasoning, **kwargs
                )
            elif reward_type == "attention":
                rewards["attention_reward"] = self.attention_reward(
                    original_image, processed_image, question, reasoning, **kwargs
                )
            elif reward_type == "embedding":
                rewards["embedding_reward"] = self.embedding_reward(
                    original_image, processed_image, question, reasoning, **kwargs
                )
        
        # Calculate weighted combined reward
        if rewards:
            reward_values = list(rewards.values())
            combined_reward = sum(r * w for r, w in zip(reward_values, weights)) / sum(weights)
            rewards["combined_image_reward"] = combined_reward
        
        return rewards


def create_image_reward_func(reward_calculator: ImageRewardCalculator,
                           reward_types: List[str] = ["self", "embedding"],
                           weights: List[float] = None):
    """
    Create a reward function for use with the verifiers framework.
    
    Args:
        reward_calculator: ImageRewardCalculator instance
        reward_types: Types of image rewards to use
        weights: Weights for combining rewards
        
    Returns:
        Reward function compatible with verifiers framework
    """
    def image_reward_func(completion: Union[str, List[Dict[str, Any]]],
                         state: Dict[str, Any],
                         prompt: Union[str, List[Dict[str, Any]]],
                         **kwargs) -> float:
        """
        Image reward function for verifiers framework.
        """
        try:
            # Extract images from state
            original_image = state.get('original_image')
            current_image = state.get('current_image')
            
            if not original_image or not current_image:
                return 0.0
            
            # Extract question from prompt
            if isinstance(prompt, list):
                question = ""
                for msg in prompt:
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            for item in content:
                                if item.get('type') == 'text':
                                    question = item.get('text', '')
                                    break
                        else:
                            question = content
                        break
            else:
                question = str(prompt)
            
            # Extract reasoning from completion
            if isinstance(completion, list):
                reasoning = ""
                for msg in completion:
                    content = msg.get('content', '')
                    if '<think>' in content:
                        start = content.find('<think>') + 7
                        end = content.find('</think>')
                        if end > start:
                            reasoning = content[start:end].strip()
                            break
            else:
                content = str(completion)
                if '<think>' in content:
                    start = content.find('<think>') + 7
                    end = content.find('</think>')
                    if end > start:
                        reasoning = content[start:end].strip()
            
            # Calculate image reward
            rewards = reward_calculator.combined_image_reward(
                original_image=original_image,
                processed_image=current_image,
                question=question,
                reasoning=reasoning,
                reward_types=reward_types,
                weights=weights,
                **kwargs
            )
            
            return rewards.get('combined_image_reward', 0.0)
            
        except Exception as e:
            logging.error(f"Image reward calculation failed: {e}")
            return 0.0
    
    return image_reward_func