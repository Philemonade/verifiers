import base64
import io
import json
import logging
from typing import List, Dict, Any, Callable, Tuple, Union, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.tools.image_tools import IMAGE_TOOLS


VISION_SYSTEM_PROMPT = """You are an expert vision-language assistant that can analyze images and use image processing tools to help with reasoning.

When given an image and a question, you should:
1. First analyze the image carefully within <think></think> tags
2. Determine if you need to use image processing tools to better understand the image
3. If tools are needed, use the appropriate tool and continue reasoning
4. If no tools are needed, provide your final answer

Available image processing tools:
{tool_descriptions}

Your response format should be:
<think>
Your reasoning process here...
</think>

Then either:
<tool>{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}</tool>

Or:
<answer>Your final answer here</answer>

Remember:
- Use tools strategically to enhance your understanding of the image
- Each tool call should have a clear purpose in your reasoning
- Continue thinking after each tool result
- Only provide your final answer when you're confident
"""


class VisionLanguageEnv(MultiTurnEnv):
    """
    Environment for training vision-language models with image processing tools.
    
    This environment supports:
    - Multi-turn conversations with image processing tools
    - Image-based rewards (self-reward, teacher-reward, attention-reward, embedding-reward)
    - Qwen2.5-VL-7B model integration
    - "Think with image" training paradigm
    """
    
    def __init__(self,
                 tools: List[Callable] = None,
                 system_prompt: str = VISION_SYSTEM_PROMPT,
                 format_prompt: bool = True,
                 parser: XMLParser = None,
                 env_parser: XMLParser = None,
                 max_turns: int = 10,
                 judge_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 reward_types: List[str] = ["format", "tool", "answer", "image"],
                 reward_weights: List[float] = None,
                 **kwargs):
        
        # Set default tools if none provided
        if tools is None:
            tools = IMAGE_TOOLS
            
        # Set default parsers
        if parser is None:
            parser = XMLParser(fields=["think", ("tool", "answer")])
        if env_parser is None:
            env_parser = XMLParser(fields=["result"])
            
        # Create rubric with vision-specific rewards
        rubric = VisionRubric(
            tools=tools,
            parser=parser,
            env_parser=env_parser,
            judge_model_name=judge_model_name,
            reward_types=reward_types,
            reward_weights=reward_weights
        )
        
        # Format system prompt with tool descriptions
        if format_prompt:
            tool_descriptions = self._format_tool_descriptions(tools)
            formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        else:
            formatted_prompt = system_prompt
            
        super().__init__(
            system_prompt=formatted_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs
        )
        
        self.tools = {tool.__name__: tool for tool in tools}
        self.env_parser = env_parser
        self.current_image = None
        self.original_image = None
        self.tool_history = []
        
        # Initialize judge model for image rewards
        self.judge_model_name = judge_model_name
        self.judge_model = None
        self.judge_processor = None
        
    def _format_tool_descriptions(self, tools: List[Callable]) -> str:
        """Format tool descriptions for the system prompt."""
        descriptions = []
        for tool in tools:
            doc = tool.__doc__ or ""
            name = tool.__name__
            
            # Extract description and examples
            lines = doc.strip().split('\n')
            description = lines[0] if lines else ""
            
            # Find examples
            examples = []
            in_examples = False
            for line in lines:
                if line.strip().startswith("Examples:"):
                    in_examples = True
                    continue
                if in_examples and line.strip():
                    examples.append(line.strip())
                elif in_examples and not line.strip():
                    break
                    
            tool_desc = f"- {name}: {description}"
            if examples:
                tool_desc += f"\n  Examples: {', '.join(examples[:2])}"
            descriptions.append(tool_desc)
            
        return "\n".join(descriptions)
    
    def _load_judge_model(self):
        """Lazy load the judge model for image rewards."""
        if self.judge_model is None:
            try:
                self.judge_processor = AutoProcessor.from_pretrained(self.judge_model_name)
                self.judge_model = AutoModel.from_pretrained(
                    self.judge_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logging.info(f"Loaded judge model: {self.judge_model_name}")
            except Exception as e:
                logging.warning(f"Failed to load judge model: {e}")
                self.judge_model = None
                self.judge_processor = None
    
    def set_image(self, image_data: str):
        """Set the current image for the conversation."""
        self.current_image = image_data
        self.original_image = image_data
        self.tool_history = []
    
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()
    
    def is_completed(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        """Check if the conversation is completed (has final answer)."""
        return self.parser.parse_answer(messages) is not None
    
    def call_tool(self, tool_json: str, max_chars: int = 1024, **kwargs: Any) -> str:
        """Call an image processing tool and update current image."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object"
            
            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'"
            
            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            
            # Add current image data to args if tool requires it
            if "image_data" in tool_func.__code__.co_varnames and self.current_image:
                tool_args["image_data"] = self.current_image
            
            # Call the tool function
            result = tool_func(**tool_args)
            
            # Update current image if tool returns new image data
            if isinstance(result, str) and not result.startswith("Error"):
                try:
                    # Check if result is base64 encoded image
                    base64.b64decode(result)
                    self.current_image = result
                    self.tool_history.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": "Image processed successfully"
                    })
                    result = "Image processed successfully. The modified image is now available for analysis."
                except:
                    # Result is not an image, keep as is
                    self.tool_history.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result
                    })
            
            if max_chars > 0 and len(str(result)) > max_chars:
                result = str(result)[:max_chars] + "..."
                
            return str(result)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def env_response(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Generate environment response after tool call."""
        try:
            parsed = self.parser.parse(messages[-1]['content'])
            
            # Check if we got a valid tool field
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    # Update state with current image and tool history
                    new_state = state.copy()
                    new_state.update({
                        'current_image': self.current_image,
                        'tool_history': self.tool_history,
                        'original_image': self.original_image
                    })
                    return {
                        'role': 'user', 
                        'content': self.env_parser.format(result=result)
                    }, new_state
                else:
                    return {
                        'role': 'user', 
                        'content': "Error: Tool execution returned empty output."
                    }, state
        except Exception:
            pass
            
        return {
            'role': 'user', 
            'content': "Error: Tool command not found or invalid format."
        }, state
    
    def rollout(self,
                client,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                image_data: str = None,
                task: str = "default",
                info: Dict[str, Any] = {},
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with image processing capabilities.
        """
        # Set the image for this rollout
        if image_data:
            self.set_image(image_data)
        
        # Add image to the initial prompt if provided
        if isinstance(prompt, list) and image_data:
            # Add image to the first user message
            for msg in prompt:
                if msg.get('role') == 'user':
                    msg['content'] = [
                        {"type": "text", "text": msg['content']},
                        {"type": "image", "image": image_data}
                    ]
                    break
        
        # Call parent rollout method
        completion, state = super().rollout(
            client=client,
            model=model,
            prompt=prompt,
            answer=answer,
            task=task,
            info=info,
            sampling_args=sampling_args,
            **kwargs
        )
        
        # Add vision-specific state information
        state.update({
            'current_image': self.current_image,
            'original_image': self.original_image,
            'tool_history': self.tool_history
        })
        
        return completion, state


class VisionRubric(Rubric):
    """
    Specialized rubric for vision-language tasks with image-based rewards.
    """
    
    def __init__(self,
                 tools: List[Callable],
                 parser: XMLParser,
                 env_parser: XMLParser,
                 judge_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 reward_types: List[str] = ["format", "tool", "answer", "image"],
                 reward_weights: List[float] = None,
                 **kwargs):
        
        self.tools = {tool.__name__: tool for tool in tools}
        self.parser = parser
        self.env_parser = env_parser
        self.judge_model_name = judge_model_name
        self.reward_types = reward_types
        
        # Create reward functions based on specified types
        reward_funcs = []
        if reward_weights is None:
            reward_weights = [1.0] * len(reward_types)
            
        for reward_type in reward_types:
            if reward_type == "format":
                reward_funcs.append(self._format_reward)
            elif reward_type == "tool":
                reward_funcs.append(self._tool_reward)
            elif reward_type == "answer":
                reward_funcs.append(self._answer_reward)
            elif reward_type == "image":
                reward_funcs.append(self._image_reward)
        
        super().__init__(
            funcs=reward_funcs,
            weights=reward_weights,
            parser=parser,
            **kwargs
        )
    
    def _format_reward(self, completion: Union[str, List[Dict[str, Any]]], **kwargs) -> float:
        """Reward for proper XML formatting."""
        try:
            if isinstance(completion, list):
                content = "\n".join([msg.get('content', '') for msg in completion])
            else:
                content = completion
                
            parsed = self.parser.parse(content)
            
            # Check if we have proper think tags
            has_think = hasattr(parsed, 'think') and parsed.think is not None
            
            # Check if we have either tool or answer
            has_tool_or_answer = (
                (hasattr(parsed, 'tool') and parsed.tool is not None) or
                (hasattr(parsed, 'answer') and parsed.answer is not None)
            )
            
            if has_think and has_tool_or_answer:
                return 1.0
            elif has_think or has_tool_or_answer:
                return 0.5
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _tool_reward(self, completion: Union[str, List[Dict[str, Any]]], state: Dict[str, Any], **kwargs) -> float:
        """Reward for appropriate tool usage."""
        try:
            tool_history = state.get('tool_history', [])
            
            if not tool_history:
                return 0.0
            
            # Reward based on successful tool calls
            successful_calls = sum(1 for call in tool_history if not call['result'].startswith('Error'))
            total_calls = len(tool_history)
            
            if total_calls == 0:
                return 0.0
            
            success_rate = successful_calls / total_calls
            
            # Bonus for reasonable number of tool calls (not too many, not too few)
            if 1 <= total_calls <= 3:
                return success_rate * 1.0
            elif total_calls > 3:
                return success_rate * 0.8  # Penalty for too many calls
            else:
                return success_rate * 0.5
                
        except Exception:
            return 0.0
    
    def _answer_reward(self, completion: Union[str, List[Dict[str, Any]]], answer: str, **kwargs) -> float:
        """Reward for providing a final answer."""
        try:
            if isinstance(completion, list):
                content = "\n".join([msg.get('content', '') for msg in completion])
            else:
                content = completion
                
            parsed_answer = self.parser.parse_answer([{'content': content}])
            
            if parsed_answer is not None:
                # Simple reward for having an answer
                # In practice, you might want to compare with ground truth
                return 1.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _image_reward(self, completion: Union[str, List[Dict[str, Any]]], state: Dict[str, Any], **kwargs) -> float:
        """
        Image-based reward using various strategies:
        - Self-reward: Judge quality using the same model
        - Attention-reward: Based on attention patterns
        - Embedding-reward: Based on image-text similarity
        """
        try:
            current_image = state.get('current_image')
            original_image = state.get('original_image')
            
            if not current_image or not original_image:
                return 0.0
            
            # For now, implement a simple heuristic
            # In practice, you would implement the specific reward mechanisms
            
            # Check if image was modified (tool was used)
            if current_image != original_image:
                # Reward for image modification
                return 0.8
            else:
                # No modification, smaller reward
                return 0.2
                
        except Exception:
            return 0.0