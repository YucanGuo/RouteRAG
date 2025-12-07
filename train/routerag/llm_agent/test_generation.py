import unittest
import torch
from unittest.mock import MagicMock, patch
from .generation import LLMGenerationManager, GenerationConfig
from verl import DataProto

class TestLLMGenerationManager(unittest.TestCase):
    def setUp(self):
        # 模拟tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.batch_decode.return_value = [
            "Let me search about this. <search>Python programming</search>",
            "Here's the answer: <answer>42</answer>",
            "Invalid response without tags"
        ]
        
        # 模拟配置
        self.config = GenerationConfig(
            max_turns=3,
            max_start_length=128,
            max_prompt_length=512,
            max_response_length=128,
            max_obs_length=256,
            num_gpus=1,
            search_url="http://localhost:8000/retrieve",
            topk=3
        )
        
        # 模拟actor_rollout_wg
        self.actor_rollout_wg = MagicMock()
        
        # 初始化manager
        self.manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=self.config
        )

    def test_postprocess_predictions(self):
        """测试预测结果的后处理"""
        predictions = [
            "Let me search about this. <search>Python programming</search>",
            "Here's the answer: <answer>42</answer>",
            "Invalid response without tags"
        ]
        
        actions, contents = self.manager.postprocess_predictions(predictions)
        
        # 验证结果
        self.assertEqual(actions, ['search', 'answer', None])
        self.assertEqual(contents, ['Python programming', '42', ''])

    def test_execute_predictions(self):
        """测试预测执行和检索处理"""
        predictions = [
            "<search>Python programming</search>",
            "<answer>42</answer>",
            "Invalid response"
        ]
        active_mask = torch.tensor([True, True, True])
        
        # Mock检索结果
        mock_search_result = {
            'result': [
                [{'document': {'contents': 'Title 1\nPython content'}}]
            ]
        }
        
        with patch.object(self.manager, '_batch_search', return_value=mock_search_result):
            next_obs, dones, valid_action, is_search = self.manager.execute_predictions(
                predictions, 
                self.tokenizer.pad_token, 
                active_mask
            )
        
        # 验证结果
        self.assertTrue('<information>' in next_obs[0])  # 检索结果包含在observation中
        self.assertTrue(dones[1])  # answer动作应该结束对话
        self.assertFalse(valid_action[2])  # 无效响应

    def test_batch_search(self):
        """测试批量检索功能"""
        queries = ["Python programming", "Machine learning"]
        
        # Mock检索结果
        mock_result = {
            'result': [
                [
                    {'document': {'contents': 'Title 1\nPython content'}},
                    {'document': {'contents': 'Title 2\nMore Python content'}}
                ],
                [
                    {'document': {'contents': 'Title 3\nML content'}},
                    {'document': {'contents': 'Title 4\nMore ML content'}}
                ]
            ]
        }
        
        with patch.object(self.manager, '_batch_search', return_value=mock_result):
            results = self.manager.batch_search(queries)
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertTrue('Title 1' in results[0])
        self.assertTrue('Title 3' in results[1])

    def test_postprocess_responses(self):
        """测试模型响应的后处理"""
        responses = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Mock tokenizer的batch_decode结果
        self.tokenizer.batch_decode.return_value = [
            "Let me search <search>query</search> extra text",
            "The answer is <answer>42</answer> extra text",
            "Just some text"
        ]
        
        # Mock batch_tokenize结果
        self.tokenizer.return_value = {'input_ids': torch.tensor([[1, 2], [3, 4], [5, 6]])}
        
        responses_processed, responses_str = self.manager.postprocess_responses(responses)
        
        # 验证结果
        self.assertEqual(responses_str[0], "Let me search <search>query</search>")
        self.assertEqual(responses_str[1], "The answer is <answer>42</answer>")

if __name__ == '__main__':
    unittest.main() 