import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import re
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModel, AutoTokenizer

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True
        return False
    
class StopOnPeriod(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated_text.endswith('.'):
            return True
        return False
    
class llama_vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    # This function is used to split Llama-3.2-90B
    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        num_layers = 100
        # GPU0: -5, GPU-1: -7
        total_cost = num_layers + 5 + 7

        # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
        num_layers_per_gpu = total_cost // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 5
        num_layers_per_gpu[-1] -= 7

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        device_map['vision_model'] = rank
        device_map['language_model.model.embed_tokens'] = rank
        device_map['language_model.model.rotary_emb'] = rank
        device_map['language_model.model.norm'] = rank + world_size * (num_gpus - 1)
        device_map['language_model.lm_head'] = rank + world_size * (num_gpus - 1)
        device_map['multi_modal_projector'] = rank + world_size * (num_gpus - 1)
        return device_map

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        if '90b' in model_path.lower():
            device_map = self.split_model()
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            ).eval()
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
            ).cuda().eval()
        
        self.reward_model = AutoModel.from_pretrained(
            "internlm/internlm-xcomposer2d5-7b-reward", 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        )
        self.reward_model = self.reward_model.to(device="cuda")
        self.reward_model.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer2d5-7b-reward", trust_remote_code=True)

        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        if 'Instruct' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=5)
        kwargs_default = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs
        self.model_name = model_path

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            for key, item in options.items():
                question += f'\n{key}. {item}'
            if '11B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i+1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['ChartQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            if '11B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'You have to think through your answer and provide a step-by-step solution. '
                    f'Once you have the solution, write the final answer in at most a few words at the end '
                    f"with the phrase \"FINAL ANSWER:\". "
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'Follow these steps carefully:\n '
                    f'Step 1: Analyze the question to understand what specific data or information is being asked for. '
                    f'Focus on whether the question is asking for a specific number or category '
                    f'from the chart image.\n '
                    f'Step 2: Identify any numbers, categories, or groups mentioned in the question '
                    f'and take note of them. Focus on detecting and matching them directly to the image. \n'
                    f'Step 3: Study the image carefully and find the relevant data corresponding to the categories '
                    f'or numbers mentioned. Avoid unnecessary assumptions or calculations; '
                    f'simply read the correct data from the image.\n '
                    f'Step 4: Develop a clear plan to solve the question by locating the right data. '
                    f'Focus only on the specific category or group that matches the question. \n'
                    f'Step 5: Use step-by-step reasoning to ensure you are referencing the correct numbers '
                    f'or data points from the image, avoiding unnecessary extra steps or interpretations.\n '
                    f"Step 6: Provide the final answer, starting with \"FINAL ANSWER:\" "
                    f'and using as few words as possible, '
                    f'simply stating the number or data point requested. \n\n '
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
    
    def xcomposer_score(self, image_path, prompt, outputs, stage_type="caption"):
        chat_bots = []
        stage_outputs = []
        if stage_type == "caption":
            prompt = prompt + "\nPlease provide a detailed description of this image based on the question and the image itself."
            pattern = r"<CAPTION>(.*?)</CAPTION>"
            for i in range(len(outputs)):
                match = re.search(pattern, outputs[i], re.DOTALL)
                if match:
                    stage_outputs.append(match.group(1))
                else:
                    stage_outputs.append(outputs[i])
        if stage_type == "reasoning":
            prompt = prompt+"\nPlease reason based on the question and the image, providing a detailed reasoning process to solve the problem."
            pattern = r"<REASONING>(.*?)</REASONING>"
            for i in range(len(outputs)):
                match = re.search(pattern, outputs[i], re.DOTALL)
                if match:
                    stage_outputs.append(match.group(1))
                else:
                    stage_outputs.append(outputs[i])
        if stage_type == "conclusion":
            prompt = prompt
            stage_outputs = outputs
            
        if stage_type == "summary+caption":
            prompt = prompt+"\nPlease provide a detailed description and a summary of this image based on the question and the image itself."
            prompt += f'Please note that a better summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
            prompt += f'Please note that a better caption should be as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
            stage_outputs = outputs

        for i in range(len(stage_outputs)):
            chat_bot = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": stage_outputs[i]}
            ]
            chat_bots.append(chat_bot)
        
        hd_num=9
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            scores = self.reward_model.get_scores(chat_bots, [[image_path] for _ in range(len(outputs))], hd_num=hd_num)
        
        return scores
    
    def xcomposer_judge(self, image_path, prompt, outputs, stage_type="caption"):
        chat_bots = []
        stage_outputs = []

        if stage_type == "summary":
            prompt = prompt + "\nPlease provide a summary of this image based on the question and the image itself."
            pattern = r"<SUMMARY>(.*?)</SUMMARY>"
            for i in range(len(outputs)):
                match = re.search(pattern, outputs[i], re.DOTALL)
                if match:
                    stage_outputs.append(match.group(1))
                else:
                    stage_outputs.append(outputs[i])
        if stage_type == "caption":
            prompt = prompt + "\nPlease provide a detailed description of this image based on the question and the image itself."
            pattern = r"<CAPTION>(.*?)</CAPTION>"
            for i in range(len(outputs)):
                match = re.search(pattern, outputs[i], re.DOTALL)
                if match:
                    stage_outputs.append(match.group(1))
                else:
                    stage_outputs.append(outputs[i])
        if stage_type == "reasoning":
            prompt = prompt+"\nPlease reason based on the question and the image, providing a detailed reasoning process to solve the problem."
            pattern = r"<REASONING>(.*?)</REASONING>"
            for i in range(len(outputs)):
                match = re.search(pattern, outputs[i], re.DOTALL)
                if match:
                    stage_outputs.append(match.group(1))
                else:
                    stage_outputs.append(outputs[i])
        if stage_type == "conclusion":
            prompt = prompt
            stage_outputs = outputs
            
        if stage_type == "summary+caption":
            prompt = prompt+"\nPlease provide a detailed description and a summary of this image based on the question and the image itself."
            prompt += f'Please note that a better summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
            prompt += f'Please note that a better caption should be as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
            stage_outputs = outputs
            
        if stage_type == "sentence":
            prompt = prompt + "\Please provide a next sentence for the answer to the question."
            stage_outputs = outputs

        for i in range(len(stage_outputs)):
            chat_bot = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": stage_outputs[i]}
            ]
            chat_bots.append(chat_bot)
        
        hd_num=9
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            scores = self.reward_model.get_scores(chat_bots, [[image_path] for _ in range(len(outputs))], hd_num=hd_num)
        rank_res = np.argsort(scores)[::-1].tolist()
        return rank_res
    
    def swires(self, message, dataset):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        if not self.use_custom_prompt(dataset):
            if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N':
                self.kwargs['max_new_tokens'] = 2048
            else:
                self.kwargs['max_new_tokens'] = 2048
        
        stages = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
        end_markers = ['</SUMMARY>', '</CAPTION>', '</REASONING>', '</CONCLUSION>']

        initial_length = len(inputs['input_ids'][0])
        
        # begin to do beam search as follow
        # generate <SUMMARY> directly
        stage = stages[0]
        end_marker = end_markers[0]
        stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])
        generation_kwargs = self.kwargs.copy()
        generation_kwargs.update({
            'stopping_criteria': stop_criteria
        })
        output = self.model.generate(**inputs, **generation_kwargs)
        generated_text = self.processor.tokenizer.decode(output[0][initial_length:], skip_special_tokens=True)
        caption_input_text = input_text+generated_text
        len_of_summary = len(generated_text)    

        # backtracking
        max_backtrack = 4
        backtrack_count = 0
        reward_mean = -0.77
        reward_std = 2.08
        Z = 0.2533
        backtrack_cutoff = reward_mean+Z*reward_std
        reason_candidates = []
        reason_partial_candidates = []
        reason_candidates_scores = []
        while backtrack_count <= max_backtrack:
            # candidate 1*4=4 choose 2 <CAPTION>
            stage = stages[1]
            end_marker = end_markers[1]
            stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])  
            candidates = []
            partial_candidates = []
            inputs = self.processor(image, caption_input_text, return_tensors='pt').to(self.device)
            for _ in range(4):
                generation_kwargs = self.kwargs.copy()
                generation_kwargs.update({
                    'stopping_criteria': stop_criteria
                })
                
                output = self.model.generate(**inputs, **generation_kwargs)
                generated_text = self.processor.tokenizer.decode(output[0][initial_length:], skip_special_tokens=True)
                
                candidates.append(generated_text)
                partial_candidates.append(generated_text[len_of_summary:])

            score_index = self.xcomposer_judge(image_path, prompt, partial_candidates, stage_type="caption")
            reason_input_texts = [input_text+candidates[score_index[_]]  for _ in range(2)]
            len_of_captions = [len_of_summary+len(partial_candidates[score_index[_]]) for _ in range(2)] 
            
            # candidate 2*2=4 choose 3 <REASONING> 
            stage = stages[2]
            end_marker = end_markers[2]
            stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])  
            generation_kwargs = self.kwargs.copy()
            generation_kwargs.update({
                'stopping_criteria': stop_criteria
            })
            for i in range(2):
                inputs = self.processor(image, reason_input_texts[i], return_tensors='pt').to(self.device)
                for _ in range(2):
                    output = self.model.generate(**inputs, **generation_kwargs)
                    generated_text = self.processor.tokenizer.decode(output[0][initial_length:], skip_special_tokens=True)
                    reason_candidates.append(generated_text)
                    reason_partial_candidates.append(generated_text[len_of_captions[i]:])
                    reason_candidates_score = self.xcomposer_score(image_path, prompt, [generated_text[len_of_captions[i]:]], stage_type="reasoning")
                    reason_candidates_scores.append(reason_candidates_score)
            # if top2 > cutoff pass the exam stop backtrack
            sorted_scores = sorted(reason_candidates_scores, reverse=True)
            if sorted_scores[1] > backtrack_cutoff:
                break
            backtrack_count += 1

        score_index = np.argsort(reason_candidates_scores)[::-1].tolist()
        conclusion_input_texts = [input_text+reason_candidates[score_index[_]]  for _ in range(3)]
        
        # candidate n*1=n choose 1 <CONCLUSION>
        stage = stages[3]
        end_marker = end_markers[3]
        stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])  
        generation_kwargs = self.kwargs.copy()
        generation_kwargs.update({
            'stopping_criteria': stop_criteria
        })
        candidates = []
        for i in range(len(conclusion_input_texts)):
            inputs = self.processor(image, conclusion_input_texts[i], return_tensors='pt').to(self.device)
            output = self.model.generate(**inputs, **generation_kwargs)
            generated_text = self.processor.tokenizer.decode(output[0][initial_length:], skip_special_tokens=True)
            
            candidates.append(generated_text)
        score_index = self.xcomposer_judge(image_path, prompt, candidates, stage_type="conclusion")
        output_text = candidates[score_index[0]]
        return output_text

    def generate_inner(self, message, dataset=None):
        return self.swires(message, dataset)