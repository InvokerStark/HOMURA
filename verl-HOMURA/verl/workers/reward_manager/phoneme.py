from pyphen import Pyphen
import re
import math

from collections import defaultdict
import torch
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
import openai
import sacrebleu
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftlangdetect import detect
from verl.workers.reward_manager.syllable_calculation import cal_syllable_count
from json_repair import repair_json


import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from transformers import AutoModelForCausalLM, AutoTokenizer

class PhonemeRewardManager:
    """The reward manager."""

    def __init__(self, config, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.config = config

        self.all_pinyin_list = "ba bo bai bei bao ban ben bang beng bi bie biao bian bin bing bu pa po pai pao pou pan pen pei pang peng pi pie piao pian pin ping pu ma mo me mai mao mou man men mei mang meng mi mie miao miu mian min ming mu fa fo fei fou fan fen fang feng fu da de dai dei dao dou dan dang den deng di die diao diu dian ding dong du duan dun dui duo ta te tai tao tou tan tang teng ti tie tiao tian ting tong tu tuan tun tui tuo na nai nei nao ne nen nan nang neng ni nie niao niu nian nin niang ning nong nou nu nuan nun nuo nü nüe la le lo lai lei lao lou lan lang leng li lia lie liao liu lian lin liang ling long lu luo lou luan lun lü lüe ga ge gai gei gao gou gan gen gang geng gong gu gua guai guan guang gui guo ka ke kai kao kou kan ken kang keng kong ku kua kuai kuan kuang kui kun kuo ha he hai han hei hao hou hen hang heng hong hu hua huai huan hui huo hun huang ji jia jie jiao jiu jian jin jiang jing jiong ju juan jun jue qi qia qie qiao qiu qian qin qiang qing qiong qu quan qun que xi xia xie xiao xiu xian xin xiang xing xiong xu xuan xun xue zha zhe zhi zhai zhao zhou zhan zhen zhang zheng zhong zhu zhua zhuai zhuan zhuang zhun zhui zhuo cha che chi chai chao chou chan chen chang cheng chong chu chua chuai chuan chuang chun chui chuo sha she shi shai shao shou shan shen shang sheng shu shua shuai shuan shuang shun shui shuo re ri rao rou ran ren rang reng rong ru rui ruan run ruo za ze zi zai zao zan zou zang zei zen zeng zong zu zuan zun zui zuo ca ce ci cai cao cou can cen cang ceng cong cu cuan cun cui cuo sa se si sai sao sou san sen sang seng song su suan sun sui suo ya yao you yan yang yu ye yue yuan yi yin yun ying yo yong wa wo wai wei wan wen wang weng wu".split()
        self.all_pinyin_list = sorted(self.all_pinyin_list, key=lambda x: len(x), reverse=True)


        self.embedding_tokenizer = AutoTokenizer.from_pretrained('/mnt/group/opensource_models/Qwen3-Embedding-0.6B', padding_side='left')
        self.embedding_model = AutoModel.from_pretrained('/mnt/group/opensource_models/Qwen3-Embedding-0.6B')
        # self.use_ppl = False

        # if self.use_ppl:
        #     self.lang_model = AutoModelForCausalLM.from_pretrained("/mnt/group/opensource_models/gpt2")
        #     self.lang_tokenizer = AutoTokenizer.from_pretrained("/mnt/group/opensource_models/gpt2")
        
        # 设置模型到合适的设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.lang_model = self.lang_model.to(device)
        # self.lang_model.eval()  # 设置为评估模式
    
    def is_sementic_similar(self, text1: str, text2: str) -> bool:
        def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


        # def get_detailed_instruct(task_description: str, query: str) -> str:
        #     return f'Instruct: {task_description}\nQuery:{query}'
        # task = 'Given a chinese sentence, retrieve relevant english translation.'
        queries = [
            text1
        ]

        documents = [
            text2
        ]

        max_length = 8192
        input_texts = queries + documents
        # Tokenize the input texts
        batch_dict = self.embedding_tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.embedding_model.device)
        outputs = self.embedding_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1) 

        scores = (embeddings[:1] @ embeddings[1:].T).tolist()
        score = scores[0][0]
        is_similar = score > 0.75  # Adjust the threshold as needed

        return int(is_similar), score


    def _process_single_sample(self, args):
        """处理单个样本的函数，用于并行处理"""
        i, data_item = args
        
        input_text = data_item.non_tensor_batch["input_dic"]['current_text']
        prompt_ids = data_item.batch["prompts"]

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        up_context = data_item.non_tensor_batch["input_dic"]["up_context"]
        down_context = data_item.non_tensor_batch["input_dic"]["down_context"]
        current_duration = data_item.non_tensor_batch["input_dic"]["duration"]

        lexy = data_item.non_tensor_batch["input_dic"].get("lexy", {})
        lexy = json.loads(lexy) if isinstance(lexy, str) else lexy

        target_language = data_item.non_tensor_batch["input_dic"]['target_language']


        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        eos_token = self.tokenizer.eos_token
        if response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]
        
        data_source = data_item.non_tensor_batch[self.reward_fn_key]

        extra_info = data_item.non_tensor_batch.get("extra_info", None)
        
        extra_info["response_id"] = valid_response_ids
        if "token_upper" not in extra_info:
            extra_info["token_upper"] = 32768
            
        try:
            json_response_str = json.loads(response_str)
            extract_response_str = json_response_str.get("translated_text", response_str)
        except Exception as e:

            return {
                'index': i,
                'valid_response_length': valid_response_length,
                'score': -1,
                'data_source': "json_error",
                'error': 1,
                'llm_response': "",
                'back_trans': "",
                'prompt_str': prompt_str,
                'response_str': response_str,
                'BLEU_score': 0,
                'BLEU_obj': "",
                'v3_quality': 0,
                'phoneme_score': 0,
                'phoneme_ratio': 0,
                'input_text': input_text,
                'reason': "",
                'lang_score': 0,
                'en_speed': 0,
            }
        
        lang_score = 1
        if target_language == 'ja':
            if not self.is_japanese_by_v3(extract_response_str, input_text=input_text):
                lang_score = 0
        else:
            if self.contains_chinese(extract_response_str):
                # print(f"contains chinese #### text: {extract_response_str} #### context: {context} #### input_text: {input_text}")
                print(f"contains chinese #### text: {extract_response_str}")
                lang_score = 0
        # else:
        #     lang_score = self.is_english_by_v3(extract_response_str)

        # if lexy:
        #     trans_in_extract_response_str = any([trans.strip().lower() in extract_response_str.lower() for noun, trans in lexy.items()])
        #     lexy_score = int(trans_in_extract_response_str)
        # else:
        #     lexy_score = 1

        en_phoneme_num = self.get_text_phoneme_num(extract_response_str, target_language)
        en_speed = en_phoneme_num / current_duration if current_duration > 0 else 0
        

        result_phoneme, phoneme_ratio = self.get_phoneme_ratio(
            source_text = input_text,
            translation_text = extract_response_str,
            en_speed = en_speed,
            target_language = target_language
        )


        back_trans = self.get_v3_back_trans(extract_response_str, up_context, down_context, target_language)
        
        sementic_res, sementic_score = self.is_sementic_similar(input_text, back_trans)

        sementic_score_clip = min(sementic_score, 0.8) / 0.8

        v3_quality, reason = self.get_v3_quality(
            translation_text=extract_response_str, 
            origin_text=input_text, 
            context=up_context + input_text + down_context,
            target_language=target_language
        )

        
        result = (sementic_score_clip + result_phoneme + v3_quality) / 3 + 0.5 * (lang_score - 1)


        score: float
        llm_response = ""
        if isinstance(result, dict):
            score = result["score"]
            retrun_extra_info = result["extra_info"]
            llm_response = retrun_extra_info.get("llm_response", "")
        else:
            score = result
        
        return {
            'index': i,
            'valid_response_length': valid_response_length,
            'score': score,
            'data_source': data_source,
            'error': 0,
            'llm_response': llm_response,
            'back_trans': back_trans,
            'prompt_str': prompt_str,
            'response_str': response_str,
            'BLEU_score': sementic_res,
            'BLEU_obj': sementic_score,
            'phoneme_score': result_phoneme,
            'phoneme_ratio': phoneme_ratio,
            'v3_quality': v3_quality,
            'input_text': input_text,
            'reason': reason,
            'lang_score': lang_score,
            'en_speed': en_speed,
        }
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # 准备并行处理的参数
        sample_datas = [(i, data[i]) for i in range(len(data))]
        
        # 使用线程池进行并行处理
        max_workers = min(16, len(data))  # 限制最大线程数，减少并发冲突
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(self._process_single_sample, args): args[0] for args in sample_datas}
            
            # 收集结果
            results = {}
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results[result['index']] = result
                except Exception as e:
                    print(f"处理样本时发生错误: {e} ### future: {future}")
                    # 如果处理失败，设置默认值
                    index = future_to_index[future]
                    results[index] = {
                        'index': index,
                        'valid_response_length': 1,
                        'score': 0.0,
                        'data_source': 'unknown',
                        'error': 1,
                        'llm_response': "",
                        'back_trans': "",
                        'prompt_str': '',
                        'response_str': '',
                        'BLEU_score': 0.0,
                        'BLEU_obj': "",
                        'v3_quality': 0.0,
                        'phoneme_score': 0.0,
                        'phoneme_ratio': 0.0,
                        'input_text': '',
                        "reason": "",
                        'lang_score': 0,
                        'en_speed': 0

                    }
        
        # 按顺序处理结果
        for i in range(len(data)):
            if i not in results:
                continue
                
            result = results[i]
            
            # 设置reward tensor
            reward_tensor[i, result['valid_response_length'] - 1] = result['score']
            
            # 收集extra info
            reward_extra_info["data_source"].append(result['data_source'])
            reward_extra_info["error"].append(result['error'])
            reward_extra_info["BLEU_score"].append(result['BLEU_score'])
            reward_extra_info["phoneme_ratio"].append(result['phoneme_ratio'])
            reward_extra_info["phoneme_score"].append(result['phoneme_score'])
            reward_extra_info["v3_quality"].append(result['v3_quality'])
            reward_extra_info["llm_response"].append(result['llm_response'])
            reward_extra_info["back_trans"].append(result['back_trans'])
            reward_extra_info["prompt_str"].append(result['prompt_str'])
            reward_extra_info["response_str"].append(result['response_str'])
            reward_extra_info["input_text"].append(result['input_text'])
            reward_extra_info["lang_score"].append(result['lang_score'])
            reward_extra_info["en_speed"].append(result['en_speed'])

            
            if result['llm_response'] is not None:
                reward_extra_info["llm_response"].append(result['llm_response'])
            
            # 处理打印逻辑
            data_source = result['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("===========[data source]", data_source, "===========")
                print("[error]", result['error'])
                print("[prompt]", result['prompt_str'])
                print("[response]", result['response_str'])
                print("[back_trans]", result['back_trans'])
                print("[input_text]", result['input_text'])
                print("[score]", result['score'])
                print("[BLEU_score]", result['BLEU_score'])
                print("[BLEU_obj]", result['BLEU_obj'])
                print("[phoneme_score]", result['phoneme_score'])
                print("[phoneme_ratio]", result['phoneme_ratio'])
                print("[v3_quality]", result['v3_quality']),
                print("[reason]", result['reason'])
                print("[lang_score]", result['lang_score'])
                print("[en_speed]", result['en_speed'])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    # 计算因素比
    def get_phoneme_ratio(self, source_text, translation_text, en_speed, target_language='en'):



        mean_length = 12
        source_len = len(source_text)
        # 从配置中读取参数，如果没有配置则使用默认值
        left_bound = self.config.reward_model.get("phoneme_left_bound", 0.8)
        right_bound = self.config.reward_model.get("phoneme_right_bound", 0.9)
        source_phoneme_num = self.get_text_phoneme_num(source_text, 'zh')

        translation_phoneme_num = self.get_text_phoneme_num(translation_text, target_language)

        # set min phoneme num to 1
        if source_phoneme_num <= 0:
            source_phoneme_num = 1
        
        phoneme_ratio = translation_phoneme_num / source_phoneme_num

        # if en_speed > 6:
        #     return 0, phoneme_ratio

        def calculate_r(x: float, y: float = 14 ) -> float:
            
            normalized = x / y
            
            exponent = 0.5 
            r = 0.4 + 0.5 * (normalized ** exponent)
            
            return r

        def calculate_distance(x: float, left_bound: float, right_bound: float) -> float:
            distance = max(abs(x - left_bound), abs(x - right_bound))

            return math.exp(-100*(distance ** 2))
        
        if source_len > 4:

            if source_len < mean_length:
                if phoneme_ratio <= right_bound and phoneme_ratio >= left_bound * calculate_r(source_len, mean_length):
                    return 1, phoneme_ratio
                else:
                    phoneme_score = calculate_distance(phoneme_ratio, left_bound * calculate_r(source_len, mean_length), right_bound)
                    return max(0, phoneme_score), phoneme_ratio

            if phoneme_ratio <= right_bound and phoneme_ratio >= left_bound:
                return 1, phoneme_ratio
            else:
                phoneme_score = calculate_distance(phoneme_ratio, left_bound, right_bound)
                return max(0, phoneme_score), phoneme_ratio
        else:
            if phoneme_ratio <= (right_bound * 1.4) and phoneme_ratio >= (left_bound * 0.6):
                    return 1, phoneme_ratio
            else:
                phoneme_score = calculate_distance(phoneme_ratio, left_bound * 0.6, right_bound * 1.4)
                return max(0, phoneme_score), phoneme_ratio

    # 计算文本的音节数
    def get_text_phoneme_num(self, text, lang='zh'):
        """
        使用 syllable_calculation 模块计算音节数
        :param text: 要计算的文本
        :param lang: 语言代码 ('zh', 'en', 'ja', 'de', 'fr', 'es', 'ko')
        :return: 音节数
        """
        return cal_syllable_count(text, lang)



    def get_v31_response(self, prompt):
        client = openai.OpenAI(
            base_url="",
            api_key="",
        )
        try:
            resp = client.chat.completions.create(
                model="deepseek-v3.1-250821",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"safety": {'input_level': 'none'}}
            )

            if resp and hasattr(resp, "choices") and resp.choices and len(resp.choices) > 0:
                if hasattr(resp.choices[0].message, "content") and resp.choices[0].message.content:
                    resp_str = resp.choices[0].message.content
            else:
                resp_str = ""
        except Exception as e:
            print(f"v3.1 error: {e}")
            resp_str = ""

        return resp_str

    def get_v3_response(self, prompt):
        client = openai.OpenAI(
            base_url="",
            api_key="",
        )

        try:
            resp = client.chat.completions.create(
                model='baidu/deepseek-v3',
                messages=[{"role": "user", "content": prompt}],
                extra_body={"safety": {"input_level": "none"}},
                )
            # print(f"deepseek_request resp: {resp}")
            if resp and hasattr(resp, "choices") and resp.choices and len(resp.choices) > 0:
                if hasattr(resp.choices[0].message, "content") and resp.choices[0].message.content:
                    resp_str = resp.choices[0].message.content
            else:
                resp_str = ""
        except Exception as e:
            print(f"v3 error: {e}")
            resp_str = ""

        return resp_str


    def get_v3_back_trans(self, text, up_context, down_context, target_language="en"):
        """
        将翻译后的文本反向翻译回中文，用于检查语义相似度
        :param text: 翻译后的文本
        :param up_context: 上文
        :param down_context: 下文
        :param target_language: 目标语言代码，如 "en", "ja", "es" 等
        :return: 反向翻译的中文文本
        """
        import os
        import json
        import hashlib
        import fcntl
        import threading

        # 语言映射
        language_name_map = {
            "en": "英文",
            "ja": "日文",
            "ko": "韩文",
            "es": "西班牙文",
            "fr": "法文",
            "de": "德文"
        }
        target_lang_name = language_name_map.get(target_language, "英文")

        cache_file = "back_trans_cache.json"
        cache_key = hashlib.md5(f"{text}_{up_context}_{down_context}_{target_language}".encode()).hexdigest()
        
        # 使用文件锁保护缓存读写
        lock_file = cache_file + ".lock"
        
        # 检查缓存是否已存在
        cache_data = {}
        if os.path.exists(cache_file):
            try:
                with open(lock_file, 'w') as lock_f:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # 独占锁
                    try:
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                    finally:
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"读取缓存文件错误: {e}")
                cache_data = {}
        
        # 检查缓存是否命中
        if cache_key in cache_data:
            return cache_data[cache_key]
        
        PROMPT_TEMPLATE_BACK_TRANS = """请你将输入的{source_lang}视频音频内容翻译成中文，输入为一个json格式，包含context和current_text。

context：包含当前文案以及其上下文信息，帮助你理解，不需要翻译。
current_text：需要翻译的{source_lang}内容。

格式示例如下，你只需要对当前文案进行翻译，上下文仅为你提供参考信息:
输入: {{"context": "xxxxxxxxx", "current_text": "xxx"}}
输出: "xxxxx"

现在请根据上述要求完成如下视频音频内容的翻译，输出翻译结果，直接返回中文，不要进行任何解释。
输入: {{"context": {context}, "current_text": {current_text}}}
输出:"""
        # 将 context 中和 input_text 一样的部分进行替换
        # context = context.replace(input_text, text)
        context = f"{up_context} {text} {down_context}"
        prompt = PROMPT_TEMPLATE_BACK_TRANS.format(
            source_lang=target_lang_name,
            context=context, 
            current_text=text
        )
        try:
            resp = self.get_v3_response(prompt)
        except Exception as e:
            print(f"get_v3_back_trans error: {e}")
            resp = ""

        # 写入缓存时使用文件锁
        try:
            with open(lock_file, 'w') as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # 独占锁
                try:
                    # 重新读取最新缓存数据
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                    else:
                        cache_data = {}
                    
                    # 更新缓存
                    cache_data[cache_key] = resp
                    
                    # 写入缓存文件
                    with open(cache_file, "w") as f:
                        json.dump(cache_data, f)
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)  # 释放锁
        except Exception as e:
            print(f"写入缓存文件错误: {e}")
        
        return resp

    def is_target_lang_by_v3(self, translated_text, target_language="en"):
        """
        判断翻译文本是否是目标语言
        :param translated_text: 翻译后的文本
        :param target_language: 目标语言代码，如 "en", "ja", "es" 等
        :return: 1 表示是目标语言，0 表示不是
        """
        import os
        import json
        import hashlib
        import fcntl

        # 语言映射
        language_name_map = {
            "en": "英文",
            "ja": "日文",
            "ko": "韩文",
            "es": "西班牙文",
            "fr": "法文",
            "de": "德文"
        }
        target_lang_name = language_name_map.get(target_language, "英文")

        cache_file = "lang_check_cache.json"
        cache_key = hashlib.md5(f"{translated_text}_{target_language}".encode()).hexdigest()

        lock_file = cache_file + ".lock"

        cache_data = {}
        if os.path.exists(cache_file):
            try:
                with open(lock_file, 'w') as lock_f:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # 独占锁
                    try:
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                    finally:
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)  # 释放锁
            except Exception as e:
                print(f"读取语言检查缓存文件错误: {e}")
                cache_data = {}

        if cache_key in cache_data:
            return cache_data[cache_key]

        prompt = """你是一个语种判断专家，你需要根据给定的文字判断是否为指定语种。这里有一段文本<text>，请判断<text>是否是一段{target_lang}文本。如果是，请返回<<1>>；否则返回<<0>>
格式为:
[think]: xxxxx
[final response]: <<1>>

<text>: {_text}"""
        resp = self.get_v3_response(prompt.format(target_lang=target_lang_name, _text=translated_text))

        if "[final response]: <<1>>" in resp:
            resp = 1
        else:
            resp = 0

        try:
            with open(lock_file, 'w') as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)  # 独占锁
                try:
                    # 重新读取最新缓存数据
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                    else:
                        cache_data = {}
                    
                    # 更新缓存
                    cache_data[cache_key] = resp
                    # 写入缓存文件
                    with open(cache_file, "w") as f:
                        json.dump(cache_data, f)
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)  # 释放锁
        except Exception as e:
            print(f"写入翻译质量缓存文件错误: {e}")

        return resp


    def is_japanese_by_v3(self, extract_response_str, input_text):
        import os
        import json
        import hashlib
        import fcntl

        # prompt = """这里有一段文本<en_text>，我需要你帮我判断一下<en_text>是否是英文文本（可以包含音译）。如果是，请返回<<1>>；否则返回<<0>>\n格式为:[final response]: <<1>>\n<en_text>: {_en_text}"""
        # prompt = """检查以下日语文本是否包含（专有名词的）中文未翻译词，非日语中本土化的表达。如果包含，请返回<<0>>；否则返回<<1>>\n格式为:[think]: xxxxx\n[final response]: <<1>>\n<text>: {_jp_text}"""

        prompt = """你是一个日语为母语的翻译专家，检查以下中文翻译日语的文本的翻译质量。判断译文中是否包含不合法的中文未翻译词，非日语中本土化的表达。
注意
- 若中文未翻译内容，也是日语中本土化表达可接受的，则不认为错误。
- 对于人名，英文专有名词，需要额外判断是否可以接受原样不翻译.

中文原文: {input_text}
日语: {translation}

输出格式为{{"是否包含":[0/1], "不合法表达":["xxx", "xxxxx"...]}}"""

        def is_chinese_char(char):
            return '\u4e00' <= char <= '\u9fff'

        try:
            resp = self.get_v3_response(prompt.format(input_text=input_text, translation=extract_response_str))
        except Exception as e:
            print(f"get_v3_language_results error: {e}")
            resp = ""
        
        # 解析返回结果
        try:
            resp = repair_json(resp)
            resp_json = json.loads(resp)
            result = resp_json.get("是否包含", 0)
            illegal_expressions = resp_json.get("不合法表达", [])

            illegal_expressions_ch = []
            for expr in illegal_expressions:
                if all(is_chinese_char(c) for c in expr):
                    illegal_expressions_ch.append(expr)
            if illegal_expressions_ch:
                resp = 0
            else:
                resp = 1
            
        except Exception as e:
            print(f"get_v3_language_results 解析响应失败: {e}, 原始响应: {resp}")
            resp = 0

        return resp

    def word_contain_pinyin(self, pinyin):
        """
        Check if the given Pinyin is valid.
        :param pinyin: The Pinyin string to check.
        :return: True if valid, False otherwise.
        """

        pinyin = pinyin.strip().lower()
        
        for valid_pinyin in self.all_pinyin_list:

            if pinyin.replace(" ", ""):
                
                pinyin = pinyin.replace(valid_pinyin, " ")


        pinyin = pinyin.replace(" ", "")

        if pinyin:
            return False
        
            
        return True
    
    def sent_contain_pinyin(self, text):

        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip().lower()

        words = text.split()
        pinyin_cnt = 0
        for word in words:

            if self.word_contain_pinyin(word):

                pinyin_cnt += 1

        if pinyin_cnt > 3:
            return True

        return False

    def get_v3_number_rewrite(self, en_text):
    # 如果文本中包含数字，判断数字是否被正确翻译
        if re.search(r'\d', en_text):
            tpl = """请将给定的英文文本中的数字，按照英文读法改成成英文单词形式，其他内容不变。
输入: {input_text}
请直接给到改写后的文本，不要任何解释。
输出: """
            prompt = tpl.format(input_text=en_text)
            try:
                resp = self.get_v3_response(prompt)
                return resp.strip()
            except Exception as e:
                print(f"get_v3_number_rewrite error: {e}")
                return en_text
        else:
            return en_text
    
        
    def get_v3_quality(self, origin_text, context, translation_text, target_language="en"):
        """
        获取翻译质量
        :param origin_text: 原文
        :param context: 上下文
        :param translation_text: 翻译文本
        :param target_language: 目标语言代码，如 "en", "ja", "es" 等
        :return: (quality_score, reason) - quality_score为1表示质量好，0表示质量不好
        """

        # 语言映射
        language_map = {
            "en": "英文",
            "ja": "日文",
            "ko": "韩文",
            "es": "西班牙文",
            "fr": "法文",
            "de": "德文"
        }
        target_lang = language_map.get(target_language, "英文")

        prompt = """你是一个{target_lang}翻译专家，给定上下文<context>供你参考背景信息, 这里有一句上下文中的文本<text>和他对应的{target_lang}翻译<translation>，我需要你帮我判断一下<translation>是否是一段符合要求的{target_lang}翻译。
要求仅考虑如下问题，任意不满足则不得分：
1. {target_lang}翻译是正确的且贴合上下文语境。
2. 对于你认为不影响理解的内容可以省略翻译，但不能无故添加内容。

输入：
<context>: {_context}
<text>: {_text}
<translation>: {_translation}

如果<translation>是一段符合要求的翻译，返回<<1>>，否则返回<<0>>。请直接返回<<0>>或者<<1>>。
输出："""
        resp = self.get_v3_response(prompt.format(
            target_lang=target_lang,
            _context=context,
            _text=origin_text,
            _translation=translation_text
        ))
        # print("v3 quality resp:", resp)
        # resp = resp.strip().split("\n")[-1].strip()
        if "<<1>>" in resp:
            return 1, ""
        else:
            return 0, ""

    def get_bleu_similarity(self, text1, text2):
        bleu_score, bleu_obj = self._bleu_similarity(text1, text2)
        # return bleu_score, bleu_obj
        if bleu_score >= 0.4:
            return 1, bleu_obj
        else:
            return 0, bleu_obj

    # 
    def _bleu_similarity(self, text1, text2):
        """BLEU相似度"""
        text1_length = len(text1)
        text2_length = len(text2)
        
        # 计算BLEU分数
        bleu = sacrebleu.corpus_bleu([text1], [[text2]], tokenize='zh')
        
        if text2_length < 4:
            # 对于短文本，只计算不超过文本长度的n-gram
            max_ngram = min(text2_length, 4)

            # 计算几何平均，只使用有效的n-gram精确度
            log_precisions = []
            for i in range(max_ngram):
                if bleu.precisions[i] > 0:
                    # 注意：sacrebleu返回的precisions是百分比形式(0-100)，需要转换为0-1
                    log_precisions.append(math.log(bleu.precisions[i] / 100))
                else:
                    # 如果某个n-gram精确度为0，使用一个很小的值避免log(0)
                    log_precisions.append(math.log(1e-10))
            
            # 计算几何平均
            if log_precisions:
                geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
            else:
                geometric_mean = 0
            
            # 计算简洁性惩罚
            # 使用分词后的长度计算简洁性惩罚
            if text1_length == 0:
                brevity_penalty = 0
            else:
                brevity_penalty = min(1.0, text2_length / text1_length)
            
            # 计算最终BLEU分数
            bleu_score = brevity_penalty * geometric_mean
        else:
            # 对于长文本，使用标准的BLEU分数
            bleu_score = bleu.score / 100
        
        return bleu_score, bleu

    def contains_chinese(self,text):
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    

if __name__ == "__main__":
    
    import openai
    def get_v3_response(prompt):
        client = openai.OpenAI(
            base_url="",
            api_key="",
        )
        try:
            resp = client.chat.completions.create(
                model="deepseek-v3",
                messages=[{"role": "user", "content": prompt}],
                )

            if resp and hasattr(resp, "choices") and resp.choices and len(resp.choices) > 0:
                if hasattr(resp.choices[0].message, "content") and resp.choices[0].message.content:
                    resp_str = resp.choices[0].message.content
        except Exception as e:
            print(f"error: {e}")
            resp_str = ""

        return resp_str


    def get_v3_quality(origin_text, context, translation_text, target_language="英文"):
        """
        获取翻译质量
        :param origin_text: 原文
        :param context: 上下文
        :param translation_text: 翻译文本
        :param target_language: 目标语言，默认为"英文"
        :return: 1 if quality is good, 0 otherwise
        """
        prompt = """给定上下文<context>供你参考背景信息, 这里有一句上下文中的文本<text>和他对应的{target_lang}翻译<translation>，我需要你帮我判断一下<translation>是否是一段符合要求的{target_lang}翻译。
要求仅考虑如下问题：
1. 翻译正确且贴合上下文语境，是流畅的{target_lang}翻译。
2. 可以适当做缩略翻译，不影响理解内容的缩略是可以接受的，但不能无故添加内容。

输入：
<context>: {_context}
<text>: {_text}
<translation>: {_translation}

如果<translation>是一段符合要求的翻译，返回<<1>>，否则返回<<0>>。请直接返回<<0>>或者<<1>>。
输出："""
        resp = get_v3_response(prompt.format(
            target_lang=target_language,
            _context=context,
            _text=origin_text,
            _translation=translation_text
        ))
        # print("v3 quality resp:", resp)
        # resp = resp.strip().split("\n")[-1].strip()
        if "<<1>>" in resp:
            return 1
        else:
            return 0

    text = "在实验室将它们和空气混合后，用紫外线照射，成功生成臭氧等刺激性物质，和洛杉矶迷雾成分一致，真相近在咫尺。"
    context = "有了推论后，他驱车到洛杉矶炼油厂和交通干道，用烧瓶收集空气样本，发现其中有高浓度的，碳氢化合物和氮氧化物。在实验室将它们和空气混合后，用紫外线照射，成功生成臭氧等刺激性物质，和洛杉矶迷雾成分一致，真相近在咫尺。他又发现拉伸的橡胶管在迷雾中会迅速老化开裂，这是臭氧与双键反应的特征。"
    translation_text = "In the lab, you mix them with air and then flash UV light on it to generate ozone — a sharp stimulant like LA's smog at last. Truth is close now."
    
    text = "在实验室将它们和空气混合后"
    context = "有了推论后，他驱车到洛杉矶炼油厂和交通干道，用烧瓶收集空气样本，发现其中有高浓度的，碳氢化合物和氮氧化物。在实验室将它们和空气混合后，用紫外线照射，成功生成臭氧等刺激性物质，和洛杉矶迷雾成分一致，真相近在咫尺。他又发现拉伸的橡胶管在迷雾中会迅速老化开裂，这是臭氧与双键反应的特征。"
    translation_text = "In the lab, he mixed them with air"
    
    print(get_v3_quality(text, context, translation_text)) 
