import os
from transformers import BertTokenizer, BartForConditionalGeneration

from http import HTTPStatus
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
import spacy
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
dashscope.api_key="sk-de9da7cb39b14506a4d45dad34d80469"


class BartChatbot:

    def __init__(self, device="cpu"):
        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        if os.path.exists("model"):
            self.model = BartForConditionalGeneration.from_pretrained("model")
        else:
            self.model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        self.model.to(device)
        self.device = device
        self.dialogue_history = []

    def chat(self, s):
        self.dialogue_history.append(s)
        input_text = " ".join(self.dialogue_history)
        encoded_dict = self.tokenizer.encode_plus(
            input_text,
            max_length=512,
            truncation=True,  # 启用截断
            truncation_strategy='longest_first',  # 如果需要截断，从最长的一端开始移除token
            return_tensors='pt'  # 返回PyTorch tensors
        )
        input_ids = encoded_dict.input_ids
        outputs = self.model.generate(input_ids, max_length=512)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text.replace(" ", "")
        self.dialogue_history.append(output_text)
        if "拜拜" in output_text or "再见" in output_text or "拜拜" in input_text or "再见" in input_text or "晚安" in output_text:
            self.dialogue_history = []
        return output_text


class ApiChatbot:
    def __init__(self, knowledge_file_path=None):
        self.history = []
        self.knowledge = []
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def extract_entities(self, input_text):
        doc = spacy.load("zh_core_web_sm")(input_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return set(entities)

    def retrieve_information(self, file_path, entity):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                entry = eval(line)
                parsed_entry = {
                    'entity': entry[0],
                    'attribute': entry[1],
                    'value': entry[2]
                }

                data.append(parsed_entry)

        relevant_data = []
        for entry in data:
            for each in entity:
                if entry["entity"].lower() == each[0].lower():
                    relevant_data.append(entry)

        return relevant_data

    def calculate_similarity(self, input_text, info):
        # Embed the input text
        input_embed = self.model.encode(input_text, convert_to_tensor=True)

        similarities = []

        # Iterate over all info pieces to find their similarities
        # use tqdm to show the progress bar
        for entry in tqdm(info, desc="Calculating similarity", total=len(info)):
            info_text = f"{entry['attribute']} 的 {entry['entity']} 是 {entry['value']}"
            info_embed = self.model.encode(info_text, convert_to_tensor=True)

            # Calculate cosine similarity
            cosine_scores = util.pytorch_cos_sim(input_embed, info_embed)
            similarity = cosine_scores.item()

            # Append similarity and entry to the list
            similarities.append((similarity, info_text))

        # Sort the list of tuples by similarity in descending order
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Retrieve the top five most similar entries
        top_similarities = similarities[:5]

        return top_similarities

    def call_with_messages(self, input_text):
        if not input_text:
            return "请输入内容", self.history

        entity = self.extract_entities(input_text)
        print(entity)

        related_data = self.retrieve_information("./data/knowledge_set.txt", entity)
        print("lenth of related data: ", len(related_data))

        top_info = self.calculate_similarity(input_text, related_data)

        for info in top_info:
            self.knowledge.append({'role': "system", 'content': info[1]})

        message = [{'role': "user", 'content': "根据以上背景信息以及对话记录回答以下问题：" + input_text}]

        response = Generation.call(Generation.Models.qwen_max, messages=self.knowledge + self.history + message,
                                   result_format='message')

        self.history.extend(message)

        self.history.append({'role': response.output.choices[0]['message']['role'],
                             'content': response.output.choices[0]['message']['content']})

        return response.output.choices[0]['message']['content']