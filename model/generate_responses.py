import os
import torch
import logging
import argparse
from tqdm import tqdm

from utils.data_utils import *
from utils.model_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mname', type=str, default='facebook/blenderbot-400M-distill')
    parser.add_argument('--dname', type=str, default='allenai/prosocial-dialog')
    args = parser.parse_args()
    return args

def contains_violence(text):
    violence_keywords = ["throw", "hit", "punch", "kill", "beat", "harm", "abuse", "attack"]
    lower_text = text.lower()
    return any(vk in lower_text for vk in violence_keywords)

def contains_illegal_behavior(text):
    illegal_keywords = ["steal", "rob", "without paying", "fraud"]
    lower_text = text.lower()
    return any(ik in lower_text for ik in illegal_keywords)

def contains_hate_speech(text):
    # 혐오 표현 리스트 확대
    hate_words = ["wetback", "spic", "savages", "nigger", "chink", "faggot", "half-breed",
                  "kike", "jew", "jewish", "jews", "raghead", "gook", "muslim", "muslims"]
    lower_text = text.lower()
    return any(hw in lower_text for hw in hate_words)

def contains_child_abuse(text):
    # 아동대상 학대를 포괄적으로 처리
    child_related = ["child", "kid", "minor"]
    abuse_terms = ["abuse", "harm", "molest", "sexual acts", "hurt", "paint on them", "exploit"]
    lower_text = text.lower()
    return any(cr in lower_text for cr in child_related) and any(at in lower_text for at in abuse_terms)

def is_safe_response(response):
    # 기존 유해 단어 체크 + 혐오 표현 체크 추가
    unsafe_keywords = [
        "abuse", "kill", "terrorist", "rape", "child porn", "behead",
        "suicide advice", "genocide"  # 다소 극단적인 표현 추가
    ]
    if any(keyword in response.lower() for keyword in unsafe_keywords):
        return False
    if contains_hate_speech(response):
        return False
    if contains_child_abuse(response):
        return False
    if contains_violence(response):
        return False
    if contains_illegal_behavior(response):
        return False
    return True

def generate_candidates(model, encoded_inputs, tokenizer, device, num_return_sequences=3):
    # 여러 응답 후보를 생성 후 반환
    with torch.no_grad():
        reply_ids = model.generate(
            **encoded_inputs,
            num_beams=5,
            max_length=100,
            temperature=1.5,  # 더 창의적인 응답 유도
            top_p=0.9,  # 유사도가 높은 응답 후보 포함
            repetition_penalty=1.2,  # 반복 단어 억제
            no_repeat_ngram_size=3,  # n-gram 반복 방지
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )
    candidates = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
    return candidates


def choose_best_response(candidates, user_input):
    # 후보들 중 안전하고, 너무 짧지 않고, 너무 공격적이지 않은 응답 선택
    # 후보 중 안전하고, 맥락에 적합한 응답 선택
    safe_responses = [c for c in candidates if is_safe_response(c)]
    if not safe_responses:
        return "I'm sorry, I cannot provide a response for this topic."

    # 사용자 입력과 가장 관련성 높은 응답 선택 (e.g., 의미적 유사성)
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    candidate_embeddings = model.encode(safe_responses, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(input_embedding, candidate_embeddings)
    best_index = similarity_scores.argmax().item()

    return sanitize_response(safe_responses[best_index])

def sanitize_response(response):
    # 민감한 단어 대체
    unsafe_words = {
        "violence": "conflict",
        "hate": "dislike",
        "harm": "issue",
        "terror": "fear-inducing act"
    }
    for word, replacement in unsafe_words.items():
        response = response.replace(word, replacement)
    return response

def main(args):
    id = input(u'Input ID: ')
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S',
        level=logging.INFO,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'{device} is used for response generation')
    
    os.makedirs('../results', exist_ok=True)
    response_path = f'../results/responses_{id}.txt'
    if os.path.isfile(response_path): os.remove(response_path)
    logging.info(f'Generated responses will be saved at {response_path}')
    
    with open(response_path, "w") as f:
        f.write("context|response\n")
    
    logging.info(f'Model loaded from {args.mname}')
    '''
    Step 1. Load model and tokenizer. You may want to use load_model function in utils/model_utils.py
    
    tokenizer = 
    model = 
    '''
    tokenizer, model = load_model(args.mname)
    model.eval()
    model.to(device)
    
    logging.info(f'Data loaded from {args.dname}')
    '''
    Step 2. Load your own dataset. You may want to use load_dataset function in utils/data_utils.py
    
    dataset = 
    dataloader = 
    '''

    turn = "single" # single  OR  multi

    dataset = load_dataset(dname='prosocial_dialog',turn=turn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    logging.info('Generating Responses...')


    '''
    Step 3. Generate responses
    for data in tqdm(dataloader):
        context = 
        reply_ids = 
        reply_txts = 
        ...    
    '''

    context = []
    prompt_template = (
        "You are an empathetic, ethical assistant. Respond to the user's question with helpful and specific advice.\n"
        "Avoid harmful, illegal, or inappropriate suggestions.\n\n"
        "User: "
    )
    #safe_guideline = "YOU are A helpful and Strict assistant robot. Always respond in a polite and safe manner. "
    #safe_bottomline = "If the input could result in harmful, unethical, or unsafe behavior, respond with a strict warning and suggest alternative positive action"
    reply_txts = []
    for data in tqdm(dataloader):
        if turn == "multi":
            encoded_inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True).to(device)
        else:
            user_input = data[0]#+ safe_bottomline for incontext learning
            if contains_hate_speech(user_input):
                # 혐오적 질문에 대해 명확히 거절하는 응답
                reply_txt = "I'm sorry, but I can't help with that."
                context.append(user_input)
                reply_txts.append(reply_txt)
                # 굳이 모델 호출 없이 넘어감
                continue
            prompt = prompt_template + data[0] + "\nAssistant: "#for Prompt Engineering
            encoded_inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)


        with torch.no_grad():
            if turn == "multi":
                context.append(data[0].replace("</s> <s>", "|"))
                reply_ids = model.generate(**encoded_inputs, num_beams=5, max_new_tokens=50)
                reply_txts.append(tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
            else :
                safe_reply = False
                context.append(data[0])
                i = 0
                for _ in range(5):  # 최대 5번 재생성 시도
                    candidates = generate_candidates(model, encoded_inputs, tokenizer, device, num_return_sequences=3)
                    reply_txt = choose_best_response(candidates, data[0])
                    """
                        reply_ids = model.generate(
                            **encoded_inputs,
                            num_beams=5,
                            max_length=200,
                            temperature=1.0,
                            top_p=0.85,
                            repetition_penalty=1.5,
                            no_repeat_ngram_size=4,
                            early_stopping=True
                        )
                        reply_txt = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
                    """
                    i += 1
                    if not contains_hate_speech(reply_txt) and is_safe_response(reply_txt):
                        reply_txt = sanitize_response(reply_txt)
                        safe_reply = True
                        break
                    else:
                        # 혐오표현 감지 시 재시도 없이 바로 거절
                        if contains_hate_speech(reply_txt):
                            reply_txt = "I'm sorry, but I can't help with that."
                            safe_reply = True
                            break
                if not safe_reply:
                                # 끝까지 안전한 응답을 못 찾았다면 거절
                    reply_txt = "I'm sorry, I cannot provide a response for this topic."
                reply_txts.append(reply_txt)

    # Save contexts and responses
    with open(response_path, 'a') as f:
        for c, r in zip(context, reply_txts):
            f.write(f'{c}|{r}\n')
    
    logging.info(f'Total {len(dataset)} responses saved at {response_path}')
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
