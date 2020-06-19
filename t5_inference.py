import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)


def greedy_decoding (inp_ids,attn_mask):
  greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
  Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return Question.strip().capitalize()


def beam_search_decoding (inp_ids,attn_mask):
  beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
  return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding (inp_ids,attn_mask):
  topkp_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               do_sample=True,
                               top_k=40,
                               top_p=0.80,
                               num_return_sequences=3,
                                no_repeat_ngram_size=2,
                                early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]
  return [Question.strip().capitalize() for Question in Questions]

start = time.time()

# passage = "Starlink, of SpaceX, is a satellite constellation project being developed by Elon Musk and team to give satellite Internet go-to access for people in any part of the world. The plan is to comprise thousands of mass-delivered little satellites in low Earth circle, orbit, working in mix with ground handheld devices, for instance, our iPhones. Elon Musk speaks about it as a grand Idea that could change the way we view and access the world around us."
# truefalse ="yes"

# passage ="About 400 years ago, a battle was unfolding about the nature of the Universe. For millennia, astronomers had accurately described the orbits of the planets using a geocentric model, where the Earth was stationary and all the other objects orbited around it."
# truefalse ="yes"

passage ='''Months earlier, Coca-Cola had begun “Project Kansas.” It sounds like a nuclear experiment but it was just a testing project for the new flavor. In individual surveys, they’d found that more than 75% of respondents loved the taste, 15% were indifferent, and 10% had a strong aversion to the taste to the point that they were angry.'''
truefalse ="yes"

# passage ="The US has passed the peak on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637000 confirmed Covid-19 cases and over 30826 deaths, the highest for any country in the world."
# truefalse = "yes"


text = "truefalse: %s passage: %s </s>" % (passage, truefalse)


max_len = 256

encoding = tokenizer.encode_plus(text, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)



print ("Context: ",passage)
# print ("\nGenerated Question: ",truefalse)

# output = greedy_decoding(input_ids,attention_masks)
# print ("\nGreedy decoding:: ",output)

output = beam_search_decoding(input_ids,attention_masks)
print ("\nBeam decoding [Most accurate questions] ::\n")
for out in output:
    print(out)


output = topkp_decoding(input_ids,attention_masks)
print ("\nTopKP decoding [Not very accurate but more variety in questions] ::\n")
for out in output:
    print (out)

end = time.time()
print ("\nTime elapsed ", end-start)
print ("\n")

