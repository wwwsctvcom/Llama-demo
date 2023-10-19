from transformers.generation.utils import GenerationConfig
from utils.dataset import *

model = LlamaForCausalLM.from_pretrained("/kaggle/working/trained")
model.generation_config = GenerationConfig.from_dict({'num_beams': 1,
                                                      'max_new_tokens': 100,
                                                      'max_length': 200})

model.generation_config.num_beams = 1
model.generation_config.max_new_tokens = 100
model.generation_config.max_length = 200


def get_ans(tensor, src_ids) -> str:
    answer = [i for i in tensor.tolist() if i not in src_ids]
    answer = "".join([id2word[i] for i in answer]).replace('<BOS>', '').replace('<EOS>', '')
    return answer


src_tokens = "文本分类任务：将一段用户给手机的评论进行分类。下面是一些范例：卡了100分钟了！我的天阿我快受不了了"
tgt_tokens = "流畅度"
x, y = [word2id['<BOS>']] + [word2id[word] for word in src_tokens], [word2id[word] for word in tgt_tokens] + [
    word2id['<EOS>']]

input_ids = torch.tensor([x])
out = model.generate(inputs=input_ids)

ans = get_ans(out[0], src_ids=x)

print(ans)
