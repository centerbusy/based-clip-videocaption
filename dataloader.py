from paddle.io import DataLoader
tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
from paddlenlp.data import Stack, Tuple, Pad

batchify_fn = Tuple(
    Stack(dtype='float32'),
    Pad(axis=0, pad_val=-1)
)
# batchify_fn = Pad(axis=0, pad_val=tokenizer.pad_token_id)
data_loader = DataLoader(dataset,batch_size=4,collate_fn=batchify_fn)
# data_loader = DataLoader(dataset,batch_size=4)
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
for i in data_loader:
    print(i[0].shape,i[1].shape)
    x = i[0].squeeze(1)
    # print(x)
    print(clip.get_image_features(x).shape)
    break