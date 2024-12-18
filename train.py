import skimage.io as io
import PIL.Image
from IPython.display import Image 
from paddlenlp.transformers import GPTLMHeadModel,GPTTokenizer
from paddlenlp.transformers import CLIPProcessor,CLIPModel
import cv2
import numpy as np
from paddle.io import DataLoader
from tqdm import tqdm
from paddlenlp.transformers import LinearDecayWithWarmup
from visualdl import LogWriter

prefix_length = 10
def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)["input_ids"][0]
    tokens = None
    scores = None
    seq_lengths = paddle.ones([beam_size])
    is_stopped = paddle.zeros([beam_size], dtype=paddle.bool)
    with paddle.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = paddle.to_tensor(tokenizer.encode(prompt)["input_ids"])
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.embedding(tokens)
        for i in range(entry_length):
            outputs = model.gpt.forward_embed(generated)
            # logits = outputs.logits
            logits = outputs
            # logits = nnf.softmax(outputs,axis = -1)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = nnf.softmax(logits,axis = -1).log()
            # logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand([beam_size, *generated.shape[1:]])
                next_tokens, scores = next_tokens.transpose([1, 0]), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand([beam_size, *tokens.shape[1:]])
                    tokens = paddle.concat((tokens, next_tokens), axis=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.reshape([-1]).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = paddle.concat((tokens, next_tokens), axis=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source.numpy().tolist()]
            next_token_embed = model.gpt.embeddings(next_tokens.squeeze()).reshape(
                [generated.shape[0], 1, -1]
            )
            generated = paddle.concat((generated, next_token_embed), axis=1)
            # is_stopped = is_stopped + next_tokens.equal(stop_token_index).squeeze()
            temp_a = next_tokens.equal(paddle.full_like(next_tokens,stop_token_index)).astype("float32").squeeze()
            is_stopped = paddle.any(paddle.stack([is_stopped.astype("float32"),temp_a],axis=0).astype("bool"),axis=0)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

# def train(dataset: CaptionDataset, model: ClipCaptionModel,lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
def train(lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
    dataset = CaptionDataset()
    model = ClipCaptionModel(10, clip_length=10, prefix_size=512,
                             num_layers=8)
    batch_size = 32
    epochs = 10
    # dist.init_parallel_env()
    # lr *= 4

    # 第 3 处改动，增加 paddle.DataParallel 封装
    # model = paddle.DataParallel(model)
    # model.set_state_dict(paddle.load("~/coco/PythonAPI/_latest.pdparams")) #~/data/data175723/_latest.pdparams
    model.set_state_dict(paddle.load("~/data/data177804/-epoch_last.pdparams"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.train()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=batchify_fn, shuffle=True, drop_last=True)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = LinearDecayWithWarmup(lr, total_steps=epochs * len(train_dataloader), warmup=warmup_steps)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                       parameters=model.parameters())
    lm_tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
    step_i = 0
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        # for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
        for idx, (image, tokens) in enumerate(train_dataloader):

            prefix = clip.get_image_features(image.squeeze(1)).detach()  # [bs,512]

            # model.clear_grad()
            mask = None
            mask = tokens > -0.5  # mask is zero where we out of sequence

            tokens[~mask] = 0
            mask = mask.astype("float32")
            mask = paddle.concat((paddle.ones([batch_size, 10]), mask), axis=1)  # adding prefix mask
            outputs = model(tokens, prefix, mask)
            # print("outputs",outputs)
            # logits = nnf.softmax(outputs,axis = -1)[:, 10 - 1: -1]
            logits = outputs[:, 10 - 1: -1]
            # logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape([-1, logits.shape[-1]]), tokens.flatten(), ignore_index=0)
            loss.backward()
            log_writer.add_scalar(tag='loss', step=step_i, value=float(loss))
            optimizer.step()
            scheduler.step()
            optimizer.clear_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (step_i + 1) % 1000 == 0:
                print(generate_beam(model, lm_tokenizer, embed=prefix_embed))
                paddle.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pdparams"),
                )
            step_i += 1
        progress.close()
        # if epoch % args.save_every == 0 or epoch == epochs - 1:
        paddle.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}-epoch_last.pdparams"),
        )
    return model


# dataset = CaptionDataset()
# model = ClipCaptionModel(10, clip_length=10, prefix_size=512,
#                                   num_layers=8)
# train(dataset,model)
# dist.spawn(train)
if __name__ == "__main__":
    train()