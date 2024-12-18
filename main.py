import paddle
import matplotlib.pyplot as plt
from paddlenlp.transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from paddlenlp.transformers import CLIPProcessor, CLIPModel
# from transformers import CLIPProcessor, CLIPModel
import pyttsx3

def clear_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除 {file_path}: {e}")

        print(f"成功清空文件夹: {folder_path}")
    except Exception as e:
        print(f"清空文件夹时发生错误: {e}")

def video_clip(cap,timef):
    time_start = time.time()
    isOpened = cap.isOpened
    fps = cap.get(cv2.CAP_PROP_FPS)

    imageNum = 0
    sum = 0
    timef = timef

    while (isOpened):
        sum += 1
        (frameState, frame) = cap.read()
        if frameState == True and (sum % timef == 0):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            imageNum = imageNum + 1
            fileName = 'clp/' + str(imageNum) + '.jpg'
            cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        elif frameState == False:
            break
    print('finish!')

    time_end = time.time()
    d_time = time_end - time_start
    print("图像分帧的耗时为 %.8s S" % d_time)
    cap.release()

def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

def cle(folder_path,org_img_folder,cap,timef):
    clear_folder(folder_path)
    video_clip(cap,timef)
    imglist = getFileList(org_img_folder, [], 'jpg')
    print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]
    return sentences, words

def build_graph(words):
    graph = nx.Graph()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            similarity = compute_similarity(words[i], words[j])
            if similarity > 0:
                graph.add_edge(i, j, weight=similarity)
    return graph

def compute_similarity(sentence1, sentence2):
    vector1 = set(sentence1)
    vector2 = set(sentence2)
    intersection = vector1.intersection(vector2)
    union = vector1.union(vector2)
    similarity = len(intersection) / len(union) if len(union) > 0 else 0
    return similarity

def textrank(graph):
    scores = nx.pagerank(graph)
    return scores

def extract_summary(sentences, scores, num_sentences=1):
    ranked_sentences = sorted(((scores.get(i, 0), sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summary = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    return summary

def get_img_feature(img_path):
    imgs = [cv2.imread(x) for x in img_path]
    # print("imgs",imgs)
    clip_imgs = [clip_preprocess(images = x,return_tensors="pd")["pixel_values"] for x in imgs]

    with paddle.no_grad():
        image_fts = [clip_model.get_image_features(x) for x in clip_imgs]
        image_features = sum(image_fts)

        image_features = image_features / image_features.norm(axis=-1, keepdim=True)
        return image_features.detach()

if __name__ == '__main__':
    lm_tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')

    # clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prefix_length = 10
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = ClipCaptionModel(10, clip_length=10, prefix_size=512, num_layers=8)
    # model.set_state_dict(paddle.load("/home/aistudio/data/data175723/_latest.pdparams"))#data/data177804/-epoch_last.pdparams
    model.set_state_dict(paddle.load("-epoch_last.pdparams"))  #

    imglist = getFileList(org_img_folder, [], 'jpg')
    # print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')

    best_caption = {}

    for num in range(1, len(imglist) + 1):
        UPLOADED_FILE = "clp/" + str(num) + ".jpg"  # 图片路径
        with paddle.no_grad():
            pil_image = cv2.imread(UPLOADED_FILE)
            image = clip_preprocess(images=pil_image, return_tensors="pd")["pixel_values"]
            prefix = clip_model.get_image_features(image).astype(dtype=paddle.float32)
            prefix = prefix / prefix.norm(2, -1).item()
            prefix_embed = model.clip_project(prefix).reshape([1, prefix_length, -1])

        gen_caption = generate_beam(model, lm_tokenizer, embed=prefix_embed, beam_size=5)

        # 最佳描述
        tokenizer = AutoTokenizer.from_pretrained('clip-vit-base32')
        image_features = get_img_feature([UPLOADED_FILE])
        encoded_captions = [clip_model.get_text_features(paddle.to_tensor(tokenizer(c)["input_ids"]).unsqueeze(0)) for c
                            in gen_caption]
        encoded_captions = [x / x.norm(axis=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (paddle.concat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
        # img = Image.open(UPLOADED_FILE)
        # img.show()
        # print(num)
        # print("第" + str(num * 0.5) + "秒的内容为", gen_caption[best_clip_idx])
        best_caption[num] = gen_caption[best_clip_idx]

    texts = best_caption
    vectorizer = CountVectorizer()
    corpus = list(texts.values())
    X = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(X)

    # 定义阈值
    threshold = threshold

    similar_groups = []

    for i in range(len(texts)):
        similar_group = [i + 1]
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                similar_group.append(j + 1)
            else:
                break
        if len(similar_group) > 1:
            similar_groups.append(similar_group)

    merged_similar_groups = []
    for group in similar_groups:
        merged_group = set(group)
        for other_group in similar_groups:
            if group != other_group and set(group).issubset(other_group):
                merged_group = merged_group.union(set(other_group))
        merged_similar_groups.append(list(merged_group))

    merged_similar_groups = [sorted(sublist) for sublist in merged_similar_groups]
    merged_similar_groups = sorted(merged_similar_groups, key=lambda x: x[0])

    # 去除重复的相似文本组
    unique_merged_similar_groups = [list(group) for group in set(tuple(group) for group in merged_similar_groups)]

    # print(unique_merged_similar_groups)

    unique_merged_similar_groups = sorted(unique_merged_similar_groups, key=lambda x: x[0])

    for group in unique_merged_similar_groups:
        group = sorted(group)
        start_time = round(float(group[0]) / 6, 2)
        end_time = round(float(group[-1]) / 6, 2)
        print(f"连续区间描述为{best_caption[group[0]]}")
        # print(f"{start_time}秒到{end_time}秒连续，为{best_caption[group[0]]}")

    til = ' '.join(best_caption.values())

    text1 = til
    sentences, words = preprocess_text(text1)
    graph = build_graph(words)
    scores = textrank(graph)
    summary = extract_summary(sentences, scores)
    zy = " ".join(summary)

    print("视频总结:")
    print(zy)

    with open('../summary.txt', 'w') as f:
        f.write(zy)

