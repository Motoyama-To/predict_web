# python pet_web/app.py を実行し、app起動。
#  http://127.0.0.1:5000

# Web画像入力 → YOLOv8で犬猫検出・切り出し → ResNet18で特徴抽出 → CBAMで重要特徴を強調
# → Softmaxで品種確率を計算 → Top-5品種推定 → Grad-CAMで注目領域を可視化
# → Cosine Similarityで類似画像検索 → Wikipediaで品種説明取得 → Web画面に結果表示

# # predict_web.py

import os
import base64
import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# パス設定
# =========================================================

# pet_web フォルダ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# pet_web の1つ上 → 4th フォルダ
PROJECT_DIR = os.path.dirname(BASE_DIR)

# 学習データ
DATA_DIR = os.path.join(PROJECT_DIR, '20-100images')

# キャッシュは pet_web 内
CROP_DIR = os.path.join(BASE_DIR, 'cropped_cache')

# 学習済みモデル
CAT_MODEL_PATH = os.path.join(PROJECT_DIR, 'cat_model.pth')
DOG_MODEL_PATH = os.path.join(PROJECT_DIR, 'dog_model.pth')

# 特徴量データベース
CAT_DB_PATH = os.path.join(PROJECT_DIR, 'cat_feature_db.npz')
DOG_DB_PATH = os.path.join(PROJECT_DIR, 'dog_feature_db.npz')

# YOLOモデル
YOLO_PATH = os.path.join(PROJECT_DIR, 'yolov8n.pt')

IMAGE_SIZE = (128, 128)
SIMILAR_IMAGES_TO_DISPLAY = 10

os.makedirs(CROP_DIR, exist_ok=True)

print("PROJECT_DIR :", PROJECT_DIR)
print("CAT_MODEL   :", CAT_MODEL_PATH)
print("DOG_MODEL   :", DOG_MODEL_PATH)
print("CAT_DB      :", CAT_DB_PATH)
print("DOG_DB      :", DOG_DB_PATH)
print("YOLO_MODEL  :", YOLO_PATH)


IMAGE_SIZE = (128, 128)
SIMILAR_IMAGES_TO_DISPLAY = 10
os.makedirs(CROP_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cat_classes = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Sphynx']
dog_classes = ['american_bulldog','beagle','Boxer','chihuahua','english_setter','german_shorthaired','japanese_chin','leonshond','samoyed','shiba_inu']
cat_class_names = sorted(cat_classes)
dog_class_names = sorted(dog_classes)

predict_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize(IMAGE_SIZE), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.ca = CAM(channels); self.sa = SAM()
    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x

class BreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.cbam = CBAM(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.cbam(self.features(x)); x = self.pool(x); x = torch.flatten(x, 1)
        return self.fc(x)

def image_to_data_url(image):
    ok, buf = cv2.imencode('.jpg', image)
    if not ok: return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')

def detect_and_crop_animal(image, yolo_model):
    results = yolo_model(image, verbose=False)[0]
    detections = []
    if results.boxes is None or len(results.boxes) == 0:
        return image, 'unknown', detections
    max_area, best_crop, detected_label = 0, image, 'unknown'
    h, w = image.shape[:2]
    for box in results.boxes:
        cls_id = int(box.cls[0]); label = yolo_model.names[cls_id]; conf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        detections.append({'label': label, 'confidence': round(conf*100,2), 'box':[x1,y1,x2,y2]})
        if label not in ['cat','dog']: continue
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        if x2 <= x1 or y2 <= y1: continue
        area = (x2-x1)*(y2-y1)
        if area > max_area:
            max_area, best_crop, detected_label = area, image[y1:y2,x1:x2], label
    return best_crop, detected_label, detections


def load_model(path):

    print("モデル読み込み中 :", path)

    # BreedClassifierを作成
    model = BreedClassifier(num_classes=10)

    # 学習済み重みを読み込む
    state_dict = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    # CPU / GPUへ移動
    model = model.to(device)

    # 推論モード
    model.eval()

    print("モデル読み込み完了 :", path)

    return model


def load_db(kind):

    if kind == 'cat':
        p = CAT_DB_PATH
    elif kind == 'dog':
        p = DOG_DB_PATH
    else:
        raise ValueError(f'不明な種類です: {kind}')

    if not os.path.exists(p):
        raise FileNotFoundError(
            f'{p} がありません。\n'
            '先に元の predict_attention.py を実行して'
            '特徴量DBを作成してください。'
        )

    print(f'{kind} feature database loading : {p}')

    d = np.load(p, allow_pickle=True)

    paths = []

    for x in d['paths']:

        path = str(x)

        # npz内に絶対パスが保存されている場合
        if os.path.isabs(path):
            paths.append(path)

        # 相対パスの場合は4thを基準にする
        else:
            paths.append(
                os.path.join(PROJECT_DIR, path)
            )

    return (
        d['features'],
        d['labels'],
        paths
    )

def extract_features(image, model):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = predict_transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.cbam(model.features(tensor)); x = model.pool(x); x = torch.flatten(x,1)
    return x.cpu().numpy()[0]

def generate_gradcam(image, model, class_idx):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = predict_transform(rgb).unsqueeze(0).to(device)
    gradients, activations = [], []
    fh = model.cbam.register_forward_hook(lambda m,i,o: activations.append(o))
    bh = model.cbam.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0]))
    output = model(tensor); model.zero_grad(); output[0,class_idx].backward()
    fh.remove(); bh.remove()
    weights = torch.mean(gradients[0], dim=(2,3), keepdim=True)
    cam = F.relu(torch.sum(weights * activations[0], dim=1).squeeze()).detach().cpu().numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = (cam-cam.min())/(cam.max()-cam.min()) if cam.max()!=cam.min() else np.zeros_like(cam)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

def get_breed_description(name):
    name_map = {'german_shorthaired':'German Shorthaired Pointer','english_setter':'English Setter','Maine_Coon':'Maine Coon','Egyptian_Mau':'Egyptian Mau','British_Shorthair':'British Shorthair','japanese_chin':'Japanese Chin','american_bulldog':'American Bulldog','shiba_inu':'Shiba Inu','chihuahua':'Chihuahua (dog breed)','Boxer':'Boxer (dog breed)','leonshond':'Keeshond','samoyed':'Samoyed dog'}
    search_name = name_map.get(name, name.replace('_',' '))
    try:
        r = requests.get('https://en.wikipedia.org/w/api.php', params={'action':'query','format':'json','prop':'extracts','exintro':1,'explaintext':1,'redirects':1,'titles':search_name}, headers={'User-Agent':'BreedClassifier/1.0 (student project)'}, timeout=10)
        r.raise_for_status()
        for page in r.json()['query']['pages'].values():
            if page.get('extract'): return page['extract']
    except Exception as e:
        return f'Wikipedia取得エラー: {e}'
    return 'Wikipediaの説明が見つかりませんでした。'


class WebPredictor:

    def __init__(self):

        print("\n==============================")
        print("Web Predictor 起動")
        print("==============================")

        # -----------------------------
        # YOLO
        # -----------------------------

        if os.path.exists(YOLO_PATH):
            print("YOLO :", YOLO_PATH)
            self.yolo = YOLO(YOLO_PATH)

        else:
            print("yolov8n.pt が見つからないため自動取得します。")
            self.yolo = YOLO("yolov8n.pt")

        # -----------------------------
        # 猫モデル
        # -----------------------------
        if not os.path.exists(CAT_MODEL_PATH):
            raise FileNotFoundError(
                f"猫モデルがありません:\n{CAT_MODEL_PATH}"
            )

        print("猫モデル :", CAT_MODEL_PATH)

        self.cat_model = load_model(
            CAT_MODEL_PATH
        )

        # -----------------------------
        # 犬モデル
        # -----------------------------
        if not os.path.exists(DOG_MODEL_PATH):
            raise FileNotFoundError(
                f"犬モデルがありません:\n{DOG_MODEL_PATH}"
            )

        print("犬モデル :", DOG_MODEL_PATH)

        self.dog_model = load_model(
            DOG_MODEL_PATH
        )

        # -----------------------------
        # 特徴量DB
        # -----------------------------
        self.cat_db, self.cat_labels, self.cat_paths = \
            load_db('cat')

        self.dog_db, self.dog_labels, self.dog_paths = \
            load_db('dog')

        print("==============================")
        print("モデル・特徴量DB 読み込み完了")
        print("==============================\n")

    def _analyze(self, crop, model, db, labels, paths, class_names, animal_type):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = predict_transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        pred = int(torch.argmax(probs).item())
        top = torch.topk(probs, k=5)
        feature = extract_features(crop, model)
        similarities = cosine_similarity([feature], db)[0]
        order = np.argsort(similarities)[::-1][:SIMILAR_IMAGES_TO_DISPLAY]
        similar = []
        for idx in order:
            img = cv2.imread(paths[idx])
            if img is None: continue
            similar.append({'breed':class_names[int(labels[idx])], 'similarity':round(float(similarities[idx]),4), 'image':image_to_data_url(img)})
        gradcam = generate_gradcam(crop, model, pred)
        return {
            'animal_type': animal_type,
            'breed': class_names[pred],
            'confidence': round(float(probs[pred])*100,2),
            'top5':[{'breed':class_names[int(i)], 'probability':round(float(v)*100,2)} for v,i in zip(top.values, top.indices)],
            'gradcam': image_to_data_url(gradcam),
            'similar_images': similar,
            'description': get_breed_description(class_names[pred])
        }

    def predict(self, image):
        crop, animal_type, detections = detect_and_crop_animal(image, self.yolo)
        base = {'detected_type':animal_type, 'detections':detections, 'cropped_image':image_to_data_url(crop)}
        if animal_type == 'cat':
            base['results']=[self._analyze(crop,self.cat_model,self.cat_db,self.cat_labels,self.cat_paths,cat_class_names,'cat')]
        elif animal_type == 'dog':
            base['results']=[self._analyze(crop,self.dog_model,self.dog_db,self.dog_labels,self.dog_paths,dog_class_names,'dog')]
        else:
            cat = self._analyze(crop,self.cat_model,self.cat_db,self.cat_labels,self.cat_paths,cat_class_names,'cat')
            dog = self._analyze(crop,self.dog_model,self.dog_db,self.dog_labels,self.dog_paths,dog_class_names,'dog')
            base['results']=sorted([cat,dog], key=lambda x:x['confidence'], reverse=True)
        return base
