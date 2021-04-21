#!/usr/bin/env python
# coding: utf-8

# In[18]:


from torchvision import models, transforms
import torch.nn as nn
import torch
import os
from skimage import io
from pathlib import Path
import dropbox
import shutil


# In[37]:


def model_predict(model_name):
    model_eval = models.resnet101(pretrained=True)
    class_output = {
        'binary': {'Ghost': 0, 'Animal': 1, 'Unknown': 2},
        'class': {'Ghost': 0, 'Aves': 1, 'Mammalia': 2, 'Unknown': 3},
        'species': {
            'Ghost': 0,
            'ArremonAurantiirostris': 1,
            'Aves (Class)': 2,
            'BosTaurus': 3,
            'CaluromysPhilander': 4,
            'CerdocyonThous': 5,
            'CuniculusPaca': 6,
            'Dasyprocta (Genus)': 7,
            'DasypusNovemcinctus': 8,
            'DidelphisAurita': 9,
            'EiraBarbara': 10,
            'Equus (Genus)': 11,
            'Leopardus (Genus)': 12,
            'LeptotilaPallida': 13,
            'Mammalia (Class)': 14,
            'MazamaAmericana': 15,
            'Metachirus (Genus)': 16,
            'Momota (Genus)': 17,
            'Nasua (Genus)': 18,
            'PecariTajacu': 19,
            'ProcyonCancrivorus': 20,
            'Rodentia (Order)': 21,
            'Sciurus (Genus)': 22,
            'SusScrofa': 23,
            'TamanduaTetradactyla': 24,
            'TinamusMajor': 25,
            'Unknown': 26}
    }

    model_map = class_output[model_name]
    in_features = len(model_map)
    reverse_model_map = {v: k for k, v in model_map.items()}

    using_gpu = torch.cuda.is_available()
    if using_gpu:
        yield('Image Prediction Starts. Using GPU! <br/>')
    else:
        yield('Image Prediction Starts. Using CPU!<br/>')
    device = torch.device("cuda:0" if using_gpu else "cpu")

    num_ftrs = model_eval.fc.in_features
    model_eval.fc = nn.Linear(num_ftrs, in_features)

    model_eval = model_eval.to(device)

    model_full_name = sorted([i for i in os.listdir(
        './models/') if model_name.lower() in i.split('_')], reverse=True)[0]

    try:
        model_eval.load_state_dict(torch.load(
            f'./models/{model_full_name}',  map_location=device))
        yield(f'Loading {model_full_name} pre-trained model<br/>')
    except Exception as e:
        yield(f'cannot load model! {e}<br/>')

    regulated_size = 300, 450
    default_val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=regulated_size),
        transforms.ToTensor(),
    ])
    model_eval.eval()
    softmax = torch.nn.Softmax(dim=1)

    # Dropbox API
    access_token = "IdSksADBFacAAAAAAAAAAcTAT5d77_SGkwjhmQcxC2BHo8ak0B1j_lGKiZqfH3cy"
    dbx = dropbox.Dropbox(access_token)

    img_dropbox = '/MIDS/Input_Data/'
    img_dropbox_to = '/MIDS/Output_Data/'

    img_local = Path('cache/')
    img_local.mkdir(exist_ok=True)
    folder = dbx.files_list_folder(img_dropbox)
    for file in folder.entries:
        if file.name.lower().endswith('jpg'):
            img_local_path = img_local / file.name
            dbx.files_download_to_file(
                download_path=img_local_path, path=file.path_display)
            tmp_img = io.imread(img_local_path)
            tmp_img = default_val_transform(tmp_img)

            with torch.no_grad():
                inputs = tmp_img.to(device)
                inputs = inputs.reshape(1, *inputs.shape)
                outputs = model_eval(inputs)

                prob = softmax(outputs)
                pred_prob, pred_id = torch.max(prob, 1)
                pred_id = pred_id.tolist()

                pred_prob = pred_prob.tolist()[0]
                pred_str = reverse_model_map[pred_id[0]]
            yield(
                f'Image Name: {file.name}; Predicted Class: {pred_str}; Predicted Probability:{pred_prob * 100:.2f}%<br/>')
            dbx.files_copy_v2(from_path=file.path_display,
                              to_path=f'{img_dropbox_to}{model_name}/{pred_str}/{file.name}', autorename=True)

    shutil.rmtree(img_local)
    yield('Image Prediction Finished!<br/>')
