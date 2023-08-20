from flask import Flask, request, render_template
from models.gsl_mol import GslMolNet
from utils.config import Configs, build_criterion
import torch

app = Flask(__name__)


@app.route('/')
def home():  # put application's code here
    # return render_template('predict-page.html')
    return render_template('index.html')


@app.route('/predict-page')
def toPredictPage():
    return render_template('predict-page.html')

@app.route('/model')
def toModelPage():
    return render_template('model.html')

@app.route('/dataset')
def toDatasetPage():
    return render_template('dataset.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form.get('smiles')
    dataset = request.form.get('property')

    print("smiles:", smiles)

    print(dataset)

    if dataset == 'BACE':
        class_names = ["有活性", "无活性"]
    elif dataset == 'BBBP':
        class_names = ["能穿透血脑屏障", "不能穿透血脑屏障"]

    opt = Configs(ParamList)
    device = torch.device(
        f"cuda:{opt.args['CUDA_VISIBLE_DEVICES']}" if torch.cuda.is_available() else 'cpu')
    criterion = build_criterion(opt.args['ClassNum'], opt.args['TaskNum'], device)
    model = GslMolNet(opt, criterion).to(device)
    ckpt = torch.load(opt.args["ModelPath"], map_location=device)
    model.load_state_dict(ckpt['model'])

    model.eval()

    output = model(smiles)

    print("output", output)
    # Get class probabilities
    probabilities = torch.nn.functional.softmax(output)
    probabilities = probabilities.detach().numpy()

    for i in range(len(probabilities)):
        probabilities[i] = round(probabilities[i] * 100, 2)

    print("probabilities:", probabilities)

    # Get the index of the highest probability
    class_index = probabilities.argmax()

    # Get the predicted class and probability
    predicted_class = class_names[class_index]
    probability = probabilities[class_index]

    class_probs = list(zip(class_names, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    return render_template('predict-result.html', class_probs=class_probs,
                           predicted_class=predicted_class, probability=probability)

ParamList = {
    'CUDA_VISIBLE_DEVICES': '1',
    'TorchSeed': 8,
    'SplitSeedBias': 0,
    'TaskNum': 1,
    'ClassNum': 2,
    'OutputSize': 2,

    'ModelPath': 'models/model_bace',
    'AtomFeatureSize': 39,

    'lr': 2.5,
    'WeightDecay': 5,

    'graph_learn': True,

    # Params for GslMol only
    'max_iter': 2,
    # GslMol: graph_learner
    'graph_learn_type': 'gat_attention',
    'graph_learn_hidden_size': 128,
    'graph_learn_topk': None,
    'graph_learn_epsilon': 0.7,
    'graph_learn_num_pers': 8,
    'graph_skip_conn': 0.8,
    'graph_include_self': False,
    'update_adj_ratio': 0.1,

    # GslMol: gnn_encoder
    'gnn_type': 'gcn',
    'gnn_layer': 2,
    'emb_dim': 128,
    'gnn_batch_norm': True,
    'graph_pooling': 'mean',

    'dropout': 0,
    'gl_dropout': 0,
}

if __name__ == '__main__':
    app.run()
