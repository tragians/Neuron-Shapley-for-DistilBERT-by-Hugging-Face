###############################################################################
# imports
import os
import time
import sys
import torch
from transformers import DistilBertForSequenceClassification
from google.colab import drive
import numpy as np
from dbtokenizer import CustomTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score

# initialise some parameters
PATH_TO_MODEL_DIR = "/content/gdrive/MyDrive/0db/"
CHECKPOINT = "/content/gdrive/MyDrive/0db/distilbert_trained_model.pt"
BATCH_SIZE = 32 # reduce if problems arise!! 
TIME_START = time.time()
NUM_CLASSES = 4 
NUM_SAMPLES = 64

time.sleep(10 * np.random.random())


def remove_players(model, players):
    '''Remove selected players (filters) in the DistilBERT model.'''
    if isinstance(players, str):
        # case of just one player 
        players = [players]
    for player in players:

        neuron_idx = int(player.split('_')[-1])
        # zeroing out the weights of selected players (neurons)
        with torch.no_grad():
            model.pre_classifier.weight[neuron_idx].zero_()
        
        # layer normalization istead of batch normalization:
        # relevant only for encoder/decoder layers
        # should be consired and implemented if these layers are included
        # to the players 
        

def one_iteration(
    model, 
    players,
    test_input_ids, 
    test_attention_mask, 
    test_labels_tensor,
    base_value,
    batch_size = BATCH_SIZE,
    chosen_players=None,
    c=None, 
    truncation=None
):
    '''One iteration of Neuron-Shapley algoirhtm.'''
    # Original performance of the model with all players present.
    init_val = accuracy (model, test_input_ids, test_attention_mask, 
    
    test_labels_tensor, device, batch_size = BATCH_SIZE)
    if c is None:
        c = {i: np.array([i]) for i in range(len(players))}
    elif not isinstance(c, dict):
        c = {i: np.where(c==i)[0] for i in set(c)}
    if truncation is None:
        truncation = len(c.keys())
    if chosen_players is None:
        chosen_players = np.arange(len(c.keys()))
    # A random ordering of players
    idxs = np.random.permutation(len(c.keys()))
    # -1 default value for players that have already converged
    marginals = -np.ones(len(c.keys()))
    marginals[chosen_players] = 0.
    t = time.time()
    truncation_counter = 0
    old_val = copy.copy(init_val)

    for n, idx in enumerate(idxs[::-1]):
        if idx in chosen_players:
            if old_val is None:
                old_val = accuracy(model,test_input_ids, test_attention_mask, test_labels_tensor,
                device, batch_size = BATCH_SIZE)

            remove_players(model, players[c[idx]])
            new_val = accuracy(model, test_input_ids, test_attention_mask, test_labels_tensor,
                device, batch_size = BATCH_SIZE)
            
            marginals[c[idx]] = (old_val - new_val) / len(c[idx])
            old_val = new_val

            if isinstance(truncation, int):
                if n >= truncation:      
                    break
            else:
                if n%10 == 0:
                    print(n, time.time() - t, new_val)
                val_diff = new_val - base_value
                truncation_counter += 1
                if truncation_counter > 5:
                    break
        else:
            old_val = None
            remove_players(model, players[c[idx]])        
    return idxs.reshape((1, -1)), marginals.reshape((1, -1))


def accuracy(model, test_input_ids, test_attention_mask, test_labels_tensor, device, batch_size = BATCH_SIZE):
    '''Runs inference on provided model and returns accuracy score'''
    # Create the DataLoader.
    prediction_data = TensorDataset(test_input_ids, test_attention_mask, test_labels_tensor)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)
    
    # Switch to evaluation mode for inference
    model.eval()
    val = 0.
    #Tracking variables
    predictions, true_labels = [], []

    #Predict
    for batch in prediction_dataloader:
        
        batch = tuple(t.to(device) for t in batch)

        #Unpack from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            #Forwrd pass, calculate logit predictions
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        #Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

        flat_predictions = np.concatenate(predictions, axis=0)
        #For each sample, pick the label with the higher score
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = np.concatenate(true_labels, axis=0)


        val = accuracy_score(flat_true_labels, flat_predictions)
        
    return val

def prep_data():
    # Import a specified number of samples from the testing data 
    # to use while evaluating the importance of selected neurons to the performance of the model
    
    #Load test data
    ag_news_test = load_dataset("ag_news", split='test')

    #Split testing data into article headlines & labels
    test_titles = [row['text'] for row in ag_news_test][:NUM_SAMPLES]
    test_labels = [row['label'] for row in ag_news_test][:NUM_SAMPLES]

    #### Tokenize Test Set
    #Set up & run our tokenizer
    tokenizer = CustomTokenizer()
    max_token_length = 379
    tokenizer.tokenize(test_titles, max_token_length)

    #Retrieve tokenized input features & attention mask
    test_input_ids = tokenizer.get_tokenized_input_features()
    test_attention_mask = tokenizer.get_attention_mask()

    #Form a tensor from our training labels
    test_labels_tensor = torch.tensor(test_labels)

    return test_input_ids, test_attention_mask,test_labels_tensor


### GETTING STARTED 
if __name__ == "__main__":
    #Utilise GPU for training/testing, if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU Available: ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU Available, Switching To CPU.")

    # mount google drive 
    drive.mount('/content/gdrive')
    sys.path.append(PATH_TO_MODEL_DIR)
    # change current working directory to /0db, where are Tokenizer module is 

    ####Load trained model to manipulate its parameters
    # Standard DistilBERT Class (pre-trained, standard 6 layers)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels = 4,
        output_attentions = False,
        output_hidden_states = False,
    ).cuda()

    model.load_state_dict(torch.load(CHECKPOINT))

    #### Making preparations for saving the produced results

    ## Experiment Directory

    experiment_dir = "/content/gdrive/MyDrive/0db/NShap/results/distilBERT/"

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    ## CB directory

    experiment_name = 'cb_{}_{}_{}'.format(bound, truncation, NUM_SAMPLES)

    cb_dir = os.path.join(experiment_dir, experiment_name)
    if not os.path.isdir(cb_dir):
        os.makedirs(cb_dir)

    # Create a list of all available layers
    # similar to convs from cb_run.py in neuronshapley,
    layers = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            layers.append(name.replace(".weight", ""))

    ## Load the list of all players (neurons) in layer "pre_classifier" else save

    players = []    
    var_dic = {name.replace(".weight", ""): list(param.shape)[0] 
            for name, param in model.named_parameters() if "weight" in name }

    for layer in layers:
        if "pre_classifier" in str(layer):
            players.append(['{}_{}'.format(layer, i) for i in
                            range(var_dic[layer])])

    players = np.sort(np.concatenate(players))
        
    ## Load metric's base value (random performance)
    # metric here is accuracy
    base_value = 1./NUM_CLASSES

    ## Assign expriment number to this specific run of cb_run.py
    results = [saved for saved in os.listdir(cb_dir)
            if 'idxs' in saved]
    experiment_number = 0
    if len(results):
        experiment_number = len(results)
    print(experiment_number)

    #### Run one iteration of the code

    ## Running CB-Shapley
    ## CB stands for confidense bound

    ## Load the list of players (filters) that are determined to be not confident enough
    ## by the cb_aggregate.py running in parallel to this script

    t_init = time.time()
    test_input_ids, test_attention_mask,test_labels_tensor = prep_data()

    idxs, vals =  one_iteration(
            model, 
            players,
            test_input_ids, 
            test_attention_mask, 
            test_labels_tensor,
            base_value,
            batch_size = BATCH_SIZE,
            chosen_players=None,
            c=None, 
            truncation=None
    )

    # idxs contains information regarding the order in which the players where 
    # removed in this iteration
    # vals contains the computed contribution for each player 
    idx_path = 'idxs_' + str(experiment_number) + '.txt'
    vals_path = 'vals_' + str(experiment_number) + '.txt'

    np.savetxt(os.path.join(cb_dir, idx_path), idxs, fmt= '%i', delimiter= ",")
    np.savetxt(os.path.join(cb_dir, vals_path ), vals, fmt='%1.3f', delimiter= ",")

    #print ellapsed time
    print(time.time() - t_init, time.time() - TIME_START)

