import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import model, dataset
from core.utils import init_log, progress_bar
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

if __name__ == '__main__':
    test_loss = 0
    test_correct = 0
    total = 0
    auc_label_lst = []
    auc_pred_lst = []
    people_lst = []
    img_vis_lst = []
    file_name_lst = []
    anchor_lst = []


    n_class = 2
    net = model.attention_net(topN=PROPOSAL_NUM, n_class=n_class)
    if resume:
        ckpt = torch.load(resume, map_location=torch.device('cuda'))
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1


    test_path = '/home/edwardzhu0211/CS598-DLH/local_traniner/input/test'
    testset = dataset.SARS(root=test_path, is_train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=2, drop_last=False)
    creterion = torch.nn.CrossEntropyLoss()



    for i, data in enumerate(testloader):

    # =============================================================================
    #             if i < 1:
    #                 continue
    # =============================================================================
        with torch.no_grad():
            img, label, img_raw = data[0].cuda(), data[1].cuda(), data[2]
            batch_size = img.size(0)
            _, concat_logits, _, _, _ = net(img, img_raw, False, False)
            # calculate loss
            concat_loss = creterion(concat_logits, label)
            # calculate accuracy
            _, concat_predict = torch.max(concat_logits, 1)
            auc_label_lst += list(label.data.cpu().numpy())
            pred = torch.nn.Softmax(1)(concat_logits)
            auc_pred_lst.append(pred.data.cpu().numpy())
            people_lst.append(data[3])
            file_name_lst += list(data[4])
    # =============================================================================
    #                 img_vis_lst.append(img_vis)
    #                 anchor_lst.append(anchor)
    # =============================================================================

            total += batch_size
            test_correct += torch.sum(concat_predict.data == label.data)
            test_loss += concat_loss.item() * batch_size
            progress_bar(i, len(testloader), 'eval test set')
    test_acc = float(test_correct) / total
    test_loss = test_loss / total
    print(f'auc: {roc_auc_score(auc_label_lst, np.concatenate(auc_pred_lst, 0)[:, 1]):.4f}')
    print(test_acc)
    print(test_loss)