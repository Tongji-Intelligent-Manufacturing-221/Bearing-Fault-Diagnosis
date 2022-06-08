import argparse
import torch
import dataloader
from models import main_models
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
import os
parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches_1',type=int,default=50)#50
parser.add_argument('--n_epoches_2',type=int,default=100)
parser.add_argument('--n_epoches_3',type=int,default=50)
parser.add_argument('--n_target_samples',type=int,default=1)

opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
manual_seed = random.randint(1, 10000)
#manual_seed = 6378
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(manual_seed)  # 为GPU设置随机种子
torch.backends.cudnn.deterministic = True
print("manual_seed",manual_seed)

class MyData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        assert index < len(self.data)
        return torch.Tensor([self.data[index]]), self.labels[index]

    def __len__(self):
        return len(self.data)

    def get_tensors(self):
        return torch.Tensor([self.data]), torch.Tensor(self.labels)


def split_Train_Test_Data(data_dir, name, ratio):
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    dataset_kinds = os.listdir(data_dir)
    kinds_num = len(dataset_kinds)

    for i in range(kinds_num):
        path = data_dir + "\\" + str(i) + "\\" + name + ".npz"
        data_dict = np.load(path)
        sample_data = data_dict.files

        num_sample_train = int(len(sample_data) * ratio[0])
        random.shuffle(sample_data)  # 打乱后抽取

        for x in sample_data[:num_sample_train]:
                train_data.append(data_dict[x])
                train_labels.append(i)
        for x in sample_data[num_sample_train:]:
            val_data.append(data_dict[x])
            val_labels.append(i)

    length_train = len(train_data)
    length_val = len(val_data)

    train_dataloader = DataLoader(MyData(train_data, train_labels),
                                      batch_size=10, shuffle=True)
    val_dataloader = DataLoader(MyData(val_data, val_labels),
                                    batch_size=10, shuffle=False)
    return train_dataloader, val_dataloader, length_train, length_val


s_data_dir_train_val="G:\\数据库\\SU\\SU_data\\train\\30_2"
s_train_dataloader,_,_,_ =split_Train_Test_Data(s_data_dir_train_val,'train_data',[0.5,0.5])

t_data_dir_val = 'G:\\数据库\\SU\\SU_data\\train\\20_0'
t_train_dataloader, _,_,_ = split_Train_Test_Data(t_data_dir_val, 'train_data',[0.01,0.99])

s_data_dir_test = 'G:\\数据库\\SU\\SU_data\\test\\30_2'
s_test_dataloader, _,_,_ = split_Train_Test_Data(s_data_dir_test, 'test_data',[1,0])

t_data_dir_test = 'G:\\数据库\\SU\\SU_data\\test\\20_0'
t_test_dataloader, _,_,_ = split_Train_Test_Data(t_data_dir_test, 'test_data',[1,0])

#--------------pretrain g and h for step 1---------------------------------
classifier=main_models.Classifier()
encoder=main_models.Encoder()
discriminator=main_models.DCD(input_features=1024)

classifier.to(device)
encoder.to(device)
discriminator.to(device)

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0005,betas=(0.5,0.9))

for epoch in range(opt['n_epoches_1']):
    encoder.train()
    classifier.train()
    for data,labels in s_train_dataloader:
        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        y_pred=classifier(encoder(data))

        loss=loss_fn(y_pred,labels)
        loss.backward()

        optimizer.step()

    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        acc = 0
        for data, labels in s_test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(s_test_dataloader)), 3)

    print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
#-------------------------------------------------------------------


X_s,Y_s=dataloader.sample_data(s_train_dataloader)
X_t,Y_t=dataloader.create_target_samples(t_train_dataloader,opt['n_target_samples'])

#-----------------train DCD for step 2--------------------------------

optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.0005)


for epoch in range(opt['n_epoches_2']):
    # data
    groups,aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)

    n_iters = 4 * len(groups[1])

    index_list = torch.randperm(n_iters)
    mini_batch_size=20 #use mini_batch train can be more stable


    loss_mean=[]

    X1=[];X2=[];ground_truths=[]
    for index in range(n_iters):

        ground_truth=index_list[index]//len(groups[1])

        x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        #select data for a mini-batch to train
        if (index+1)%mini_batch_size==0:
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths=torch.LongTensor(ground_truths)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths=ground_truths.to(device)

            optimizer_D.zero_grad()
            X_cat=torch.cat([encoder(X1),encoder(X2)],1)
            y_pred=discriminator(X_cat.detach())
            loss=loss_fn(y_pred,ground_truths)
            loss.backward()
            optimizer_D.step()
            loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    print("step2----Epoch %d/%d loss:%.3f"%(epoch+1,opt['n_epoches_2'],np.mean(loss_mean)))

#----------------------------------------------------------------------

#-------------------training for step 3-------------------
optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0005)
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.0005)


for epoch in range(opt['n_epoches_3']):
    #---training g and h , DCD is frozen

    groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=opt['n_epoches_2']+epoch)
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]

    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)

    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)

    mini_batch_size_g_h = 10 #data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= 20 #data contains G1,G2,G3,G4 so use 40 as mini_batch
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]
    for index in range(n_iters):


        ground_truth=index_list[index]//len(G2)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        # y1=torch.LongTensor([y1.item()])
        # y2=torch.LongTensor([y2.item()])
        dcd_label=0 if ground_truth==0 else 2
        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)
        dcd_labels.append(dcd_label)

        if (index+1)%mini_batch_size_g_h==0:

            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1=torch.LongTensor(ground_truths_y1)
            ground_truths_y2 = torch.LongTensor(ground_truths_y2)
            dcd_labels=torch.LongTensor(dcd_labels)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1=ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            dcd_labels=dcd_labels.to(device)

            optimizer_g_h.zero_grad()

            encoder_X1=encoder(X1)
            encoder_X2=encoder(X2)

            X_cat=torch.cat([encoder_X1,encoder_X2],1)
            y_pred_X1=classifier(encoder_X1)
            y_pred_X2=classifier(encoder_X2)
            y_pred_dcd=discriminator(X_cat)

            loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
            loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
            loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

            loss_sum = loss_X1 + loss_X2 + 0.05 * loss_dcd
            #loss_sum = loss_X1 + loss_X2

            loss_sum.backward()
            optimizer_g_h.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []


    #----training dcd ,g and h frozen
    X1 = []
    X2 = []
    ground_truths = []

    for index in range(n_iters_dcd):

        ground_truth=index_list_dcd[index]//len(groups[1])

        x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index + 1) % mini_batch_size_dcd == 0:
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            ground_truths = torch.LongTensor(ground_truths)
            X1 = X1.to(device)
            X2 = X2.to(device)
            ground_truths = ground_truths.to(device)

            optimizer_d.zero_grad()
            X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
            y_pred = discriminator(X_cat.detach())
            loss = loss_fn(y_pred, ground_truths)
            loss.backward()
            optimizer_d.step()
            # loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    #testing
    acc = 0
    for data, labels in t_test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(t_test_dataloader)), 3)

    print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_3'], accuracy))
        





















