import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch
import matplotlib.pyplot as plt
from torch import autograd
import os
import math
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./runs")

cuda = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

classes = {
        0:"ball",
        1:"comb",
        2:"health",
        3:"inner",
        4:"outer"
    }

version_higher = (torch.__version__ >= "1.5.0")

#manual_seed = random.randint(1, 10000)
manual_seed = 1997
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(manual_seed)  # 为GPU设置随机种子
torch.backends.cudnn.deterministic = True
print("manual_seed",manual_seed)


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch



    Arguments:

        params (iterable): iterable of parameters to optimize or dicts defining

            parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing

            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve

            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        amsgrad (boolean, optional): whether to use the AMSGrad variant of this

            algorithm from the paper `On the Convergence of Adam and Beyond`_

            (default: False)

        weight_decouple (boolean, optional): ( default: False) If set as True, then

            the optimizer uses decoupled weight decay as in AdamW

        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple

            is set as True.

            When fixed_decay == True, the weight decay is performed as

            $W_{new} = W_{old} - W_{old} \times decay$.

            When fixed_decay == False, the weight decay is performed as

            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the

            weight decay ratio decreases with learning rate (lr).

        rectify (boolean, optional): (default: False) If set as True, then perform the rectified

            update similar to RAdam



    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients

               NeurIPS 2020 Spotlight

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay, amsgrad=amsgrad)

        super(AdaBelief, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple

        self.rectify = rectify

        self.fixed_decay = fixed_decay

        if self.weight_decouple:

            print('Weight decoupling enabled in AdaBelief')

            if self.fixed_decay:
                print('Weight decay fixed')

        if self.rectify:
            print('Rectification enabled in AdaBelief')

        if amsgrad:
            print('AMS enabled in AdaBelief')

    def __setstate__(self, state):

        super(AdaBelief, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):

        for group in self.param_groups:

            for p in group['params']:

                state = self.state[p]

                amsgrad = group['amsgrad']

                # State initialization

                state['step'] = 0

                # Exponential moving average of gradient values

                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values

                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values

                    state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)\
                        if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):

        """Performs a single optimization step.



        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')

                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization

                if len(state) == 0:

                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(
                        p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values

                        state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                            if version_higher else torch.zeros_like(p.data)

                # get current state variable

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                # perform weight decay, check if decoupled weight decay

                if self.weight_decouple:

                    if not self.fixed_decay:

                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                    else:

                        p.data.mul_(1.0 - group['weight_decay'])

                else:

                    if group['weight_decay'] != 0:
                        grad.add_(p.data,alpha=group['weight_decay'])

                # Update first and second moment running average

                exp_avg.mul_(beta1).add_(grad,alpha=1 - beta1)

                grad_residual = grad - exp_avg

                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual,value=1 - beta2)

                if amsgrad:

                    max_exp_avg_var = state['max_exp_avg_var']

                    # Maintains the maximum of all 2nd moment running avg. till now

                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient

                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                else:

                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:

                    # Default update

                    step_size = group['lr'] / bias_correction1

                    p.data.addcdiv_(exp_avg, denom,value=-step_size)



                else:  # Rectified update

                    # calculate rho_t

                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (

                            1.0 - beta2 ** state['step'])

                    if state['rho_t'] > 4:  # perform Adam style update if variance is small

                        rho_inf, rho_t = state['rho_inf'], state['rho_t']

                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t

                        rt = math.sqrt(rt)

                        step_size = rt * group['lr'] / bias_correction1

                        p.data.addcdiv_(exp_avg, denom, value=-step_size)



                    else:  # perform SGD style update

                        p.data.add_(-group['lr'], exp_avg)

        return loss


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 13 * 13, 500)

    def forward(self, x):
        conv_out = self.encoder(x)
        feat = self.fc1(conv_out.view(-1, 50 * 13 * 13))

        return feat


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=500, out_features=5)
        )

    def forward(self, x):
        output = self.classifier(x)

        return output


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(500, 500))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU())
        self.domain_classifier.add_module('d_fc2', nn.Linear(500, 1))

    def forward(self, input):
        """Forward the discriminator."""
        out = self.domain_classifier(input)
        return out

class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(reduction='sum', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss


def weight_init(m):

    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        xavier_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        xavier_uniform_(m.weight.data)


def batch_norm_init(m):

    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.reset_running_stats()

def train(dataset_s, dataset_val, current_epoch, epochs, encoder, classifier):
    torch.cuda.empty_cache()
    encoder.train()
    classifier.train()
    length = len(dataset_s.dataset)

    lr = 0.0005
    optimizer = AdaBelief(list(encoder.parameters()) + list(classifier.parameters()),
                          lr=lr, weight_decay=0.001)
    criterion = NMTCritierion()

    for index, (s_data_train, s_label_train) in enumerate(dataset_s):

        s_data_train = s_data_train.float().cuda()
        s_label_train = s_label_train.long().cuda()

        s_clas_out_train = classifier(encoder(s_data_train))

        loss_c = criterion(s_clas_out_train, s_label_train)
        loss = loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(s_clas_out_train.data, 1)
        correct = pred.eq(s_label_train).cpu().sum()
        acc = 100. * correct.item() / len(s_data_train)

        if index % 2 == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)] \t Loss_c: {:.6f}  Acc: {:.2f}%'.format
                  (current_epoch, epochs, (index + 1) * len(s_data_train), length, 100. * (batchsize * index / length),
                   loss_c.item(), acc))

    # model_eval
    encoder.eval()
    classifier.eval()
    length_val = len(dataset_val.dataset)
    correct = 0
    sum_loss = 0
    for _, (s_data_val, s_label_val) in enumerate(dataset_val):
        with torch.no_grad():
            s_data_val = s_data_val.float().cuda()
            s_label_val = s_label_val.long().cuda()

            s_clas_out_val = classifier(encoder(s_data_val))
            loss_val_c = criterion(s_clas_out_val, s_label_val)
            pred_val = torch.argmax(s_clas_out_val, 1)
            correct += pred_val.eq(s_label_val).cpu().sum()
            sum_loss += loss_val_c

    acc = 100. * correct.item() / length_val
    average_loss = sum_loss.item() / length_val
    print('\n The {}/{} epoch result : Average loss: {:.6f}, Acc_val: {:.2f}%'.format(
        current_epoch, epochs, average_loss, acc
    ))

    return encoder, classifier


def test(dataset, encoder, classifier):
    encoder.eval()
    classifier.eval()
    length = len(dataset.dataset)
    correct = 0
    feature = []

    for index, (data_test, label_test) in enumerate(dataset):
        with torch.no_grad():
            data_test = data_test.float().cuda()
            label_test = label_test.long().cuda()
            fea = encoder(data_test)
            out = classifier(fea)
            pred = torch.argmax(out.data, 1)
            feature.append(fea.detach().cpu().numpy())
            correct += pred.eq(label_test).cpu().sum()
    acc = 100. * correct.item() / length
    return acc, np.array(feature)


def calc_gradient_penalty(netD, real_data, fake_data):

    # print "real_data: ", real_data.size(), fake_data.size()

    alpha = torch.rand(fake_data.shape[0], 1)

    alpha = alpha.expand(fake_data.shape[0], int(real_data.nelement()/len(real_data)))

    alpha = alpha.cuda()

    index = np.random.permutation(len(real_data))

    index = index[:len(fake_data)]

    real_data = real_data[index]

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,

                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),

                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradient_penalty


def train_tgt(src_encoder, tgt_encoder, critic, dataset_s, dataset_t):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    optimizer_tgt = AdaBelief(tgt_encoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_critic = AdaBelief(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    one = torch.FloatTensor([1]).cuda()
    mone = -1 * one
    mone.cuda()

    loss_source_d = []
    loss_target_d = []
    loss_gen = []

    ####################
    # 2. train network #
    ####################

    # zip source and target data pair
    data_zip = enumerate(zip(dataset_s, dataset_t))

    for step, ((images_src, _), (images_tgt)) in data_zip:

        for p in critic.parameters():  # reset requires_grad
            p.requires_grad = True

        ###########################
        # 2.1 train discriminator #
        ###########################
        # make images variable
        images_src = images_src.float().cuda()
        images_tgt = images_tgt[0].float().cuda()

        # extract and concat features
        feat_src = src_encoder(images_src)
        feat_tgt = tgt_encoder(images_tgt)

        # predict on discriminator
        pred_src = critic(feat_src)
        pred_tgt = critic(feat_tgt.detach())

        optimizer_critic.zero_grad()

        gradient = calc_gradient_penalty(critic, feat_src, feat_tgt)
        gradient.backward()

        loss_d_src = pred_src.mean().reshape(-1)
        loss_d_src.backward(one)

        loss_d_tgt = pred_tgt.mean().reshape(-1)
        loss_d_tgt.backward(mone)

        optimizer_critic.step()

        ############################
        # 2.2 train target encoder #
        ############################
        # zero gradients for optimizer

        if (step + 1) % 2 == 0:

            for p in critic.parameters():
                p.requires_grad = False

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)
            loss_g = pred_tgt.mean().reshape(-1)

            optimizer_tgt.zero_grad()
            loss_g.backward(one)
            optimizer_tgt.step()

            loss_gen.append(loss_g)

        loss_source_d.append(loss_d_src)
        loss_target_d.append(loss_d_tgt)
        #######################
        # 2.3 print step info #
        #######################

        if (step + 1) % 4 == 0:
            print("Epoch [{}/{}] Step [{}/{}]:d_loss_s={:.5f}  d_loss_t={:.5f} g_loss={:.5f}".format
                  (epoch + 1, num_epochs, step + 1, len(dataset_s), loss_d_src.item(), loss_d_tgt.item(),
                   loss_g.item()))

    return tgt_encoder, loss_source_d, loss_target_d, loss_gen


if __name__ == '__main__':
    torch.cuda.empty_cache()


    # load dataset
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
                                      batch_size=20, shuffle=True)
        val_dataloader = DataLoader(MyData(val_data, val_labels),
                                    batch_size=20, shuffle=False)
        return train_dataloader, val_dataloader, length_train, length_val


    s_data_dir_train_val = "G:\\数据库\\SU\\SU_data\\train\\30_2"
    s_train_dataloader, s_val_dataloader, s_train_length, s_val_length = split_Train_Test_Data(s_data_dir_train_val,
                                                                                               'train_data', [1, 0])

    t_data_dir_val = 'G:\\数据库\\SU\\SU_data\\train\\20_0'
    t_train_dataloader, t_val_dataloader, _, t_val_length = split_Train_Test_Data(t_data_dir_val, 'train_data',
                                                                                  [1, 0])

    s_data_dir_test = 'G:\\数据库\\SU\\SU_data\\test\\30_2'
    s_test_dataloader, _, s_test_length, _ = split_Train_Test_Data(s_data_dir_test, 'test_data', [1, 0])

    t_data_dir_test = 'G:\\数据库\\SU\\SU_data\\test\\20_0'
    t_test_dataloader, _, t_test_length, _ = split_Train_Test_Data(t_data_dir_test, 'test_data', [1, 0])

    test_t = []
    test_n = []
    cal_epoch = []

    for iteration in range(1):

        src_encoder = Encoder()
        src_classifier = Classifier()
        tgt_encoder = Encoder()
        critic = Discriminator()

        src_encoder.cuda()
        src_classifier.cuda()
        tgt_encoder.cuda()
        critic.cuda()

        src_encoder.apply(weight_init)
        src_classifier.apply(weight_init)
        tgt_encoder.apply(weight_init)
        critic.apply(weight_init)

        src_encoder.apply(batch_norm_init)
        tgt_encoder.apply(batch_norm_init)

        batchsize=20
        epochs = 50
        num_epochs = 100


        loss_d_epochs = []
        loss_g_epochs = []

        test_acc_s = []
        test_acc_t = []

        for current_epoch in range(epochs):
            src_encoder, src_classifier = train(s_train_dataloader, s_test_dataloader,
                                                current_epoch, epochs, src_encoder, src_classifier)
        torch.save(src_encoder.state_dict(), "weight/src_encoder.pth")
        torch.save(src_classifier.state_dict(), "weight/src_classifier.pth")

        s_test_acc, _ = test(s_test_dataloader, src_encoder, src_classifier)

        tgt_encoder.load_state_dict(src_encoder.state_dict())
        t_test_acc, _ = test(t_test_dataloader, tgt_encoder, src_classifier)

        print('source_test_acc: {:.5f}%  target_test_acc: {:.5f}%'.format(s_test_acc, t_test_acc))

        print("=== Training encoder for target domain ===")
        best_target_test_acc=0

        for epoch in range(num_epochs):
            tgt_encoder, l_d_s, l_d_t, l_g = train_tgt(src_encoder, tgt_encoder, critic,
                                                       s_train_dataloader, t_train_dataloader)

            s_test_acc, s_f = test(s_test_dataloader, src_encoder, src_classifier)

            t_test_acc, t_f = test(t_test_dataloader, tgt_encoder, src_classifier)
            print('\n The {}/{} epoch result : t_test_acc: {:.2f}%'.format(
                epoch+1, num_epochs, t_test_acc
            ))

            writer.add_scalar("Target domain Test Accuracy Rate",t_test_acc,epoch)

            test_acc_t.append(t_test_acc)

            if t_test_acc > best_target_test_acc:
                best_target_test_acc = t_test_acc
                torch.save(tgt_encoder.state_dict(), "weight/tgt_encoder_best.pth")

            if epoch == num_epochs-1:
                torch.save(tgt_encoder.state_dict(), "weight/tgt_encoder_final.pth")

        print("The best target test acc",best_target_test_acc)
        # plt.plot(test_acc_t, 'g-', marker='*')
        # plt.xlabel('Epochs')
        # plt.ylabel('Test accuracy')
        # plt.title('Target domain Test Accuracy Rate')
        # plt.show()
            # if t_test_acc > 99:
            #     s_f = s_f.reshape(-1, 500)
            #     num_s = np.random.randint(low=0, high=100, size=1)
            #     plt.imshow(s_f[num_s].reshape(5, -1), cmap='GnBu')
            #     plt.title('class_s', fontdict={'weight':'normal','size': 25})
            #     plt.show()
            #
            #     t_f = t_f.reshape(-1, 500)
            #     plt.imshow(t_f)
            #     plt.show()
            #     num_t = np.random.randint(low=0, high=100, size=3)
            #     print(num_t)
            #     plt.imshow(t_f[num_t[0]].reshape(5, -1), cmap='GnBu')
            #     plt.title('class_t', fontdict={'weight':'normal','size': 25})
            #     plt.show()
            #     plt.imshow(t_f[num_t[1]].reshape(5, -1), cmap='GnBu')
            #     plt.title('class_t', fontdict={'weight':'normal','size': 25})
            #     plt.show()
            #     plt.imshow(t_f[num_t[1]].reshape(5, -1), cmap='GnBu')
            #     plt.title('class_t', fontdict={'weight':'normal','size': 25})
            #     plt.show()
            #
            #     break
                # plt.rcParams.update({'font.size': 20})
                # plt.legend(loc='upper right')
                # plt.show()



