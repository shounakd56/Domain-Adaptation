import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD,NAdam,AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap 
import matplotlib.pyplot as plt

import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BackBone(object):

    def lateral():
        lateral_2 = nn.Conv2d(256, 256, 1)
        lateral_3 = nn.Conv2d(512, 256, 1)
        lateral_4 = nn.Conv2d(1024, 256, 1)
        lateral_5 = nn.Conv2d(2048, 256, 1)

        return lateral_5, lateral_4, lateral_3, lateral_2
   

    def conv():
        resnet101_M = resnet101(pretrained=True)

        model = list(resnet101_M.children())

        conv_1 = nn.Sequential(*model[:3])
        conv_2 = nn.Sequential(*([model[3]] + list(model[4].children())))
        conv_3 = model[5]
        conv_4 = model[6]
        conv_5 = model[7]

        return conv_1, conv_2, conv_3, conv_4, conv_5
        

class Fpn(nn.Module):
    def __init__(self):
        super(Fpn, self).__init__()

        #self.Upsample = nn.Upsample(size=(), mode='nearest')
        self.Conv_3x3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_c1, self.conv_c2, self.conv_c3, self.conv_c4, self.conv_c5 = nn.Sequential(*BackBone.conv())
        self.lateral_c5, self.lateral_c4, self.lateral_c3, self.lateral_c2 = nn.Sequential(*BackBone.lateral())
    def forward(self, x):
        c1 = self.conv_c1(x)
        c2 = self.conv_c2(c1)
        c3 = self.conv_c3(c2)
        c4 = self.conv_c4(c3)
        c5 = self.conv_c5(c4)

        p5 = self.lateral_c5(c5)
        b_p4 = self.lateral_c4(c4) + F.interpolate(input=p5, size=(c4.shape[2], c4.shape[3]), mode='nearest')
        b_p3 = self.lateral_c3(c3) + F.interpolate(input=b_p4, size=(c3.shape[2], c3.shape[3]), mode='nearest')
        b_p2 = self.lateral_c2(c2) + F.interpolate(input=b_p3, size=(c2.shape[2], c2.shape[3]), mode='nearest')
   
        p4 = self.Conv_3x3(b_p4)
        p3 = self.Conv_3x3(b_p3)
        p2 = self.Conv_3x3(b_p2)

        p6 = nn.MaxPool2d(1, 2)(p5)

        return p2, p3, p4, p5, p6

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def guassiankernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

class TWeight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=65):
        batch_size = s_label.size()[0]

        s_sca_label = s_label.cpu().data.max(1)[1].numpy()
        s_vec_label = s_label.cpu().data.numpy()
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

def tlmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = TWeight.cal_weight(s_label, t_label, type='visual')
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassiankernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    return loss

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=31):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def im(outputs_test, gent=True):
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    test_iter=ForeverDataIterator(test_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    #backbone=ConvMixer(1024, 2, kernel_size=9, patch_size=7, n_classes=31)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler

    optimizer = AdamW(all_parameters, args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)

    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args)
        if epoch>-1:
                    feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        #target_feature = collect_feature(train_target_loader, feature_extractor, device)
                    test_feature = collect_feature(test_loader, feature_extractor, device)
        #combined_features = np.vstack((source_feature, target_feature))
                    all_labels2 = []

                    for i in range(len(test_loader)):
                        x_t, labels_t = next(test_iter)[:2]
                        labels_t=labels_t.to(device)
                        all_labels2.append(labels_t.cpu().numpy())
                    all_labels_tgt = np.concatenate(all_labels2, axis=0)
                    unique_labels = np.unique(all_labels_tgt)
        #unique_labels2 = np.unique(all_labels_src)
                    print(unique_labels)
                    sample_size_per_class =60  # Adjust the number of samples per class as needed
                    sampled_indices = []
                    sampled_indices2 = []
                    for label in range(65):
                        label_indices = np.where(all_labels_tgt == label)[0]
                        sampled_indices.extend(np.random.choice(label_indices, size=sample_size_per_class, replace=True))

                    sampled_features = test_feature[sampled_indices]
                    sampled_labels = all_labels_tgt[sampled_indices]
    #    sampled_features2 = source_feature[sampled_indices2]
     #   sampled_labels2 = all_labels_src[sampled_indices2]
# Perform t-SNE dimensionality reduction
     #   tsne = TSNE(n_components=2, random_state=0)
      #  reduced_features = tsne.fit_transform(sampled_features)
                    tsne = TSNE(n_components=2, random_state=0)
                    reduced_features = tsne.fit_transform(sampled_features)
                #    reduced_features2 = tsne.fit_transform(sampled_features2)
            # Get unique class labels


                    cmap1= get_cmap('tab20')
                    cmap2= get_cmap('tab20b')
                    cmap3= get_cmap('tab20c')
                    cmap4= get_cmap('tab10')

            # Create a scatter plot with different colors for each class in both source and target domains
                    plt.figure(figsize=(10, 8))

                    for i, label in enumerate(unique_labels):

            #                    cls="Benign"
                            if i<20:
                                label_indices = np.where(sampled_labels == label)
             #                   label_indices2 = np.where(sampled_labels2 == label)
                                plt.scatter(reduced_features[label_indices, 0],
                                reduced_features[label_indices, 1],
                                color=cmap1(i),
                                label=label,
                                marker="*"
                                 )
                            elif i<40:
                                label_indices = np.where(sampled_labels == label)
             #                   label_indices2 = np.where(sampled_labels2 == label)
                                plt.scatter(reduced_features[label_indices, 0],
                                reduced_features[label_indices, 1],
                                color=cmap2(i-20),
                                label=label,
                                marker="*"
                                 )
                            elif i<60:
                                label_indices = np.where(sampled_labels == label)
             #                   label_indices2 = np.where(sampled_labels2 == label)
                                plt.scatter(reduced_features[label_indices, 0],
                                reduced_features[label_indices, 1],
                                color=cmap3(i-40),
                                label=label,
                                marker="*"
                                 )
                            else:
                                label_indices = np.where(sampled_labels == label)
             #                   label_indices2 = np.where(sampled_labels2 == label)
                                plt.scatter(reduced_features[label_indices, 0],
                                reduced_features[label_indices, 1],
                                color=cmap4(i-60),
                                label=label,
                                marker="*"
                                 )

                  #  plt.legend()
                 #   plt.title("t-SNE Visualization of Source and Target Classes")
                  #  plt.xlabel("t-SNE Dimension 1")
                   # plt.ylabel("t-SNE Dimension 2")
                    plt.savefig('./Transfer-Learning-Library/examples/domain_adaptation/'+'offHPR_60_untrained' + str(epoch) + '.png')


        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    MMD_Loss=AverageMeter('MMD Loss', ':3.1f')
    mccloss=AverageMeter('mcc Loss', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs,MMD_Loss,mccloss],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
       # x_t, = next(train_target_iter)[:1]
        x_t,labels_t = next(train_target_iter)[:2]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        ab=labels_s.numpy()
        cd=labels_t.numpy()

        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        logits_s,x_s=model(x_s)
        logits_t,x_t = model(x_t )        
        trial=(MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5))
        mmd=trial(x_s,x_t)
        outputs_target_temp = y_t /2.5
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = 32 * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / 65

        cls_loss = F.cross_entropy(y_s, labels_s)
       # labels=labels_s.view(16,1).float()
        #s_labels = torch.tensor(labels, dtype=torch.float32).cuda()
        #labelt=labels_t.view(16,1).float()
        #s_labelt = torch.tensor(labelt, dtype=torch.float32).cuda()

       # print(labels.dtype)num_classes = 31

# Convert labels_list into a 16x31 matrix (one-hot encoded labels)
        labels_matrix = np.zeros((len(ab), 65))
        labels_matrix[np.arange(len(ab)), ab] = 1
        #print(labels_matrix.shape)labels_matrix
        labelt_matrix = np.zeros((len(cd), 65))
        labelt_matrix[np.arange(len(cd)), cd] = 1
        labels_matrix=torch.tensor(labels_matrix)
        labelt_matrix=torch.tensor(labelt_matrix)

        #print(labels_matrix.shape) 
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss_im = im(logits_t.view(-1, 65))
        new_mmd=tlmmd(x_s, x_t, labels_matrix, labelt_matrix, kernel_mul=2.0, kernel_num=5, fix_sigma=None)
        loss = cls_loss + transfer_loss * args.trade_off+0.05*loss_im+0.25*mmd+0.2*mcc_loss+0.25*new_mmd

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        MMD_Loss.update(new_mmd.item(),x_s.size(0))
        mccloss.update((0.1*mcc_loss).item(),x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--mcc', default=0.1, type=float,)
    parser.add_argument('--mmd', default=0.25, type=float)
    parser.add_argument('--new_mmd', default=0.25, type=float)
    parser.add_argument('--ent', default=0.05, type=float)
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
