import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.data import CoraGraphDataset

class GCN( nn.Module ):
    def __init__(self,
                 g, #DGL的图对象
                 in_feats, #输入特征的维度
                 n_hidden, #隐层的特征维度
                 n_classes, #类别数
                 n_layers, #网络层数
                 activation, #激活函数
                 dropout #dropout系数
                 ):
        super( GCN, self ).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append( GraphConv( in_feats, n_hidden, activation = activation ))
        # 隐层
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation = activation ))
        # 输出层
        self.layers.append( GraphConv( n_hidden, n_classes ) )
        self.dropout = nn.Dropout(p = dropout)

    def forward( self, features ):
        h = features
        for i, layer in enumerate( self.layers ):
            if i != 0:
                h = self.dropout( h ) #对特征进行Dropout处理。
            h = layer( self.g, h )  #将处理后的特征h输入到当前层layer中，得到输出特征\hat(h)
        return h

def evaluate(model, features, labels, mask):
    model.eval() #将模型设置为评估模式，这会关闭模型中的dropout层和其他正则化层的功能，以便在评估阶段保持一致的行为。
    with torch.no_grad(): #使用torch.no_grad()上下文管理器，确保在评估阶段不会计算梯度，从而节省内存和计算资源。
        logits = model(features) #通过模型前向传播计算得到预测 logits（未经 softmax 或其他概率变换）。
        logits = logits[mask] #根据掩码mask选择出需要评估的样本的logits
        labels = labels[mask] #根据掩码mask选择出需要评估的样本的标签
        _, indices = torch.max(logits, dim=1)  #在logits的每一行中，找到最大值的索引，即预测的类别。 “_”：临时变量，存储最大值；“indices”：存储最大值的索引
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(n_epochs=100, lr=1e-2, weight_decay=5e-4, n_hidden=16, n_layers=1, activation=F.relu , dropout=0.5):
    data = CoraGraphDataset()
    g=data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes

    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                activation,
                dropout)

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( model.parameters(),
                                 lr = lr,
                                 weight_decay = weight_decay)
    for epoch in range( n_epochs ):
        model.train()
        logits = model( features )
        loss = loss_fcn( logits[ train_mask ], labels[ train_mask ] )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, loss.item(), acc ))
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

if __name__ == '__main__':
    train()