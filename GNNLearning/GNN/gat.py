import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from dgl.nn import GATConv

class GAT( nn.Module ):
    def __init__(self,
                 g, #DGL的图对象
                 n_layers, #层数
                 in_feats, #输入特征维度
                 n_hidden, #隐层特征维度
                 n_classes, #类别数
                 heads, #多头注意力的数量
                 activation, #激活函数
                 in_drop, #输入特征的Dropout比例
                 at_drop, #注意力特征的Dropout比例
                 negative_slope, #注意力计算中Leaky ReLU的a值
                 ):
        super( GAT, self ).__init__( )
        self.g = g
        self.num_layers = n_layers
        self.activation = activation

        self.gat_layers = nn.ModuleList()

        self.gat_layers.append( GATConv(
            in_feats, n_hidden, heads[0],
            in_drop, at_drop, negative_slope, activation=self.activation ) )

        for l in range(1, n_layers):
            self.gat_layers.append( GATConv(
                n_hidden * heads[l-1], n_hidden, heads[l],
                in_drop, at_drop, negative_slope, activation=self.activation))

        self.gat_layers.append( GATConv(
            n_hidden * heads[-2], n_classes, heads[-1],
            in_drop, at_drop, negative_slope, activation=None) )

    def forward( self, inputs ):
        h = inputs
        for l in range( self.num_layers ):
            h = self.gat_layers[l]( self.g, h ).flatten( 1 )
        logits = self.gat_layers[-1]( self.g, h ).mean( 1 )
        return logits


def evaluate( model, features, labels, mask ):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def evaluate_with_predictions(model, features, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        return indices.cpu().numpy()  # 返回预测结果的 numpy 数组

def train( n_epochs = 100,
           lr = 5e-3,
           weight_decay = 5e-4,
           n_hidden = 16,
           n_layers = 1,
           activation = F.elu,
           n_heads = 3, #中间层多头注意力的数量
           n_out_heads = 1, #输出层多头注意力的数量
           feat_drop = 0.6,
           attn_drop = 0.6,
           negative_slope = 0.2):
    data = CoraGraphDataset() #加载 Cora 数据集，用于节点分类任务
    g=data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1] #特征矩阵的列数，即节点特征的维度
    n_classes = data.num_classes
    heads = ([n_heads] * n_layers) + [n_out_heads] #对于中间层的head=n_heads=3，输出层的head=n_out_heads=1
    #实例化模型
    model = GAT( g,
                 n_layers,
                 in_feats,
                 n_hidden,
                 n_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope
                 )
    #loss函数采用交叉熵损失函数 
    loss_fcn = torch.nn.CrossEntropyLoss()
    #使用了 Adam 优化器
    optimizer = torch.optim.Adam( model.parameters(),#返回模型中所有需要更新的参数，参数通常是模型中的权重和偏置项
                                 lr = lr,#学习率
                                 weight_decay = weight_decay) #权重衰减参数，用于控制模型的正则化
    for epoch in range( n_epochs ):
        model.train()
        logits = model( features )
        loss = loss_fcn( logits[ train_mask ], labels[ train_mask ] )
        optimizer.zero_grad() #将之前计算得到的梯度清零，以避免梯度的累积。
        loss.backward() #根据当前的损失值，计算损失函数关于模型参数的梯度。使用这些梯度来更新模型参数
        optimizer.step()
        acc = evaluate( model, features, labels, val_mask )
        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, loss.item(), acc ))
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))
   # test_predictions = evaluate_with_predictions(model, features, test_mask)
   # print("Test predictions:", test_predictions)
if __name__ == '__main__':
    train()