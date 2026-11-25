import copy
import torch
import torch.nn.functional as F
import networkx as nx
from models.line import LINE

import torch.nn as nn
import torch_clustering


class AdaptiveModule(nn.Module):
    def __init__(self):
        super(AdaptiveModule, self).__init__()
        self.register_buffer("a", torch.tensor(0.5))  # 初始化为0.5

    def update(self, loss_inter, loss_intra):
        # 根据loss_inter和loss_intra的大小动态调整a
        if loss_inter > loss_intra + 1:
            self.a = torch.clamp(self.a - 0.1, 0, 1)
        else:
            self.a = torch.clamp(self.a + 0.1, 0, 1)

    def forward(self):
        return self.a



class CAFE(nn.Module):
    def __init__(self, n_views, n_samples, layer_dims, temperature, n_classes, drop_rate=0.5):
        super(CAFE, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])

        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.n_samples = n_samples
        self.psedo_labels = torch.zeros((self.n_samples,)).long().cuda()
        self.temperature_l = temperature
        self.temperature_f = temperature
        self.similarity = nn.CosineSimilarity(dim=2)

    @torch.no_grad()
    def compute_feature(self, data):
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        return z, p

    def forward(self, data, momentum, warm_up):
        self._update_target_branch(momentum)

        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]

        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            mp = torch.eye(z[0].shape[0]).cuda()
            mp = [mp, mp]
        else:
            mp = [self.kernel_affinity(z_t[i]) for i in range(self.n_views)]

        adaptive_module = AdaptiveModule()

        a = adaptive_module()
        l_inter = a*(self.cl(p[0], z_t[1], mp[1]) + self.cl(p[1], z_t[0], mp[0]))
        l_intra = (1-a)*(self.cl(z[0], z_t[0], mp[0]) + self.cl(z[1], z_t[1], mp[1]))

        loss = l_inter + l_intra

        return loss

    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):  # 计算样本之间的亲和力矩阵，并通过高阶随机游走的方法进一步捕捉长距离的关系
        z = L2norm(z)   # 对输入的特征矩阵 z 进行 L2 归一化处理。L2norm 函数是对每个向量进行标准化，使得每个向量的 L2 范数为 1
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)    # 这一行代码计算了特征矩阵 z 中样本之间的距离矩阵
        G = torch.exp(-G / temperature)     # 将负距离输入到指数函数中，得到一个高斯核（Gaussian kernel）的相似度矩阵，类似于热核（heat kernel
        G = G / G.sum(dim=1, keepdim=True) # 对相似度矩阵 G 的每一行进行归一化处理 确保所有相似度值的总和为1，这样可以将矩阵 G 视为一个条件概率矩阵，表示某个样本与其他样本之间的概率分布

        G = G.cpu().numpy()
        G = nx.from_numpy_array(G)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)    # 这一行代码计算了特征矩阵 z 中样本之间的距离矩阵
        G = torch.exp(-G / temperature)     # 将负距离输入到指数函数中，得到一个高斯核（Gaussian kernel）的相似度矩阵
        G = G / G.sum(dim=1, keepdim=True) # 对相似度矩阵 G 的每一行进行归一化处理 确保所有相似度值的总和为1，这样可以将矩阵 G 视为一个条件概率矩阵

        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)  # 将单位矩阵（自连接）与亲和力矩阵 G 进行加权融合, 通过自连接和邻居信息的平衡，确保在高阶游走过程中，不会完全丧失当前样本自身的信息。
        return G

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z = [L2norm(z[i]) for i in range(self.n_views)]

        return z

    @torch.no_grad()
    def get_weights(self):
        # 使用注意力机制计算视图权重
        with torch.no_grad():
            # 假设self.weights是一个可学习的参数，初始化为均匀分布
            weights = torch.softmax(self.weights, dim=0)
            return weights

    @torch.no_grad()
    def fusion(self, zs):
        # 基于注意力机制的视图融合
        zs = [z.cuda() for z in zs]  # 将 zs 中的所有张量移动到 GPU
        weights = self.get_weights().cuda()  # 确保权重在 GPU 上

        # 计算每个视图的注意力权重
        attn_weights = []
        for i in range(self.n_views):
            attn = torch.sum(zs[i] * weights[i], dim=1)
            attn_weights.append(attn)

        # 归一化注意力权重
        attn_weights = torch.stack(attn_weights)
        attn_weights = torch.softmax(attn_weights, dim=0)

        # 加权融合
        weighted_zs = []
        for i in range(self.n_views):
            weighted_z = zs[i] * attn_weights[i].unsqueeze(1)
            weighted_zs.append(weighted_z)

        common_z = torch.sum(torch.stack(weighted_zs), dim=0)
        return common_z



    
    @torch.no_grad()
    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.n_classes, n_samples).to(x)
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers
    @torch.no_grad()
    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.n_classes, n_samples).to(x)
            weight[psedo_labels, torch.arange(n_samples)] = 1
    
        # 动态调整权重
        weight = self._adjust_weights(weight, psedo_labels)
    
        # 结合图结构信息
        affinity_matrix = self._compute_affinity_matrix(x)
    
        weight = torch.mm(weight, affinity_matrix)  # 矩阵乘法

        # 计算聚类中心
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)

        return centers
    
    @torch.no_grad()
    def _adjust_weights(self, weight, psedo_labels):
        # 根据类别分布动态调整权重
        class_counts = torch.bincount(psedo_labels, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)  # 避免除以零
        class_weights = class_weights / class_weights.sum()  # 归一化
        weight = weight * class_weights.view(-1, 1)
        return weight
    
    @torch.no_grad()
    def _compute_affinity_matrix(self, x):
        # 计算样本之间的亲和力矩阵
        affinity_matrix = torch.matmul(x, x.T)
        affinity_matrix = F.normalize(affinity_matrix, p=1, dim=1)
        return affinity_matrix





    @torch.no_grad()
    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.n_classes,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))

        return psedo_labels

    @torch.no_grad()
    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
        loss_single = self.compute_single_view_cluster_loss( q_centers, k_centers, psedo_labels)
        loss_fuse = self.compute_fused_view_cluster_loss( q_centers, k_centers, psedo_labels)
        loss = loss_single + loss_fuse
        return loss

    @torch.no_grad()
    def compute_single_view_cluster_loss(self, q_centers, k_centers, psedo_labels):
        # 计算单视图聚类中心的损失
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.n_classes), torch.arange(self.n_classes)] = d_k

        zero_classes = torch.arange(self.n_classes).cuda()[
            torch.sum(F.one_hot(torch.unique(psedo_labels), self.n_classes), dim=0) == 0]
        mask = torch.zeros((self.n_classes, self.n_classes), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        pos = torch.sigmoid(pos)

        loss = - pos
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.n_classes - len(zero_classes))

        return loss

    @torch.no_grad()
    def compute_fused_view_cluster_loss(self, q_centers, k_centers, psedo_labels):
        # 计算融合视图聚类中心的损失
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.n_classes), torch.arange(self.n_classes)] = d_k

        zero_classes = torch.arange(self.n_classes).cuda()[
            torch.sum(F.one_hot(torch.unique(psedo_labels), self.n_classes), dim=0) == 0]
        mask = torch.zeros((self.n_classes, self.n_classes), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.n_classes, self.n_classes))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.n_classes - 1)

        # 为 pos 和 neg 添加激活函数
        pos = torch.sigmoid(pos)  # 例如，使用 Sigmoid 激活函数
        neg = torch.sigmoid(neg)  # 例如，使用 Sigmoid 激活函数

        loss = torch.logsumexp(torch.cat([pos.reshape(self.n_classes, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.n_classes - len(zero_classes))

        return loss

    @torch.no_grad()
    def feature_loss(self, zi, z, w, y_pse):
        cross_view_distance = self.similarity(zi.unsqueeze(1), z.unsqueeze(0)) / self.temperature_f
        N = z.size(0)
        w = w + torch.eye(N, dtype=int).to(w.device)
        positive_loss = (w & y_pse) * cross_view_distance
        inter_view_distance = self.similarity(zi.unsqueeze(1), zi.unsqueeze(0)) / self.temperature_f
        positive_loss = -torch.sum(positive_loss)
        negated_w = w ^ True
        negated_y = y_pse ^ True
        SMALL_NUM = torch.log(torch.tensor(1e-45)).to(zi.device)
        negtive_cross = (negated_w & negated_y) * cross_view_distance
        negtive_cross[negtive_cross == 0.] = SMALL_NUM
        negtive_inter = (negated_w & negated_y) * inter_view_distance
        negtive_inter[negtive_inter == 0.] = SMALL_NUM
        negtive_similarity = torch.cat((negtive_inter, negtive_cross), dim=1) / self.temperature_f
        negtive_loss = torch.logsumexp(negtive_similarity, dim=1, keepdim=False)
        negtive_loss = torch.sum(negtive_loss)
        return (positive_loss + negtive_loss) / N


L2norm = nn.functional.normalize

class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class AttentionNetwork(nn.Module):
    def __init__(self, dim_layer, num_heads=1, norm_layer=None, act_layer=None, drop_out=0.0):
        super(AttentionNetwork, self).__init__()
        self.num_layers = len(dim_layer) - 1
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.LayerNorm

        # Project the input to the required embedding dimension for MultiheadAttention
        self.input_projection = nn.Linear(dim_layer[0], dim_layer[1])  # Project from 20 to 1024 if necessary
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        # Define attention, norm, and activation layers
        for i in range(1, self.num_layers - 1):
            self.attention_layers.append(nn.MultiheadAttention(embed_dim=dim_layer[i], num_heads=num_heads, dropout=drop_out))
            self.norm_layers.append(norm_layer(dim_layer[i]))
            self.activation_layers.append(act_layer())

        # Final output layer to project to last dimension
        self.output_layer = nn.Linear(dim_layer[-2], dim_layer[-1])

    def forward(self, x):
        # Project the input to the required embedding dimension
        x = self.input_projection(x)  # 将输入映射到第一个注意力层的期望维度
        x = x.unsqueeze(0)  # Add sequence dimension for attention layer

        # Forward pass through attention layers
        for i in range(len(self.attention_layers)):
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = x + attn_output  # Residual connection
            x = self.norm_layers[i](x)  # Layer normalization
            x = self.activation_layers[i](x)  # Activation

        # Final projection
        x = x.squeeze(0)  # Remove sequence dimension
        return self.output_layer(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_ratio=4.0, act_layer=None):
        super(ResNetBlock, self).__init__()
        act_layer = act_layer or nn.ReLU
        dim_hidden = int(dim_in * hidden_ratio)

        # 定义残差块的两个线性层
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.activation = act_layer()
        self.fc2 = nn.Linear(dim_hidden, dim_out)

        # 如果输入和输出维度不一致，需要使用1x1卷积来调整维度
        if dim_in != dim_out:
            self.residual_connection = nn.Linear(dim_in, dim_out)
        else:
            self.residual_connection = None

    def forward(self, x):
        # 主分支
        residual = x  # 保存输入以便之后的残差连接
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        # 残差连接
        if self.residual_connection:
            residual = self.residual_connection(residual)

        x += residual  # 将主分支输出与残差相加
        x = self.activation(x)  # 再通过激活函数
        return x


class ResNet(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, num_blocks=3, act_layer=None):
        super(ResNet, self).__init__()
        dim_out = dim_out or dim_in
        act_layer = act_layer or nn.ReLU

        # 构建多个残差块
        layers = [ResNetBlock(dim_in, dim_out, hidden_ratio, act_layer)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(dim_out, dim_out, hidden_ratio, act_layer))

        self.resnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss
