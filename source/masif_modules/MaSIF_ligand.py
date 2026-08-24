import tensorflow as tf
import numpy as np


class MaSIF_ligand:

    """
    The neural network model.
    """

    def count_number_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print("Total number parameters: %d" % total_parameters)

    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm

    def build_sparse_matrix_softmax(self, idx_non_zero_values, X, dense_shape_A):
        A = tf.SparseTensorValue(idx_non_zero_values, tf.squeeze(X), dense_shape_A)
        A = tf.sparse_reorder(A)  # n_edges x n_edges
        A = tf.sparse_softmax(A)

        return A

    def compute_initial_coordinates(self):
        ##########################################
        #####  返回 80个grid的坐标：(rho, theta)
        ##########################################
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1) ##分成rho_bin=5段
        grid_rho = grid_rho[1:] ## 5个元素：2.4, 4.8, 7.2, 9.6, 12.0
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1] ## 16个元素：0, pi/8, pi/4, ... 15pi/8

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        ## grid_rho_(每行都相同) & grid_theta_(每列都相同): 16行5列
        grid_rho_ = (
            grid_rho_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_theta_ = (
            grid_theta_.T
        )  # the traspose here is needed to have the same behaviour as Matlab code
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection
        print(coords.shape)   ## 每一行都代表一个grid: (rho(5种取值), theta(16种取值))；共16*5=80行 --> 代表80个grid
        return coords

    def inference(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        n_samples = tf.shape(rho_coords)[0] ## batch_size
        n_vertices = tf.shape(rho_coords)[1] ## n_vertices

        all_conv_feat = []
        for k in range(self.n_rotations):
            #### --------------------------------- ####
            ##    Learned soft grid 层：5rho * 16theta
            ##     对应于一个patch的一个feature channel
            #### --------------------------------- ####
            rho_coords_ = tf.reshape(rho_coords, [-1, 1])  ## 将张量变为一维列向量:shape:(n, 1) # batch_size*n_vertices
            thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices

            thetas_coords_ += k * 2 * np.pi / self.n_rotations 
            thetas_coords_ = tf.mod(thetas_coords_, 2 * np.pi) ## 返回除法余数（即限制在[0, 2pi)）    ### 只改变了theta值（16种）：旋转不变性    
            ########## 
            ##   注意：从下面几行代码看到：权重参数的计算中，高斯核“中心”的位置选在：
            ##        mu_rho: 80维行向量；5种取值(2.4, 4.8, 7.2, 9.6, 12.0)
            ##        mu_theta: 80维行向量；16种取值(0, pi/8, pi/4, ... 15pi/8)
            ##########
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - mu_rho) / (tf.square(sigma_rho) + eps)
            ) ## 权重参数rho部分      ## 注意：这里不同维张量的运算涉及tf中“广播”的概念；最后的到的shape: （bs*nv, 80）
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - mu_theta) / (tf.square(sigma_theta) + eps)
            ) ## 权重参数theta部分    ## 张量广播，同上

            gauss_activations = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss ## rho部分和theta部分相乘得到权重参数矩阵vij  ## ！！注意：n_gauss即rho_bin * theta_bin=80
            gauss_activations = tf.reshape(
                gauss_activations, [n_samples, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss ## 不改变元素之间的顺序
            gauss_activations = tf.multiply(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(gauss_activations, 1, keep_dims=True) + eps
                )  # batch_size, n_vertices, n_gauss ## 每个vertice在不同高斯核(80个bins)上的权重总和为1

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 3
            )  # batch_size, n_vertices, n_feat, 1

            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,   ## 特征与权重参数相乘：vij(x, x')f(x')            gauss_desc = tf.reduce_sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos]
            )  # batch_size, 80   ## 对于每一个"soft patch"Dij(x)（即由rho和theta定位的每一个bin），对patch内的每个vertice求和

            #### --------------------------------- ####
            ##     Covolutional 层 ：80filters （？？？？感觉没有卷积）
            ##  W_cov:权重参数；(80, 80) --> 一列代表一种filter，共80列/个
            ##  b_conv: 偏置参数；(80,) --> 80个值分别对应着80个filter，计算中被广播为(32, 80)，即平等地加给每个filter操作后的32个patch
            #### --------------------------------- ####
            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80 ##其中，b_conv的
            all_conv_feat.append(conv_feat)
        #### --------------------------------- ####
        ## Angular Max-pooling 层：16 rotations取最大
        #### --------------------------------- ####
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(all_conv_feat, 0) ## 每个sample（patch channel）的每个bin(一共16*5=80个bins)在16个值里挑最大值
        #### ----------------- ####
        ##        Relu 层
        #### ----------------- ####
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    def __init__(
        self,
        session,
        max_rho,
        n_ligands,
        n_thetas=16,
        n_rhos=5,
        n_gamma=1.0,
        learning_rate=1e-4,
        n_rotations=16,
        idx_gpu="/gpu:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0],
        costfun="dprime",
    ):

        # order of the spectral filters
        self.max_rho = max_rho     ## params["max_distance"] = 12.0 （patch半径）
        self.n_thetas = n_thetas   ## 16 （角度bin数）
        self.n_rhos = n_rhos       ## 5   (径向距离bin数)
        self.n_ligands = n_ligands ## params["n_classes"] = 7 （口袋种类数）
        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations ## 16
        self.n_feat = int(sum(feat_mask)) ## 使用的特征数量：5

        # with tf.Graph().as_default() as g:
        with tf.get_default_graph().as_default() as g:
            self.graph = g
            tf.set_random_seed(0)
            for pr in range(1):

                initial_coords = self.compute_initial_coordinates()
                # self.rotation_angles = tf.Variable(np.arange(0, 2*np.pi, 2*np.pi/self.n_rotations).astype('float32'))
                mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype(
                    "float32"
                )
                mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype(
                    "float32"
                )
                self.mu_rho = []
                self.mu_theta = []
                self.sigma_rho = []
                self.sigma_theta = []
                for i in range(self.n_feat):
                    ## 以下几个变量的形状：(1, 80)
                    ## tf.Variable：创建变量并命名
                    self.mu_rho.append(
                        tf.Variable(mu_rho_initial, name="mu_rho_{}".format(i))
                    )  # 1, n_gauss  #### 80维行向量；5种取值
                    self.mu_theta.append(
                        tf.Variable(mu_theta_initial, name="mu_theta_{}".format(i))
                    )  # 1, n_gauss  #### 80维行向量；16种取值
                    self.sigma_rho.append(
                        tf.Variable(
                            np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                            name="sigma_rho_{}".format(i),
                        )
                    )  # 1, n_gauss  #### 1.5
                    self.sigma_theta.append(
                        tf.Variable(
                            (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                            name="sigma_theta_{}".format(i),
                        )
                    )  # 1, n_gauss  #### 1.0

                ## tf.placeholder：占位符
                ### 它指定了被喂给graph的data的【Shape & Type】
                self.keep_prob = tf.placeholder(tf.float32)
                self.rho_coords = tf.placeholder(
                    tf.float32
                )  # batch_size, n_vertices, 1
                self.theta_coords = tf.placeholder(
                    tf.float32
                )  # batch_size, n_vertices, 1
                self.input_feat = tf.placeholder(
                    tf.float32, shape=[None, None, self.n_feat]
                )  # batch_size, n_vertices, n_feat
                self.mask = tf.placeholder(tf.float32)  # batch_size, n_vertices, 1
                self.labels = tf.placeholder(tf.float32)
                self.global_desc_1 = []
                b_conv = []
                for i in range(self.n_feat):
                    b_conv.append(
                        tf.Variable(
                            tf.zeros([self.n_thetas * self.n_rhos]),
                            name="b_conv_{}".format(i),
                        )
                    )
                for i in range(self.n_feat):
                    # self.flipped_input_feat = tf.concat([tf.expand_dims(-self.input_feat[:,:,0], 2), -self.input_feat[:,:,1:]], 2)
                    my_input_feat = tf.expand_dims(self.input_feat[:, :, i], 2)

                    # self.flipped_theta_coords = 0*self.theta_coords;

                    W_conv = tf.get_variable(
                        "W_conv_{}".format(i),
                        shape=[
                            self.n_thetas * self.n_rhos,
                            self.n_thetas * self.n_rhos,
                        ],
                        initializer=tf.contrib.layers.xavier_initializer(),
                    ) ## 初始化程序"Xavier"：返回初始化权重矩阵

                    ######################################
                    #### 形状说明：
                    ## my_input_feat: batch_size, n_vertices, 1, 1
                    ## coords: batch_size, n_vertices, 1
                    ## W_conv (i.e. W_conv[i]): 80, 80,
                    ## b_conv[i]: 80,
                    ## [i]: 1, 80
                    ######################################
                    self.global_desc_1.append(
                        self.inference(
                            my_input_feat,
                            self.rho_coords,
                            self.theta_coords,
                            self.mask,
                            W_conv,
                            b_conv[i],
                            self.mu_rho[i],
                            self.sigma_rho[i],
                            self.mu_theta[i],
                            self.sigma_theta[i],
                        )
                    )  # batch_size, n_gauss*1
                    ### append后得到 (n_feat, batch_size, 80)

                #### --------------------------------- ####
                ##  Fully-connected 层：合并feat通道，
                ##        为每一patch(即一行)生成80-D指纹
                ##    (bs, nf*80) --> (bs, 80)
                #### --------------------------------- ####
                # global_desc_1 and global_desc_2 are n_feat, batch_size, n_gauss*1
                # They should be batch_size, n_feat*n_gauss
                self.global_desc_1 = tf.stack(self.global_desc_1, axis=1) ## (bs, nf, 80) 
                self.global_desc_1 = tf.reshape(
                    self.global_desc_1, [-1, self.n_thetas * self.n_rhos * self.n_feat]
                ) ## (bs, nf*ng) ## 注意：reshape不改变内部元素数量/相对位置；“-1”可以理解为“占位”，此维度的大小由元素数目和其他维度大小填充推定

                # refine global desc with MLP
                self.global_desc_1 = tf.contrib.layers.fully_connected(
                    self.global_desc_1,
                    self.n_thetas * self.n_rhos,
                    activation_fn=tf.nn.relu,
                ) ## Output units: rho*theta = n_gauss = 80 ## 注意，这里输出的形状为(batch_size, 80)
                ## 参数说明： tf.contrib.layers.fully_connected（F, num_outputs,activation_fn）
                ##      F：tensor:[batch_size, image_pixels]; num_outputs: number of outputs:[batch_size, num_outputs]


                #### --------------------------------- ####
                ##  Covariance Matrix 处理：
                ##     把每一个fingerprint位(共80位)当做一种“特征”；
                ##      以32块patch为32种sample，计算协方差矩阵：公式：A.T * A（在草稿本上有推导）
                ##       其中A的格式为(Samples, features)-->
                ##         即每一行代表同一sample的不同特征；每一列代表一种feature的不同采样
                ##    (80, bs)*(bs, 80) --> (80, 80)
                #### --------------------------------- ####
                self.global_desc_1 = tf.matmul(
                    tf.transpose(self.global_desc_1), self.global_desc_1
                ) / tf.cast(tf.shape(self.global_desc_1)[0], tf.float32)
                '''
                *关于上述协方差矩阵的理解：
                batch_size： 是数据样本的数量（ 32 个样本）
                n_features： 是特征的数量（ 80 个特征，即 80个维度的指纹）
                ==> 自相关矩阵的每个元素 (i, j) 反映了特征 i 与特征 j 在整个32个数据中的相互关系
                ==> 即：捕捉不同特征之间在所有数据样本上的整体相互关系
                '''


                #### --------------------------------- ####
                ##  FC layers：
                ##        最终实现分类器功能
                ##    (bs, 80) --> (7)
                #### --------------------------------- ####
                self.global_desc_1 = tf.reshape(self.global_desc_1, [1, -1]) ## flatten操作
                self.global_desc_1 = tf.nn.dropout(self.global_desc_1, self.keep_prob) ## keep_prob: 每一个神经元被保存下的频率（随机关掉一些神经元防止过拟合）
                self.global_desc_1 = tf.contrib.layers.fully_connected(
                    self.global_desc_1, 64, activation_fn=tf.nn.relu
                ) ## FC64层 
                self.logits = tf.contrib.layers.fully_connected(
                    self.global_desc_1, self.n_ligands, activation_fn=tf.identity
                ) ######### FC7层


                # compute data loss
                self.labels = tf.expand_dims(self.labels, axis=0) ## (1, labels)
                self.logits = tf.expand_dims(self.logits, axis=0) ## (1, 7)
                self.logits_softmax = tf.nn.softmax(self.logits) ## softmax操作：归一化多维向量（默认在最后一维操作）
                self.computed_loss = tf.reduce_mean(
                    -tf.reduce_sum(self.labels * tf.log(self.logits_softmax), [1])
                )

                self.data_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=self.logits
                )
                # definition of the solver
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate
                ).minimize(self.data_loss)

                self.var_grad = tf.gradients(self.data_loss, tf.trainable_variables())
                for k in range(len(self.var_grad)):
                    if self.var_grad[k] is None:
                        print(tf.trainable_variables()[k])
                self.norm_grad = self.frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0)
                )

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = session
                self.saver = tf.train.Saver()

                # Run the Op to initialize the variables.
                init = tf.global_variables_initializer()
                self.session.run(init)
                self.count_number_parameters()

