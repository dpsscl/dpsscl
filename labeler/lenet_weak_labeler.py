import numpy as np
import torch
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


class LeNetWeakLabeler(torch.nn.Module):

    # Need to input input dims and output dims
    # Can input dict_training_param: {learning rate, num batches, num epoches, fc1 size, fc2 size}
    def __init__(self, in_dim_h=28, in_dim_w=28, in_dim_c=1, out_dim=2, dict_training_param=None):
        super(LeNetWeakLabeler, self).__init__()

        self.in_dim_h = in_dim_h
        self.in_dim_w = in_dim_w
        self.in_dim_c = in_dim_c
        self.out_dim = out_dim

        if dict_training_param is not None:
            self.learning_rate = dict_training_param["learning_rate"]
            self.num_batches = dict_training_param["num_batches"]
            self.num_epochs = dict_training_param["num_epochs"]
            if "dropout" in dict_training_param.keys():
                self.dropout = dict_training_param["dropout"]
            else:
                self.dropout = False
        else:  # If training params not specified, use random params as follows
            self.learning_rate = 1e-3
            self.num_batches = 5
            self.num_epochs = 80
            self.dropout = False


        # architecture parameters
        self.out_c_conv1 = 6
        self.k_conv1 = 5
        self.p_conv1 = 2
        self.k_pool1 = 2
        self.s_pool1 = 2
        self.out_c_conv2 = 16
        self.k_conv2 = 5
        self.k_pool2 = 2
        self.s_pool2 = 2
        self.d_fc1 = self.compute_fc_size()
        self.d_fc2 = 45
        self.d_fc3 = 21

        self.conv1 = torch.nn.Conv2d(in_channels=self.in_dim_c, out_channels=self.out_c_conv1, kernel_size=self.k_conv1, padding=self.p_conv1)
        self.dropout1 = torch.nn.Dropout2d(p=0.2)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=self.k_pool1, stride=self.s_pool1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_c_conv1, out_channels=self.out_c_conv2, kernel_size=self.k_conv2)
        self.dropout2 = torch.nn.Dropout2d(p=0.3)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=self.k_pool2, stride=self.s_pool2)
        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Sequential(torch.nn.Linear(self.d_fc1, self.d_fc2), torch.nn.Sigmoid())
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(self.d_fc2, self.d_fc3), torch.nn.Sigmoid())
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(self.d_fc3, self.out_dim), torch.nn.Softmax())

        # Indicate task index if this weak labeler is transferred from earlier task
        # this is task-order-related index, not (absolute) task index
        self.earlier_task_id = -1
        self.converged_epoch_ratio = -1
        self.training_acc_history = []

    def forward(self, x):
        out = self.conv1(x)
        out = torch.sigmoid(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.avgpool1(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.avgpool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        if self.dropout:
            out = self.dropout3(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    # update training params
    def update_training_params(self, dict_training_param: dict):
        self.learning_rate = dict_training_param["learning_rate"]
        self.num_batches = dict_training_param["num_batches"]
        self.num_epochs = dict_training_param["num_epochs"]
        if "dropout" in dict_training_param.keys():
            self.dropout = dict_training_param["dropout"]
        else:
            self.dropout = False
        return

    # compute shape of tensor output from previous params to determine first fc size
    def compute_fc_size(self):
        h_conv1 = compute_next_shape(prev_shape=self.in_dim_h, padding=self.p_conv1, dilation=1, kernel_size=self.k_conv1, stride=1)
        h_pool1 = compute_next_shape(prev_shape=h_conv1, padding=0, dilation=1, kernel_size=self.k_pool1, stride=self.s_pool1)
        h_conv2 = compute_next_shape(prev_shape=h_pool1, padding=0, dilation=1, kernel_size=self.k_conv2, stride=1)
        h_pool2 = compute_next_shape(prev_shape=h_conv2, padding=0, dilation=1, kernel_size=self.k_pool2, stride=self.s_pool2)
        w_conv1 = compute_next_shape(prev_shape=self.in_dim_w, padding=self.p_conv1, dilation=1, kernel_size=self.k_conv1, stride=1)
        w_pool1 = compute_next_shape(prev_shape=w_conv1, padding=0, dilation=1, kernel_size=self.k_pool1, stride=self.s_pool1)
        w_conv2 = compute_next_shape(prev_shape=w_pool1, padding=0, dilation=1, kernel_size=self.k_conv2, stride=1)
        w_pool2 = compute_next_shape(prev_shape=w_conv2, padding=0, dilation=1, kernel_size=self.k_pool2, stride=self.s_pool2)
        fc_size = self.out_c_conv2 * h_pool2 * w_pool2
        return fc_size

    # retrieve variable tensors
    def get_parameters(self):
        conv1_w = self.conv1.weight
        conv1_b = self.conv1.bias
        conv2_w = self.conv2.weight
        conv2_b = self.conv2.bias
        fc1_w = self.fc1[0].weight
        fc1_b = self.fc1[0].bias
        fc2_w = self.fc2[0].weight
        fc2_b = self.fc2[0].bias
        fc3_w = self.fc3[0].weight
        fc3_b = self.fc3[0].bias
        return {"conv1_w": conv1_w, "conv1_b": conv1_b, "conv2_w": conv2_w, "conv2_b": conv2_b,
                "fc1_w": fc1_w, "fc1_b": fc1_b, "fc2_w": fc2_w, "fc2_b": fc2_b, "fc3_w": fc3_w, "fc3_b": fc3_b}

    # retrieve size of the model in bytes
    def get_size(self):
        dict_params = self.get_parameters()
        size = 0
        for key in dict_params.keys():
            size += torch.numel(dict_params[key])
        return size * 4

    # input float tensor output prob of class 1 as numpy array
    def marginal(self, X):
        if X.shape[0] >= 20:
            num_split = 20
        else:
            num_split = 1
        X_split = np.array_split(X, num_split, axis=0) # use array split, so chunks are allowed to be not equal
        out_all = []
        for i in range(num_split):
            X_tensor = torch.from_numpy(X_split[i]).to(DEVICE)
            out = self(X_tensor).cpu().detach().numpy()
            out_all.append(out)
        out_all = np.concatenate(out_all)
        return out_all[:, 1]

    # input float tensor output entire prob matrix as numpy array
    def prob_matrix(self, X):
        if X.shape[0] >= 20:
            num_split = 20
        else:
            num_split = 1
        X_split = np.array_split(X, num_split, axis=0)
        out_all = []
        for i in range(num_split):
            X_tensor = torch.from_numpy(X_split[i]).to(DEVICE)
            out = self(X_tensor).cpu().detach().numpy()
            out_all.append(out)
        out_all = np.concatenate(out_all)
        return out_all

    # train the model
    def train_cnn(self, X, y, verbose=False):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        loss_list = []
        acc_list = []
        total_step = self.num_batches

        batches = batch_loader(X, y, num_batches=self.num_batches)

        for epoch in range(self.num_epochs):
            total2, correct2 = 0, 0
            for i, (images, labels) in enumerate(batches):
                torch.cuda.empty_cache() # deallocate gpu
                images_tensor = torch.from_numpy(images).to(DEVICE)
                labels_tensor = torch.from_numpy(labels).long().to(DEVICE)
                # Run the forward pass
                outputs = self(images_tensor)
                loss = criterion(outputs, labels_tensor)
                loss_list.append(loss.item())
                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total2 += labels.shape[0]
                _, predicted = torch.max(outputs.data, 1)
                correct2 += (predicted == labels_tensor).sum().item()

                if verbose:
                    # Track the accuracy
                    total = labels.shape[0]
                    correct = (predicted == labels_tensor).sum().item()
                    acc_list.append(correct / total)
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
            self.training_acc_history.append(correct2/(total2+1e-10))
        for i, (acc_tmp) in enumerate(self.training_acc_history):
            if acc_tmp >= 0.95*self.training_acc_history[-1]:
                self.converged_epoch_ratio = (i+1)/len(self.training_acc_history)
                break
        return self

    def update_info_transferred_task(self, task_id):
        self.earlier_task_id = task_id


# compare if two models have the same parameters
def cnn_equal(model1, model2):
    params1 = model1.get_parameters()
    params2 = model2.get_parameters()
    for key in params1.keys():
        if key not in params2.keys():
            return False
        if not torch.equal(params1[key], params2[key]):
            return False
    return True


# batch loader of data, input X, y are numpy arrays
def batch_loader(X, y, num_batches):
    shuffler = np.random.permutation(X.shape[0])
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    X_split = np.array_split(X_shuffled, num_batches)
    y_split = np.array_split(y_shuffled, num_batches)

    data = []
    for i in range(num_batches):
        data.append((X_split[i], y_split[i]))

    return data


# compute shape of next layer in one dimension
def compute_next_shape(prev_shape, padding, dilation, kernel_size, stride):
    return (prev_shape + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
