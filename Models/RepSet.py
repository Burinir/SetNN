import torch


# Parameters
# n_hidden_sets = 10 # number of hidden sets
# n_elements = 20 # cardinality of each hidden set
# d = 300 # dimension of each vector
# batch_size = 64  # batch size

# @inproceedings{skianis2020rep,
#   title={Rep the Set: Neural Networks for Learning Set Representations},
#   author={Skianis, Konstantinos and Nikolentzos, Giannis and Limnios, Stratis and Vazirgiannis, Michalis},
#   booktitle={Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics},
#   pages={1410--1420},
#   year={2020}
# }



class ApproxRepSet(torch.nn.Module):
    def __init__(self, n_hidden_sets, n_elements, d, n_out):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.Wc = torch.nn.parameter.Parameter(torch.Tensor(d, n_hidden_sets*n_elements))
        self.fc1 = torch.nn.Linear(n_hidden_sets, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, n_out)
        self.relu = torch.nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t,_ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.relu(self.fc1(t))
        t = self.relu(self.fc2(t))
        out = self.fc3(t)

        return out




class ApproxRepSetClassifier(torch.nn.Module):
    def __init__(self, n_hidden_sets, n_elements, d, n_classes):
        super(ApproxRepSetClassifier, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.Wc = torch.nn.parameter.Parameter(torch.Tensor(d, n_hidden_sets*n_elements))
        self.fc1 = torch.nn.Linear(n_hidden_sets, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, n_classes)
        self.relu = torch.nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t,_ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.relu(self.fc1(t))
        t = self.relu(self.fc2(t))
        out = self.fc3(t)

        return torch.nn.functional.log_softmax(out, dim=1)
