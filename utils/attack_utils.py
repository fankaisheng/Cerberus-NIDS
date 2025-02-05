from utils.utils import *


class ActiveThief():
    def __init__(self, pcap_dataset, class_num):
        self.class_num = class_num
        self.pcap_dataset = pcap_dataset
        self.pcap_dataloader = DataLoader(self.pcap_dataset, batch_size=1000,shuffle=True)
        self.data_num = self.pcap_dataset.items.shape[0]
        self.data_shape = self.pcap_dataset.items[0].shape
        self.dim = self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
    

    def benign_querry(self, generate_num):
        index_list = list(range(self.data_num))
        random.shuffle(index_list)
        select_index = index_list[:generate_num]
        return self.pcap_dataset.items[select_index]


    def target_querry(self, generate_num, target_label=0):
        select_data = torch.zeros(0,*self.data_shape)
        for data, label in self.pcap_dataloader:
            index = torch.nonzero(label == target_label).cpu().detach().reshape(-1).numpy().tolist()
            if len(index) > 0:
                select_data = torch.cat([select_data, data[index]], dim=0)
            if select_data.shape[0] > generate_num:
                break
        return select_data
    

    def noise_querry_normal(self, victim_model, generate_num=20000, target_label=0):
        target_data = torch.zeros(0,self.dim)
        target_num = 5000
        select_datas = torch.zeros(0,self.dim)
        for i in range(self.class_num):
            select_data = self.target_querry(target_num, target_label=i).reshape(-1, self.dim).detach()
            select_datas = torch.concat([select_datas, select_data])
        select_datas = select_datas.detach().numpy()
        mean = np.mean(select_datas, axis=0)
        std = np.std(select_datas, axis=0)
        noise_data = np.random.normal(loc=mean, scale=std, size=(generate_num, self.dim))
        noise_data = torch.tensor(np.clip(noise_data,a_min=0,a_max=1))
        target_data = torch.cat([target_data,noise_data],dim=0)
        target_data = target_data.reshape(-1,*self.data_shape).to(torch.float32)
        return target_data
    

    def noise_querry_random(self, generate_num):
        noise_data = torch.zeros(0,self.dim)
        num = int(generate_num/4)
        normal_samples = torch.tensor(np.clip(np.random.normal(loc=0.5, scale=0.2, size=(num, self.dim)),0,1))
        binomial_samples = torch.tensor(np.clip(np.random.normal(loc=0, scale=0.5, size=(num, self.dim)),0,1))
        uniform_samples = torch.tensor(np.random.uniform(low=0, high=255, size=(num, self.dim))/255) 
        exponential_samples = np.random.exponential(scale=1, size=(num, self.dim))
        exponential_samples = torch.tensor(exponential_samples/np.max(exponential_samples))

        # binomial_samples = torch.tensor(np.random.binomial(10, 0.5, size=(num, self.dim)))
        noise_data = torch.cat([noise_data, normal_samples],dim=0)
        noise_data = torch.cat([noise_data, uniform_samples],dim=0)
        noise_data = torch.cat([noise_data, exponential_samples],dim=0)
        noise_data = torch.cat([noise_data, binomial_samples],dim=0)

        index = list(range(noise_data.shape[0]))
        random.shuffle(index)
        noise_data = noise_data[index].reshape(-1,*self.data_shape).to(torch.float32)
        return noise_data


class JBDA():
    def __init__(self, MODEL, pcap_dataset, class_num=7, epoch=2):
        self.pcap_dataset = pcap_dataset
        self.data_num = self.pcap_dataset.items.shape[0]
        self.data_shape = self.pcap_dataset.items[0].shape
        self.dim = self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
        self.class_num = class_num
        self.clone_model = MODEL(class_num)
        self.optimizer = torch.optim.Adam(self.clone_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = epoch



    def noise_generate_init(self, generate_num):
        data_path = 'noise_data_normal.pt'
        label_path = 'noise_data_normal_label.pt'
        noise_data = torch.load(data_path).reshape(-1,1,32,32)
        noise_label = torch.load(label_path)
        select_index = []

        for i in range(self.class_num):
            index = torch.nonzero(noise_label == i).reshape(-1).numpy().tolist()
            random.shuffle(index)
            index = index[:generate_num]
            select_index += index
        
        noise_data = noise_data[select_index]
        return noise_data
    

    def real_generate_init(self, generate_num):
        data = self.pcap_dataset.items
        label = self.pcap_dataset.label

        select_index = []

        for i in range(self.class_num):
            index = torch.nonzero(label == i).reshape(-1).numpy().tolist()
            random.shuffle(index)
            index = index[:generate_num]
            select_index += index
        
        select_data = data[select_index]
        return select_data


    def train_clone(self, train_loader):
        self.clone_model = self.clone_model.to(device)
        self.clone_model.train()
        for _ in range(self.epoch):
            for data, label in train_loader:
                pred = self.clone_model(data.to(device))
                label = label.to(device)
                loss = self.criterion(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    

    def JBDA_augment(self, data, lmbda=0.1):
        optimizer = torch.optim.Adam(self.clone_model.parameters(), lr=0.03)
        data_aug = data.detach()
        self.clone_model = self.clone_model.to(device)
        for ind, x in enumerate(data):
            optimizer.zero_grad()
            x = x.reshape(-1,*self.data_shape).to(device)
            x_var = x.clone().requires_grad_(True)
            x_var.retain_grad()
            # x_var = x_var
            pred = torch.max(self.clone_model(x_var),1)[1]
            score = self.clone_model(x_var)[:, pred]
            score.backward()
            # x_var = x_var.cpu()
            # print(x_var.shape, x_var.grad)
            grad_val = x_var.grad.data.detach().cpu()
            x = x.detach().cpu()
            data_aug[ind] = x + lmbda * grad_val
        optimizer.zero_grad()
        return data_aug
    
    def JBDA_augment_batch(self, data, lmbda=0.1, batch=8):
        optimizer = torch.optim.Adam(self.clone_model.parameters(), lr=0.03)
        data_aug = data.detach()
        label_aug = torch.zeros(data_aug.shape[0])
        querry_dataset = QUERRY_DATASET(data_aug, label_aug)
        querry_dataloader = DataLoader(querry_dataset, batch_size=batch, shuffle=True)
        self.clone_model = self.clone_model.to(device)
        new_data = torch.zeros(0,1,32,32)

        for ind, D in enumerate(querry_dataloader):
            x, _ = D
            optimizer.zero_grad()
            x = x.reshape(-1,*self.data_shape).to(device)
            x_var = x.clone().requires_grad_(True)
            x_var.retain_grad()
            pred = self.clone_model(x_var)
            score = torch.max(pred,1)[1]
            loss = self.criterion(pred, score)
            loss.backward()
            grad_val = x_var.grad.data.detach().cpu()
            x_var = x_var.detach().cpu()
            new_data = torch.cat([new_data,(x_var + lmbda * grad_val)],dim=0)
            
        optimizer.zero_grad()
        return new_data
    

    def test_clone(self, test_loader, victim_model):
        victim_predictions = []
        clone_predictions = []
        all_true_labels = []
        self.clone_model = self.clone_model.cpu()
        self.clone_model.eval()
        for data, label in test_loader:
            clone_pred = torch.max(self.clone_model(data),1)[1]
            victim_pred = torch.max(victim_model(data),1)[1]
            victim_predictions.extend(victim_pred.numpy())
            all_true_labels.extend(label.numpy())
            clone_predictions.extend(clone_pred.numpy())
        real_accuracy = accuracy_score(all_true_labels, clone_predictions)
        # real_precision = precision_recall_fscore_support(all_true_labels, clone_predictions, average='weighted')[0]

        clone_accuracy = accuracy_score(victim_predictions, clone_predictions)
        # clone_precision = precision_recall_fscore_support(victim_predictions, clone_predictions, average='weighted')[0]

        return real_accuracy, clone_accuracy


class Knockoff():
    def __init__(self):
        self.data_shape = (1,32,32)
        
    def knockoff_cifar(self, generate_num):
        transform = transforms.Compose(
        [   transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        cifar_dataset = torchvision.datasets.CIFAR10(root='data', train=True,
                                            download=True,transform=transform)
        data_loader = DataLoader(cifar_dataset, batch_size=500, shuffle=True)
        select_data = torch.zeros(0,*self.data_shape)
        for data, _ in data_loader:
            if select_data.shape[0] > generate_num:
                break
            else:
                # random_number = random.randint(0, data.shape[1]-1)
                # querry_data = data[:, random_number, :, :].reshape(-1,*self.data_shape)
                # select_data = torch.cat([select_data, querry_data],dim=0)
                querry_data = data
                select_data = torch.cat([select_data, querry_data],dim=0)
        return select_data


    def knockoff_mnist(self, generate_num):
        transform = transforms.Compose([
            
            transforms.Resize((32, 32)),  # 将图像大小调整为 (32, 32)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 将图像转换为 PyTorch 张量
            ])
        MNIST_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        data_loader = DataLoader(MNIST_dataset, batch_size=500, shuffle=True)
        select_data = torch.zeros(0,*self.data_shape)
        for data, _ in data_loader:
            if select_data.shape[0] > generate_num:
                break
            else:
                # random_number = random.randint(0, data.shape[1]-1)
                querry_data = data
                select_data = torch.cat([select_data, querry_data],dim=0)
        
        return select_data
    
    def knockoff_malware(self, generate_num):
        file = "data/malanal.json"
        items = torch.zeros(0,1024)
        with open(file, 'r', encoding='utf-8') as json_file:
            buf = json.load(json_file)
            class_num = int(generate_num/len(buf.keys()))
            for i, key in enumerate(buf.keys()):
                data = torch.tensor(buf[key][:class_num]).reshape(-1,1024)
                items = torch.cat([items, data],dim=0)
        items = (items.reshape(-1,1,32,32)/ 255).to(torch.float32)
        return items



# def pairwise_distances(A, B):
#     na = torch.sum(A**2, dim=1, keepdim=True)
#     nb = torch.sum(B**2, dim=1, keepdim=True)
    
#     D = torch.sqrt(torch.clamp(na - 2 * torch.matmul(A, B.t()) + nb.t(), min=0.0))
#     # ||a-b||^2 = ||a||^2 - 2ab + ||b||^2
#     # 假设A有N个点， B有M个点，得到的距离矩阵D是M*N的。第i的第j个位置表示A中第i个点与B中第j个点的距离
#     return D

# class KCenter:
#     def __init__(self):
#         self.A = None
#         self.B = None

#     def compute(self, A, B):
#         self.A = A
#         self.B = B
        
#         D = pairwise_distances(self.A, self.B)
        
#         D_min = torch.min(D, dim=1)[0]
#         self.D_min_max = torch.max(D_min)
#         self.D_min_argmax = torch.argmax(D_min)


#     def get_subset(Y_vec, init_cluster, size, batch_size):
#         X = Y_vec
#         Y = init_cluster
        
#         batch_size = 100 * batch_size
#         n_batches = math.ceil(len(X) / float(batch_size))
        
#         m = KCenter()
        
#         points = []
        
#         for _ in range(size):
#             p = []
#             q = []
#             for i in range(n_batches):
#                 start = i * batch_size
#                 end = start + batch_size
#                 X_b = X[start:end]
#                 D_argmax_val, D_min_max_val = m.compute(X_b, Y)

#                 p.append(D_argmax_val)
#                 q.append(D_min_max_val)

#             b_indx = np.argmax(q)
#             indx = b_indx * batch_size + p[b_indx]
#             Y = torch.cat((Y, X[indx].unsqueeze(0)), 0)
            
#             points.append(indx)
            
#         return points