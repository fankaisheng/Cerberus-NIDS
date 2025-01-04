from utils.utils import *



def get_cdf(prob_list):
    hist, bins = np.histogram(prob_list, bins=200, density=True)
    cumulative_hist = np.cumsum(hist * np.diff(bins))
    Y,X = cumulative_hist, bins
    c = 0
    for i in range(199):
        x = X[i+1] - X[i]
        y = Y[i]
        c += x*y
    return c


def get_c(c_user, c_victim=0.0243):
    # ResNet：0.0243
    # 
    c = (c_victim)/(c_user)
    if c > 1:
        c = 1
    return c


def Victim_C(victim_model, querry_dataloader, data_length):
    c_min = 1
    c_max = -1
    c_sum = 0
    num = 0
    for data, _ in querry_dataloader:
        if data.shape[0] != data_length:
            break
        label = torch.zeros(data.shape[0])
        querry_dataset = QUERRY_DATASET(data, label)
        mini_dataloader = DataLoader(querry_dataset, batch_size=100, shuffle=True)
        pred_list = np.zeros(0)
        for querry_data, _ in mini_dataloader:
            pred = F.softmax((victim_model(querry_data.to(device))),1)
            pred_prob, _ = torch.max(pred,1)
            pred_prob = pred_prob.cpu().detach().numpy()
            pred_list = np.concatenate((pred_list, pred_prob))

        c = get_cdf(pred_list)
        num += 1
        c_sum += c
        if c < c_min:
            c_min = c
        if c > c_max:
            c_max = c
    
    c_victim = c_min
    c_beta = c_min/c_max
    c_avg = c_sum/num 
    return round(c_victim,4), round(c_avg,4), round(c_beta,4)


# 单独测算C
def Pravicy_C(victim_model, querry_dataloader, data_length, c_victim, c_beta):
    num = 0
    detect_num = 0
    for querry_data, _ in querry_dataloader:
        if querry_data.shape[0] != data_length:
            break
        pred = F.softmax((victim_model(querry_data.to(device))),1)
        pred_prob, _ = torch.max(pred,1)
        pred_prob = pred_prob.cpu().detach().numpy()
        c_user = get_cdf(pred_prob)
        C = get_c(c_user, c_victim)
        if C < c_beta:
            detect_num += 1
        num += 1
    
    detect_rate = (detect_num / num)*100

    return detect_rate




def compute_RE(model, data):
    label = torch.zeros(data.shape[0])
    querry_dataset = QUERRY_DATASET(data, label)
    mini_dataloader = DataLoader(querry_dataset, batch_size=100, shuffle=True)
    criterion = nn.MSELoss()
    all_loss = 0
    num = 0
    for querry_data, _ in mini_dataloader:
        outputs = model(querry_data.to(device)).cpu().detach()
        loss = criterion(outputs, querry_data)
        all_loss += loss
        num += querry_data.shape[0]

    avg_loss = all_loss/num
    return avg_loss



def compute_R(r_user, r_victim):
    r = (r_victim)/(r_user)
    if r > 1:
        r = 1
    return r


def Victim_R(victim_model, querry_dataloader, data_length):
    r_min = 100
    r_max = -100
    r_sum = 0
    num = 0
    for data, _ in querry_dataloader:
        if data.shape[0] != data_length:
            break
        r_mini = compute_RE(victim_model, data)
        r_sum += r_mini
        num += 1
        if r_min > r_mini:
            r_min = r_mini
        if r_max < r_mini:
            r_max = r_mini
    r_min = r_min.detach().item()
    r_max = r_max.detach().item()
    r_sum = r_sum.detach().item()
    
    r_victim = r_min
    r_beta = r_min/r_max
    r_avg = r_sum/num 
    return round(r_victim,6), round(r_avg,6), round(r_beta,4)


# 单独测算R
def Pravicy_R(victim_model, querry_dataloader, data_length, r_victim, r_beta):
    num = 0
    detect_num = 0
    for querry_data, _ in querry_dataloader:
        if querry_data.shape[0] != data_length:
            break
        r_user = compute_RE(victim_model, querry_data)
        R = compute_R(r_user, r_victim)     
        if R < r_beta:
            detect_num += 1
        num += 1
    detect_rate = (detect_num / num)*100
    return detect_rate


def Pravicy_P(victim_model, AE, querry_dataloader, data_length,c_victim, r_victim, beta, a1=0.5):
    num = 0
    detect_num = 0
    a2 = 1-a1
    for querry_data, _ in querry_dataloader:
        if querry_data.shape[0] != data_length:
            break
        r_user = compute_RE(AE, querry_data)
        R = compute_R(r_user, r_victim) 

        pred = F.softmax((victim_model(querry_data.to(device))),1)
        pred_prob, _ = torch.max(pred,1)
        pred_prob = pred_prob.cpu().detach().numpy()
        c_user = get_cdf(pred_prob)
        C = get_c(c_user, c_victim)

        P = a1*C+a2*R
        if P < beta:
            detect_num += 1
        num += 1
    
    detect_rate = (detect_num / num)*100
    return detect_rate