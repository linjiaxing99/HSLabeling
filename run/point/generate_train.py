import sys
# print(sys.path)
sys.path.append('/home/isalab206/LJX/HSLabeling-master')
from options import *
from utils import *
from dataset import *
from HSLabeling import generate
from train import train_validate

print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device("cuda:0")
print('Device:', device)


def main():
    opt = Point_Options().parse()
    general_budget = 20000
    budget_part = general_budget / 4
    current_budget = 0
    current_surface_budget = 0
    current_line_budget = 0
    current_point_budget = 0
    while current_budget < general_budget:
        surface_budget, line_budget, point_budget = generate(opt, current_budget_part=current_budget, budget_part=budget_part)
        current_budget = current_budget + budget_part

        data_root = os.path.join(opt.data_root, opt.dataset)
        train_path = data_root + '/train/al_' + opt.data_label
        data_inform_path = os.path.join(opt.data_inform_path, opt.dataset)
        train_txt = open(data_inform_path + '/seg_train_' + opt.data_label + '.txt', 'w')
        list = os.listdir(train_path)
        for idx, i in enumerate(list):
            train_txt.write(i + '\n')
        train_txt.close()

        train_txt_name = 'seg_train_' + opt.data_label + '.txt'
        current_surface_budget = current_surface_budget + surface_budget
        current_line_budget = current_line_budget + line_budget
        current_point_budget = current_point_budget + point_budget
        mid = sorted([current_surface_budget, current_line_budget, current_point_budget])[1]
        surface_ratio = current_surface_budget / mid
        line_ratio = current_line_budget / mid
        point_ratio = current_point_budget / mid
        print('surface_ratio', surface_ratio)
        print('line_ratio', line_ratio)
        print('point_ratio', point_ratio)
        train_validate(opt, train_txt_name, current_budget, surface_ratio, line_ratio, point_ratio)
        print('surface_ratio', surface_ratio)
        print('line_ratio', line_ratio)
        print('point_ratio', point_ratio)

    # opt = Point_Options().parse()
    # general_budget = 8000
    # budget_part = general_budget / 4
    # current_budget = 2000
    # current_surface_budget = 667
    # current_line_budget = 667
    # current_point_budget = 667
    # while current_budget < general_budget:
    #     surface_budget, line_budget, point_budget = generate(opt, current_budget_part=current_budget,
    #                                                          budget_part=budget_part)
    #     current_budget = current_budget + budget_part
    #
    #     data_root = os.path.join(opt.data_root, opt.dataset)
    #     train_path = data_root + '/train/al_' + opt.data_label
    #     data_inform_path = os.path.join(opt.data_inform_path, opt.dataset)
    #     train_txt = open(data_inform_path + '/seg_train_' + opt.data_label + '.txt', 'w')
    #     list = os.listdir(train_path)
    #     for idx, i in enumerate(list):
    #         train_txt.write(i + '\n')
    #     train_txt.close()
    #
    #     train_txt_name = 'seg_train_' + opt.data_label + '.txt'
    #     current_surface_budget = current_surface_budget + surface_budget
    #     current_line_budget = current_line_budget + line_budget
    #     current_point_budget = current_point_budget + point_budget
    #     mid = sorted([current_surface_budget, current_line_budget, current_point_budget])[1]
    #     surface_ratio = current_surface_budget / mid
    #     line_ratio = current_line_budget / mid
    #     point_ratio = current_point_budget / mid
    #     print('surface_ratio', surface_ratio)
    #     print('line_ratio', line_ratio)
    #     print('point_ratio', point_ratio)
    #     train_validate(opt, train_txt_name, current_budget, surface_ratio, line_ratio, point_ratio)
    #     print('surface_ratio', surface_ratio)
    #     print('line_ratio', line_ratio)
    #     print('point_ratio', point_ratio)

    # opt = Point_Options().parse()
    # current_budget = 16002
    # train_txt_name = 'seg_train_' + opt.data_label + '.txt'
    # surface_ratio = 1
    # line_ratio = 0.0761
    # point_ratio = 1.3729
    # train_validate(opt, train_txt_name, current_budget, surface_ratio, line_ratio, point_ratio)



if __name__ == '__main__':
   main()