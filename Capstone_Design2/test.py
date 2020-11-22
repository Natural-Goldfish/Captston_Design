from src.dataset import ASPDataset
from src.model import ASPModel
from src.utils import *
from torch.utils.data import DataLoader
import argparse
import os

_CUDA_FLAG = torch.cuda.is_available()
_MODEL_PATH = "data\\models"
_MODEL_NAME = "ASPModel_{}_checkpoint.pth"
_INDEX = 0
_DATA_LOAD_PATH = "./data/processed/"
_FILE_TEST_NAME = "70man_test.csv"
_FILE_TRAIN_NAME = "70man_train.csv"

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True, help = "this is a csv file name to load")
    parser.add_argument("--index", type = int, required = True, help = "this is a index you will load in the test csv file")
    args = parser.parse_args()
    mode = ['train', 'test']

    with torch.no_grad():
        norm = Normalization()
        test_dataset = ASPDataset(mode = 'test')
        input_data, label = test_dataset[_INDEX]

        model = ASPModel()
        model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_NAME.format(args.model_name))))
        model.eval()

        if _CUDA_FLAG :
            model.cuda()
            input_data = input_data.cuda()
            label = label.cuda()

        output = model(input_data)
        prediction = norm.de_normalize(output).view(-1)
        #visual(input_data.cpu(), prediction.cpu(), label.cpu(), 'test')
        #checkingdiabete(input_data.cpu(), label.cpu(), 'test')
        find_meattime(input_data.cpu(), label.cpu(), 'test')
        """
        visual(input_data, prediction, label, mode)
        #1.당뇨 체크 기준 1 (혈당 200이상인 구간 존재유뮤)
        #checkingdiabete(input_data,label,mode)
            
        #2.당뇨 체크 기준 2 (식전 혈당과 식후 혈당 차이가 과다 유뮤)
        #find_meattime(input_data, label,mode)

        #3.당뇨 체크 기준 3 ( 잠들기 전에 혈당 수치 정도)
        #이건 이제 만들어야 함.
        """
        plt.show()
            

if __name__ == "__main__":
    test()